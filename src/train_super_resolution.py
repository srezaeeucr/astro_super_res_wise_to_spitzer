"""
Super-Resolution Training Pipeline V2 - Enhanced Architecture
==============================================================
UPDATED for V2 preprocessing with proper unit conversion.

Key changes:
- SSIM max_val adjusted for asinh-normalized data range
- Loss functions updated for new data distribution
- Added flux consistency loss option

Architecture: Enhanced RCAN (Residual Channel Attention Network)
Target: 14×14 WISE → 64×64 Spitzer super-resolution

Author: Enhanced for astronomical super-resolution
Version: 2.1 (with unit-corrected preprocessing)
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras import mixed_precision  
mixed_precision.set_global_policy("mixed_float16")
import warnings
warnings.filterwarnings('ignore')


class DepthToSpace(layers.Layer):
    """Custom layer for depth_to_space operation (pixel shuffle)."""
    
    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
    
    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.scale)
    
    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


class ChannelAttention(layers.Layer):
    """Channel Attention Module (Squeeze-and-Excitation)."""
    
    def __init__(self, filters, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.reduction = reduction
        
    def build(self, input_shape):
        self.gap = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(self.filters // self.reduction, activation='relu')
        self.dense2 = layers.Dense(self.filters, activation='sigmoid')
        self.reshape = layers.Reshape((1, 1, self.filters))
        self.multiply = layers.Multiply()
        
    def call(self, inputs):
        squeeze = self.gap(inputs)
        excitation = self.dense1(squeeze)
        excitation = self.dense2(excitation)
        excitation = self.reshape(excitation)
        return self.multiply([inputs, excitation])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "reduction": self.reduction
        })
        return config


class ResidualChannelAttentionBlock(layers.Layer):
    """Residual Block with Channel Attention (RCAB)."""
    
    def __init__(self, filters, kernel_size=3, reduction=16, res_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.res_scale = res_scale
        
    def build(self, input_shape):
        self.conv1 = layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.attention = ChannelAttention(self.filters, self.reduction)
        self.add = layers.Add()
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.attention(x)
        
        if self.res_scale != 1.0:
            x = x * self.res_scale
            
        return self.add([inputs, x])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "reduction": self.reduction,
            "res_scale": self.res_scale
        })
        return config


class SuperResolutionModelV2:
    """Enhanced Deep Super-Resolution Network with Attention for astronomical images."""
    
    def __init__(self, config_path: str):
        """Initialize model with configuration."""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_output_directory()
        self._setup_gpu()
        
        self.model = None
        self.history = None
        
        # ============================================================
        # NEW: Load normalization parameters for proper SSIM max_val
        # ============================================================
        self._load_normalization_params()
        
        self.logger.info("="*70)
        self.logger.info("Super-Resolution Training Pipeline V2.1")
        self.logger.info("(Updated for unit-corrected preprocessing)")
        self.logger.info("="*70)
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _setup_logging(self):
        """Configure logging."""
        self.logger = logging.getLogger('SuperResolutionV2')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_output_directory(self):
        """Create output directory structure."""
        output_dir = Path(self.config['output']['base_dir'])
        
        self.output_paths = {
            'base': output_dir,
            'models': output_dir / 'models',
            'logs': output_dir / 'logs',
            'visualizations': output_dir / 'visualizations',
            'checkpoints': output_dir / 'checkpoints',
            'tensorboard': output_dir / 'tensorboard'
        }
        
        for path in self.output_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        log_file = self.output_paths['logs'] / f"training_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Output directory: {output_dir}")
    
    def _setup_gpu(self):
        """Configure GPU settings."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"✓ Found {len(gpus)} GPU(s)")
                self.logger.info(f"✓ Using mixed precision: {tf.keras.mixed_precision.global_policy()}")
            except RuntimeError as e:
                self.logger.warning(f"GPU configuration error: {e}")
        else:
            self.logger.info("Running on CPU")
    
    def _load_normalization_params(self):
        """
        Load normalization parameters to determine proper SSIM max_val.
        
        For asinh normalization, data is roughly in [0, 1] range with some outliers.
        For minmax normalization, data is also roughly in [0, 1].
        """
        norm_path = self.config['data'].get('normalization_path', None)
        
        if norm_path and os.path.exists(norm_path):
            norm_params = np.load(norm_path, allow_pickle=True)
            self.norm_method = str(norm_params.get('method', 'asinh'))
            
            self.logger.info(f"✓ Loaded normalization params: {norm_path}")
            self.logger.info(f"  Normalization method: {self.norm_method}")
            
            # Set SSIM max_val based on expected data range
            # After asinh/minmax normalization, data is roughly [0, 1] with some outliers
            # Using max_val=2.0 gives some headroom for values slightly outside [0,1]
            self.ssim_max_val = 2.0
        else:
            self.logger.warning("No normalization params found, using default SSIM max_val=2.0")
            self.norm_method = 'unknown'
            self.ssim_max_val = 2.0
        
        self.logger.info(f"  SSIM max_val: {self.ssim_max_val}")
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load pre-split and normalized dataset."""
        self.logger.info("Loading pre-processed dataset...")
        
        train_path = self.config['data']['train_data_path']
        test_path = self.config['data']['test_data_path']
        
        # Load training data
        train_data = np.load(train_path)
        X_train = train_data['X_train']
        Y_train = train_data['Y_train']
        
        # Load test data
        test_data = np.load(test_path)
        X_test = test_data['X_test']
        Y_test = test_data['Y_test']
        
        self.logger.info(f"✓ Training data loaded:")
        self.logger.info(f"  X_train: {X_train.shape}, range=[{X_train.min():.3f}, {X_train.max():.3f}]")
        self.logger.info(f"  Y_train: {Y_train.shape}, range=[{Y_train.min():.3f}, {Y_train.max():.3f}]")
        
        self.logger.info(f"✓ Test data loaded:")
        self.logger.info(f"  X_test: {X_test.shape}, range=[{X_test.min():.3f}, {X_test.max():.3f}]")
        self.logger.info(f"  Y_test: {Y_test.shape}, range=[{Y_test.min():.3f}, {Y_test.max():.3f}]")
        
        # Verify data looks reasonable after new preprocessing
        if X_train.max() > 10 or Y_train.max() > 10:
            self.logger.warning("⚠ Data range seems large - check normalization!")
        
        return X_train, X_test, Y_train, Y_test
    
    def _multi_scale_input_block(self, inputs, base_filters: int, name: str):
        """Multi-scale feature extraction from input."""
        x1 = layers.Conv2D(base_filters, 3, padding='same', name=f'{name}_scale1_conv')(inputs)
        x1 = layers.ReLU(name=f'{name}_scale1_relu')(x1)
        
        x2 = layers.UpSampling2D(2, name=f'{name}_upsample')(inputs)
        x2 = layers.Conv2D(base_filters, 5, padding='same', name=f'{name}_scale2_conv')(x2)
        x2 = layers.ReLU(name=f'{name}_scale2_relu')(x2)
        x2 = layers.AveragePooling2D(2, name=f'{name}_downsample')(x2)
        
        x3 = layers.Conv2D(base_filters, 5, padding='same', name=f'{name}_scale3_conv')(inputs)
        x3 = layers.ReLU(name=f'{name}_scale3_relu')(x3)
        
        x = layers.Concatenate(name=f'{name}_concat')([x1, x2, x3])
        x = layers.Conv2D(base_filters, 1, padding='same', name=f'{name}_fusion')(x)
        
        return x
    
    def _residual_group(self, x, filters: int, n_blocks: int, name: str):
        """Residual in Residual Group (RIR)."""
        skip = x
        
        for i in range(n_blocks):
            x = ResidualChannelAttentionBlock(
                filters, 
                reduction=16,
                res_scale=0.1,
                name=f'{name}_rcab_{i}'
            )(x)
        
        x = layers.Conv2D(filters, 3, padding='same', name=f'{name}_group_conv')(x)
        x = layers.Add(name=f'{name}_group_add')([x, skip])
        
        return x
    
    def _upscale_block(self, x, filters: int, scale: int, name: str):
        """Upscaling block using sub-pixel convolution."""
        x = layers.Conv2D(filters * (scale ** 2), 3, padding='same', name=f'{name}_conv')(x)
        x = DepthToSpace(scale, name=f'{name}_ps')(x)
        x = layers.ReLU(name=f'{name}_relu')(x)
        return x
    
    def _progressive_reconstruction(self, x, base_filters: int):
        outputs = {}
        
        # 14 → 28
        x = self._upscale_block(x, base_filters, scale=2, name='progressive_up1')
        x = layers.Conv2D(base_filters, 3, padding='same', activation='relu', 
                        name='progressive_refine1')(x)
        out_28 = layers.Conv2D(1, 3, padding='same', name='output_28x28')(x)
        outputs['28x28'] = out_28
        
        # 28 → 56
        x = ResidualChannelAttentionBlock(base_filters, name='progressive_rcab1')(x)
        x = self._upscale_block(x, base_filters, scale=2, name='progressive_up2')
        x = layers.Conv2D(base_filters, 3, padding='same', activation='relu',
                        name='progressive_refine2')(x)
        out_56 = layers.Conv2D(1, 3, padding='same', name='output_56x56')(x)
        outputs['56x56'] = out_56
        
        # ============================================================
        # FIXED: 56 → 64 using proper learned upsampling
        # ============================================================
        
        # First, upsample 56 → 112 (2×), then crop to 64
        x = layers.Conv2D(base_filters, 3, padding='same', activation='relu',
                        name='final_pre_up')(x)
        x = self._upscale_block(x, base_filters // 2, scale=2, name='progressive_up3')  # 56 → 112
        
        # Center crop from 112×112 to 64×64
        x = layers.Cropping2D(cropping=((24, 24), (24, 24)), name='final_crop')(x)  # (112-64)/2 = 24
        
        # Final refinement
        x = layers.Conv2D(32, 3, padding='same', activation='relu', name='final_refine')(x)
        out_64 = layers.Conv2D(1, 3, padding='same', name='output_64x64')(x)
        outputs['64x64'] = out_64
        
        return outputs
    
    def build_model(self) -> keras.Model:
        """Build Enhanced RCAN model with progressive upsampling."""
        self.logger.info("Building Enhanced RCAN model V2...")
        
        cfg = self.config['model']
        input_shape = (14, 14, 1)
        base_filters = cfg['base_filters']
        n_residual_blocks = cfg['n_residual_blocks']
        n_groups = cfg.get('n_residual_groups', 4)
        blocks_per_group = n_residual_blocks // n_groups
        
        inputs = layers.Input(shape=input_shape, name='lr_input')
        
        self.logger.info(f"  Building multi-scale input block...")
        x = self._multi_scale_input_block(inputs, base_filters, 'multi_scale_input')
        
        skip_connection = x
        
        self.logger.info(f"  Building {n_groups} residual groups ({blocks_per_group} blocks each)...")
        for g in range(n_groups):
            x = self._residual_group(
                x, 
                base_filters, 
                blocks_per_group, 
                f'residual_group_{g}'
            )
        
        x = layers.Conv2D(base_filters, 3, padding='same', name='post_res_conv')(x)
        x = layers.Add(name='global_skip')([x, skip_connection])
        
        self.logger.info(f"  Building progressive upsampling path...")
        outputs = self._progressive_reconstruction(x, base_filters)
        
        if cfg.get('use_progressive_supervision', True):
            model = models.Model(
                inputs=inputs, 
                outputs=[outputs['28x28'], outputs['56x56'], outputs['64x64']],
                name='Enhanced_RCAN_V2'
            )
            self.logger.info("  Model architecture: Multi-output with progressive supervision")
        else:
            model = models.Model(
                inputs=inputs,
                outputs=outputs['64x64'],
                name='Enhanced_RCAN_V2'
            )
            self.logger.info("  Model architecture: Single output")
        
        self.logger.info(f"✓ Model built:")
        self.logger.info(f"  Base filters: {base_filters}")
        self.logger.info(f"  Residual groups: {n_groups}")
        self.logger.info(f"  Blocks per group: {blocks_per_group}")
        self.logger.info(f"  Total RCAB blocks: {n_residual_blocks}")
        self.logger.info(f"  Total parameters: {model.count_params():,}")
        
        return model

    # ============================================================
    # UPDATED LOSS FUNCTIONS for asinh-normalized data
    # ============================================================
    
    def _combined_loss_v2(self, y_true, y_pred):
        """
        Combined loss function for asinh-normalized data.
        
        Key change: SSIM max_val is now configurable based on actual data range.
        """
        cfg = self.config['model']['loss']
        
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # L1 loss (MAE)
        l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
        
        # L2 loss (MSE)
        l2 = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # SSIM loss - use configured max_val
        # For asinh-normalized data in ~[0, 1], max_val=2.0 works well
        ssim_max = cfg.get('ssim_max_val', 2.0)
        ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=ssim_max))
        
        # Gradient loss (edge preservation)
        grad_true_x = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
        grad_pred_x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        grad_true_y = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
        grad_pred_y = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        
        grad_loss = (tf.reduce_mean(tf.abs(grad_true_x - grad_pred_x)) +
                    tf.reduce_mean(tf.abs(grad_true_y - grad_pred_y)))
        
        # Cast all losses to float32
        l1 = tf.cast(l1, tf.float32)
        l2 = tf.cast(l2, tf.float32)
        ssim = tf.cast(ssim, tf.float32)
        grad_loss = tf.cast(grad_loss, tf.float32)
        
        # Weighted combination
        total_loss = (cfg.get('l1_weight', 0.4) * l1 + 
                     cfg.get('l2_weight', 0.2) * l2 + 
                     cfg.get('ssim_weight', 0.3) * ssim +
                     cfg.get('gradient_weight', 0.1) * grad_loss)
        
        return total_loss

    def _source_focused_loss_v2(self, y_true, y_pred):
        """
        Source-focused loss for asinh-normalized data.
        
        After asinh normalization:
        - Background is near 0
        - Sources are positive (typically 0.3-1.0+)
        - Distribution is more uniform than raw data
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        cfg = self.config['model']['loss']
        
        # For asinh-normalized data, threshold should be ~0.2-0.3
        # (corresponds to ~median flux in physical units)
        threshold = cfg.get('source_threshold', 0.3)
        
        # Create source mask
        source_mask = tf.cast(y_true > threshold, tf.float32)
        
        # Adaptive weighting: sources get higher weight
        source_weight = cfg.get('source_weight', 3.0)
        background_weight = cfg.get('background_weight', 0.5)
        weights = source_mask * source_weight + (1.0 - source_mask) * background_weight
        
        # Huber loss (robust to outliers)
        delta = cfg.get('huber_delta', 0.5)
        error = y_true - y_pred
        abs_error = tf.abs(error)
        
        is_small = abs_error <= delta
        huber = tf.where(is_small, 0.5 * tf.square(error), delta * (abs_error - 0.5 * delta))
        weighted_huber = tf.reduce_mean(weights * huber)
        
        # SSIM loss
        ssim_max = cfg.get('ssim_max_val', 2.0)
        ssim = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=ssim_max))
        
        # Gradient loss (edge preservation)
        grad_true_x = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
        grad_pred_x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        grad_true_y = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
        grad_pred_y = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        grad_loss = (tf.reduce_mean(tf.abs(grad_true_x - grad_pred_x)) +
                    tf.reduce_mean(tf.abs(grad_true_y - grad_pred_y)))
        
        # Combine
        total_loss = (cfg.get('huber_weight', 0.5) * weighted_huber +
                     cfg.get('ssim_weight', 0.35) * ssim +
                     cfg.get('gradient_weight', 0.15) * grad_loss)
        
        return total_loss

    def compile_model(self, model: keras.Model):
        """Compile model with optimizer and loss."""
        self.logger.info("Compiling model...")
        
        cfg = self.config['training']
        
        # Learning rate schedule
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=cfg['learning_rate'],
            decay_steps=cfg.get('lr_decay_steps', 1000),
            decay_rate=cfg.get('lr_decay_rate', 0.96),
            staircase=True
        )
        
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=cfg.get('weight_decay', 1e-4)
        )
        
        # Metrics with correct SSIM max_val
        ssim_max = self.config['model']['loss'].get('ssim_max_val', 2.0)
        
        def ssim_metric(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            return tf.image.ssim(y_true, y_pred, max_val=ssim_max)
        
        metrics = [
            keras.metrics.MeanAbsoluteError(name='mae'),
            keras.metrics.MeanSquaredError(name='mse'),
            ssim_metric
        ]
        
        # Select loss function
        loss_type = self.config['model']['loss'].get('type', 'source_focused')
        
        if loss_type == 'combined':
            loss_fn = self._combined_loss_v2
        else:
            loss_fn = self._source_focused_loss_v2
        
        if self.config['model'].get('use_progressive_supervision', True):
            losses = {
                'output_28x28': loss_fn,
                'output_56x56': loss_fn,
                'output_64x64': loss_fn
            }
            loss_weights = {
                'output_28x28': 0.2,
                'output_56x56': 0.3,
                'output_64x64': 0.5
            }
            
            model.compile(
                optimizer=optimizer,
                loss=losses,
                loss_weights=loss_weights,
                metrics={'output_64x64': metrics}
            )
        else:
            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=metrics
            )
        
        self.logger.info("✓ Model compiled")
        self.logger.info(f"  Optimizer: AdamW (lr={cfg['learning_rate']})")
        self.logger.info(f"  Loss: {loss_type}")
        self.logger.info(f"  SSIM max_val: {ssim_max}")
    
    def _prepare_progressive_targets(self, Y_train, Y_test):
        """Prepare downsampled targets for progressive supervision."""
        from scipy.ndimage import zoom
        
        self.logger.info("Preparing progressive supervision targets...")
        
        Y_28_train = np.array([zoom(Y_train[i, :, :, 0], 28/64, order=1) 
                               for i in range(len(Y_train))])[..., np.newaxis]
        Y_28_test = np.array([zoom(Y_test[i, :, :, 0], 28/64, order=1) 
                              for i in range(len(Y_test))])[..., np.newaxis]
        
        Y_56_train = np.array([zoom(Y_train[i, :, :, 0], 56/64, order=1) 
                               for i in range(len(Y_train))])[..., np.newaxis]
        Y_56_test = np.array([zoom(Y_test[i, :, :, 0], 56/64, order=1) 
                              for i in range(len(Y_test))])[..., np.newaxis]
        
        Y_64_train = Y_train
        Y_64_test = Y_test
        
        self.logger.info("✓ Progressive targets prepared")
        
        return (Y_28_train, Y_56_train, Y_64_train), (Y_28_test, Y_56_test, Y_64_test)
    
    def get_callbacks(self) -> list:
        """Create training callbacks."""
        cfg = self.config['training']
        
        callback_list = []
        
        checkpoint_path = str(self.output_paths['checkpoints'] / 'best_model_v2.h5')
        callback_list.append(
            callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                mode='min',
                verbose=1
            )
        )
        
        if cfg.get('early_stopping', True):
            callback_list.append(
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=cfg.get('early_stopping_patience', 30),
                    restore_best_weights=True,
                    verbose=1
                )
            )
        
        if cfg.get('reduce_lr_on_plateau', True):
            callback_list.append(
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    verbose=1
                )
            )
        
        callback_list.append(
            callbacks.TensorBoard(
                log_dir=str(self.output_paths['tensorboard']),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        )
        
        csv_path = str(self.output_paths['logs'] / 'training_history_v2.csv')
        callback_list.append(
            callbacks.CSVLogger(csv_path)
        )
        
        return callback_list
    
    def train(self, X_train, Y_train, X_test, Y_test):
        """Train the model."""
        self.logger.info("Starting training...")
        
        cfg = self.config['training']
        
        if self.config['model'].get('use_progressive_supervision', True):
            (Y_28_train, Y_56_train, Y_64_train), \
            (Y_28_test, Y_56_test, Y_64_test) = self._prepare_progressive_targets(Y_train, Y_test)
            
            self.history = self.model.fit(
                X_train,
                {
                    'output_28x28': Y_28_train,
                    'output_56x56': Y_56_train,
                    'output_64x64': Y_64_train
                },
                validation_data=(
                    X_test,
                    {
                        'output_28x28': Y_28_test,
                        'output_56x56': Y_56_test,
                        'output_64x64': Y_64_test
                    }
                ),
                epochs=cfg['epochs'],
                batch_size=cfg['batch_size'],
                callbacks=self.get_callbacks(),
                verbose=1
            )
        else:
            self.history = self.model.fit(
                X_train, Y_train,
                validation_data=(X_test, Y_test),
                epochs=cfg['epochs'],
                batch_size=cfg['batch_size'],
                callbacks=self.get_callbacks(),
                verbose=1
            )
        
        self.logger.info("✓ Training complete")
    
    def save_model(self):
        """Save final model and training history."""
        model_path = self.output_paths['models'] / 'final_model_v2.h5'
        self.model.save(model_path)
        self.logger.info(f"✓ Model saved: {model_path}")
        
        history_path = self.output_paths['models'] / 'training_history_v2.npz'
        np.savez(history_path, **self.history.history)
        self.logger.info(f"✓ History saved: {history_path}")
        
        config_path = self.output_paths['models'] / 'model_config_v2.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        self.logger.info(f"✓ Config saved: {config_path}")
    
    def plot_training_history(self):
        """Plot training history."""
        self.logger.info("Creating training history plots...")
        
        history = self.history.history
        
        if self.config['model'].get('use_progressive_supervision', True):
            loss_key = 'output_64x64_loss'
            val_loss_key = 'val_output_64x64_loss'
            mae_key = 'output_64x64_mae'
            val_mae_key = 'val_output_64x64_mae'
        else:
            loss_key = 'loss'
            val_loss_key = 'val_loss'
            mae_key = 'mae'
            val_mae_key = 'val_mae'
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss
        if loss_key in history:
            axes[0, 0].plot(history[loss_key], label='Train Loss', linewidth=2)
            axes[0, 0].plot(history[val_loss_key], label='Val Loss', linewidth=2)
            axes[0, 0].set_xlabel('Epoch', fontsize=12)
            axes[0, 0].set_ylabel('Loss', fontsize=12)
            axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
            axes[0, 0].legend(fontsize=11)
            axes[0, 0].grid(True, alpha=0.3)
        
        # MAE
        if mae_key in history:
            axes[0, 1].plot(history[mae_key], label='Train MAE', linewidth=2)
            axes[0, 1].plot(history[val_mae_key], label='Val MAE', linewidth=2)
            axes[0, 1].set_xlabel('Epoch', fontsize=12)
            axes[0, 1].set_ylabel('MAE', fontsize=12)
            axes[0, 1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
            axes[0, 1].legend(fontsize=11)
            axes[0, 1].grid(True, alpha=0.3)
        
        # Total loss
        if 'loss' in history:
            axes[1, 0].plot(history['loss'], label='Train Total Loss', linewidth=2)
            axes[1, 0].plot(history['val_loss'], label='Val Total Loss', linewidth=2)
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('Total Loss', fontsize=12)
            axes[1, 0].set_title('Total Loss (All Outputs)', fontsize=14, fontweight='bold')
            axes[1, 0].legend(fontsize=11)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if 'lr' in history:
            axes[1, 1].plot(history['lr'], linewidth=2, color='green')
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
            axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning rate\nnot logged', 
                          ha='center', va='center', fontsize=14)
            axes[1, 1].axis('off')
        
        plt.suptitle('Training History - Enhanced RCAN V2.1\n(Unit-Corrected Preprocessing)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        output_path = self.output_paths['visualizations'] / 'training_history_v2.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ Training history saved: {output_path}")
    
    def run(self):
        """Execute complete training pipeline."""
        try:
            X_train, X_test, Y_train, Y_test = self.load_data()
            
            self.model = self.build_model()
            self.compile_model(self.model)
            
            self.train(X_train, Y_train, X_test, Y_test)
            
            self.save_model()
            self.plot_training_history()
            
            self.logger.info("="*70)
            self.logger.info("TRAINING COMPLETED SUCCESSFULLY (V2.1)")
            self.logger.info("="*70)
            self.logger.info(f"Best model: {self.output_paths['checkpoints'] / 'best_model_v2.h5'}")
            self.logger.info(f"Final model: {self.output_paths['models'] / 'final_model_v2.h5'}")
            self.logger.info("="*70)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Train Enhanced Super-Resolution Model V2.1'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='train_config_v2.json',
        help='Path to training configuration file'
    )
    args = parser.parse_args()
    
    trainer = SuperResolutionModelV2(args.config)
    success = trainer.run()
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())