"""
Super-Resolution Model Evaluation V2
=====================================
UPDATED for V2 preprocessing with proper unit conversion.

Key changes:
- Proper denormalization for asinh or minmax normalized data
- Converts predictions back to physical units (MJy/sr)
- Updated metrics for astronomical data quality assessment

Author: Updated for unit-corrected preprocessing
Version: 2.1
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from tensorflow.keras.optimizers import AdamW
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Custom layers (must match training script)
# ============================================================

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
    """Channel Attention Module."""
    
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
    """Residual Block with Channel Attention."""
    
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


# ============================================================
# Denormalization functions
# ============================================================

def denormalize_asinh(data: np.ndarray, norm_params: dict) -> np.ndarray:
    """
    Inverse asinh transform to recover physical units (MJy/sr).
    
    Parameters:
    -----------
    data : array
        Normalized data (e.g., model output)
    norm_params : dict
        Normalization parameters from preprocessing
        
    Returns:
    --------
    Physical flux in MJy/sr
    """
    # Step 1: Undo percentile scaling
    Y_p01 = norm_params['Y_p01']
    Y_p99 = norm_params['Y_p99']
    data_asinh = data * (Y_p99 - Y_p01) + Y_p01
    
    # Step 2: Inverse asinh: sinh(x) * softening
    Y_soft = norm_params['Y_soft']
    data_physical = np.sinh(data_asinh) * Y_soft
    
    return data_physical


def denormalize_minmax(data: np.ndarray, norm_params: dict) -> np.ndarray:
    """
    Inverse min-max scaling to recover physical units (MJy/sr).
    """
    Y_min = norm_params['Y_min']
    Y_max = norm_params['Y_max']
    return data * (Y_max - Y_min) + Y_min


def denormalize(data: np.ndarray, norm_params: dict) -> np.ndarray:
    """
    Auto-select denormalization method based on norm_params.
    """
    method = norm_params.get('method', 'asinh')
    
    if method == 'asinh':
        return denormalize_asinh(data, norm_params)
    elif method == 'minmax':
        return denormalize_minmax(data, norm_params)
    else:
        # Fallback for old-style params (mean/std normalization)
        if 'Y_mean' in norm_params and 'Y_std' in norm_params:
            return data * norm_params['Y_std'] + norm_params['Y_mean']
        else:
            raise ValueError(f"Unknown normalization method: {method}")


class SuperResolutionEvaluator:
    """Evaluate super-resolution model performance with proper denormalization."""
    
    def __init__(self, config_path: str):
        """Initialize evaluator."""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_output_directory()
        
        self.logger.info("="*70)
        self.logger.info("Super-Resolution Model Evaluation V2.1")
        self.logger.info("(Updated for unit-corrected preprocessing)")
        self.logger.info("="*70)
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _setup_logging(self):
        """Setup logging."""
        self.logger = logging.getLogger('Evaluator')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_output_directory(self):
        """Create output directories."""
        output_dir = Path(self.config['output']['eval_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        
        self.logger.info(f"Output directory: {output_dir}")
    
    def load_model_and_data(self):
        """Load trained model and pre-split test data."""
        self.logger.info("Loading model and data...")
        
        # Load model with custom objects
        model_path = self.config['model']['model_path']
        self.model = keras.models.load_model(
            model_path,
            custom_objects={
                'DepthToSpace': DepthToSpace,
                'ChannelAttention': ChannelAttention,
                'ResidualChannelAttentionBlock': ResidualChannelAttentionBlock,
                'AdamW': AdamW,
                '_combined_loss': self._combined_loss,
                '_combined_loss_v2': self._combined_loss_v2,
                '_source_focused_loss_v2': self._source_focused_loss_v2,
                '_ssim_metric': self._ssim_metric,
                'ssim_metric': self._ssim_metric
            },
            compile=False,
            safe_mode=False
        )
        self.logger.info(f"✓ Model loaded from: {model_path}")
        
        # Load test data
        test_data_path = self.config['data']['test_data_path']
        test_data = np.load(test_data_path)
        
        self.X_test = test_data['X_test']
        self.Y_test = test_data['Y_test']
        self.ra_test = test_data['ra_test']
        self.dec_test = test_data['dec_test']
        
        self.logger.info(f"✓ Test data loaded: {self.X_test.shape[0]} samples")
        self.logger.info(f"  X range: [{self.X_test.min():.4f}, {self.X_test.max():.4f}]")
        self.logger.info(f"  Y range: [{self.Y_test.min():.4f}, {self.Y_test.max():.4f}]")
        
        # ============================================================
        # Load normalization parameters (NEW!)
        # ============================================================
        norm_path = self.config['model']['normalization_path']
        norm_data = np.load(norm_path, allow_pickle=True)
        
        # Convert to dict
        self.norm_params = {}
        for key in norm_data.files:
            val = norm_data[key]
            # Handle scalar arrays
            if val.ndim == 0:
                self.norm_params[key] = val.item()
            else:
                self.norm_params[key] = val
        
        self.norm_method = self.norm_params.get('method', 'unknown')
        self.logger.info(f"✓ Normalization params loaded: {norm_path}")
        self.logger.info(f"  Method: {self.norm_method}")
        
        # Determine SSIM max_val based on data range
        self.ssim_max_val = 2.0  # Default for [0, 1] normalized data
        self.logger.info(f"  SSIM max_val: {self.ssim_max_val}")
    
    # ============================================================
    # Loss functions (for model loading)
    # ============================================================
    
    @staticmethod
    def _combined_loss(y_true, y_pred):
        """Combined loss for V1 model."""
        l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
        l2 = tf.reduce_mean(tf.square(y_true - y_pred))
        ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=2.0))
        return 0.5 * l1 + 0.3 * l2 + 0.2 * ssim
    
    @staticmethod
    def _combined_loss_v2(y_true, y_pred):
        """Enhanced combined loss for V2 model."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
        l2 = tf.reduce_mean(tf.square(y_true - y_pred))
        ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=2.0))
        
        grad_true_x = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
        grad_pred_x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        grad_true_y = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
        grad_pred_y = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        
        grad_loss = (tf.reduce_mean(tf.abs(grad_true_x - grad_pred_x)) +
                    tf.reduce_mean(tf.abs(grad_true_y - grad_pred_y)))
        
        return 0.4 * l1 + 0.2 * l2 + 0.3 * ssim + 0.1 * grad_loss
    
    @staticmethod
    def _source_focused_loss_v2(y_true, y_pred):
        """Source-focused loss for V2.1 model."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        threshold = 0.3
        source_mask = tf.cast(y_true > threshold, tf.float32)
        weights = source_mask * 3.0 + (1.0 - source_mask) * 0.5
        
        delta = 0.5
        error = y_true - y_pred
        abs_error = tf.abs(error)
        is_small = abs_error <= delta
        huber = tf.where(is_small, 0.5 * tf.square(error), delta * (abs_error - 0.5 * delta))
        weighted_huber = tf.reduce_mean(weights * huber)
        
        ssim = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=2.0))
        
        grad_true_x = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
        grad_pred_x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        grad_true_y = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
        grad_pred_y = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        grad_loss = (tf.reduce_mean(tf.abs(grad_true_x - grad_pred_x)) +
                    tf.reduce_mean(tf.abs(grad_true_y - grad_pred_y)))
        
        return 0.5 * weighted_huber + 0.35 * ssim + 0.15 * grad_loss
    
    @staticmethod
    def _ssim_metric(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.image.ssim(y_true, y_pred, max_val=2.0)
    
    def compute_metrics(self, y_true, y_pred) -> Dict:
        """Compute metrics on normalized data."""
        self.logger.info("Computing metrics...")
        
        metrics = {}
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Basic error metrics
        metrics['mse'] = mean_squared_error(y_true_flat, y_pred_flat)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true_flat, y_pred_flat)
        
        # Normalized metrics
        y_range = y_true_flat.max() - y_true_flat.min()
        metrics['nmse'] = metrics['mse'] / (y_range ** 2)
        metrics['nrmse'] = metrics['rmse'] / y_range
        
        # SSIM
        ssim_values = []
        for i in range(len(y_true)):
            ssim_val = tf.image.ssim(
                y_true[i:i+1], y_pred[i:i+1], 
                max_val=self.ssim_max_val
            ).numpy()[0]
            ssim_values.append(ssim_val)
        metrics['ssim'] = np.mean(ssim_values)
        metrics['ssim_std'] = np.std(ssim_values)
        
        # Correlation
        metrics['pearson_r'], metrics['pearson_p'] = pearsonr(y_true_flat, y_pred_flat)
        metrics['mean_relative_error'] = np.mean(
            np.abs((y_true_flat - y_pred_flat) / (np.abs(y_true_flat) + 1e-8))
        )
        
        return metrics
    
    def compute_source_metrics(self, y_true, y_pred) -> Dict:
        """
        Compute metrics that focus on source regions.
        
        For asinh-normalized data, sources are typically > 0.3
        """
        metrics = {}
        
        # Threshold for sources (adjusted for normalized data)
        threshold = 0.3
        
        # Source-only PSNR
        source_psnrs = []
        for i in range(len(y_true)):
            source_mask = y_true[i] > threshold
            
            if source_mask.sum() > 0:
                true_sources = y_true[i][source_mask]
                pred_sources = y_pred[i][source_mask]
                
                mse = np.mean((true_sources - pred_sources) ** 2)
                if mse > 0:
                    psnr = 20 * np.log10(true_sources.max() / np.sqrt(mse))
                    source_psnrs.append(psnr)
        
        metrics['source_psnr_mean'] = np.mean(source_psnrs) if source_psnrs else 0
        metrics['source_psnr_median'] = np.median(source_psnrs) if source_psnrs else 0
        
        # Peak flux accuracy
        peak_errors = []
        for i in range(len(y_true)):
            true_peak = y_true[i].max()
            pred_peak = y_pred[i].max()
            relative_error = abs(true_peak - pred_peak) / (true_peak + 1e-8)
            peak_errors.append(relative_error)
        
        metrics['peak_flux_error_mean'] = np.mean(peak_errors)
        metrics['peak_flux_error_median'] = np.median(peak_errors)
        
        # Integrated flux error
        flux_errors = []
        for i in range(len(y_true)):
            source_mask = y_true[i] > threshold
            
            if source_mask.sum() > 0:
                true_flux = y_true[i][source_mask].sum()
                pred_flux = y_pred[i][source_mask].sum()
                relative_error = abs(true_flux - pred_flux) / (true_flux + 1e-8)
                flux_errors.append(relative_error)
        
        metrics['integrated_flux_error_mean'] = np.mean(flux_errors) if flux_errors else 0
        metrics['integrated_flux_error_median'] = np.median(flux_errors) if flux_errors else 0
        
        # Contrast ratio
        contrast_ratios_true = []
        contrast_ratios_pred = []
        
        for i in range(len(y_true)):
            source_mask = y_true[i] > threshold
            
            if source_mask.sum() > 0 and (~source_mask).sum() > 0:
                true_signal = y_true[i][source_mask].mean()
                true_background = y_true[i][~source_mask].std()
                true_contrast = true_signal / (true_background + 1e-8)
                
                pred_signal = y_pred[i][source_mask].mean()
                pred_background = y_pred[i][~source_mask].std()
                pred_contrast = pred_signal / (pred_background + 1e-8)
                
                contrast_ratios_true.append(true_contrast)
                contrast_ratios_pred.append(pred_contrast)
        
        metrics['contrast_ratio_true'] = np.mean(contrast_ratios_true) if contrast_ratios_true else 0
        metrics['contrast_ratio_pred'] = np.mean(contrast_ratios_pred) if contrast_ratios_pred else 0
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print global metrics."""
        self.logger.info("\n" + "="*70)
        self.logger.info("EVALUATION METRICS (Normalized Space)")
        self.logger.info("="*70)
        self.logger.info(f"{'Metric':<30} {'Value':>20}")
        self.logger.info("-"*70)
        self.logger.info(f"{'Mean Squared Error (MSE)':<30} {metrics['mse']:>20.6e}")
        self.logger.info(f"{'Root MSE (RMSE)':<30} {metrics['rmse']:>20.6e}")
        self.logger.info(f"{'Mean Absolute Error (MAE)':<30} {metrics['mae']:>20.6e}")
        self.logger.info(f"{'Normalized RMSE':<30} {metrics['nrmse']:>20.6f}")
        self.logger.info(f"{'SSIM':<30} {metrics['ssim']:>20.4f}")
        self.logger.info(f"{'SSIM Std Dev':<30} {metrics['ssim_std']:>20.4f}")
        self.logger.info(f"{'Pearson Correlation':<30} {metrics['pearson_r']:>20.4f}")
        self.logger.info("="*70)
    
    def print_comprehensive_metrics(self, global_metrics: Dict, source_metrics: Dict):
        """Print both global and source-focused metrics."""
        
        print("\n" + "="*70)
        print("COMPREHENSIVE EVALUATION METRICS")
        print("="*70)
        
        print("\nGLOBAL METRICS (entire image, including background):")
        print("-"*70)
        print(f"  SSIM:               {global_metrics['ssim']:.4f}")
        print(f"  MAE:                {global_metrics['mae']:.6e}")
        print(f"  Pearson r:          {global_metrics['pearson_r']:.4f}")
        
        print("\nSOURCE-FOCUSED METRICS (astronomical quality):")
        print("-"*70)
        print(f"  Source PSNR (mean): {source_metrics['source_psnr_mean']:.2f} dB")
        print(f"  Source PSNR (med):  {source_metrics['source_psnr_median']:.2f} dB")
        print(f"  Peak Flux Error:    {source_metrics['peak_flux_error_mean']:.2%}")
        print(f"  Integrated Flux Err: {source_metrics['integrated_flux_error_mean']:.2%}")
        print(f"  True Contrast:      {source_metrics['contrast_ratio_true']:.1f}")
        print(f"  Pred Contrast:      {source_metrics['contrast_ratio_pred']:.1f}")
        
        print("\nINTERPRETATION:")
        print("-"*70)
        
        if source_metrics['source_psnr_mean'] > 25:
            print("  ✓ Source reconstruction is EXCELLENT")
        elif source_metrics['source_psnr_mean'] > 20:
            print("  ✓ Source reconstruction is GOOD")
        elif source_metrics['source_psnr_mean'] > 15:
            print("  ⚠ Source reconstruction is MEDIOCRE")
        else:
            print("  ✗ Source reconstruction is POOR")
        
        if source_metrics['peak_flux_error_mean'] < 0.1:
            print("  ✓ Peak flux accuracy is EXCELLENT (<10% error)")
        elif source_metrics['peak_flux_error_mean'] < 0.2:
            print("  ✓ Peak flux accuracy is GOOD (<20% error)")
        elif source_metrics['peak_flux_error_mean'] < 0.5:
            print("  ⚠ Peak flux accuracy is MEDIOCRE (20-50% error)")
        else:
            print("  ✗ Peak flux accuracy is POOR (>50% error)")
        
        if source_metrics['contrast_ratio_true'] > 0:
            contrast_ratio = source_metrics['contrast_ratio_pred'] / source_metrics['contrast_ratio_true']
            if 0.8 < contrast_ratio < 1.2:
                print("  ✓ Contrast preservation is GOOD")
            elif 0.5 < contrast_ratio < 1.5:
                print("  ⚠ Contrast preservation is MEDIOCRE")
            else:
                print("  ✗ Contrast preservation is POOR")
        
        print("="*70)
    
    def create_comparison_grid(self, n_samples: int = 10):
        """Create comparison grid with proper denormalization."""
        self.logger.info(f"Creating comparison grid ({n_samples} samples)...")
        
        # Get model predictions
        Y_pred = self.model.predict(self.X_test, verbose=0)
        
        if isinstance(Y_pred, list):
            self.logger.info(f"  Multi-output model detected ({len(Y_pred)} outputs)")
            Y_pred = Y_pred[-1]
        
        if Y_pred.dtype == np.float16:
            Y_pred = Y_pred.astype(np.float32)
        
        # Select random samples
        np.random.seed(42)
        indices = np.random.choice(len(self.X_test), n_samples, replace=False)
        
        # ============================================================
        # Denormalize to physical units (MJy/sr)
        # ============================================================
        Y_pred_physical = denormalize(Y_pred[indices], self.norm_params)
        Y_true_physical = denormalize(self.Y_test[indices], self.norm_params)
        X_test_physical = denormalize(self.X_test[indices], self.norm_params)  # Using Y params as approximation
        
        self.logger.info(f"  Denormalized to physical units (MJy/sr)")
        self.logger.info(f"    Pred range: [{Y_pred_physical.min():.4e}, {Y_pred_physical.max():.4e}]")
        self.logger.info(f"    True range: [{Y_true_physical.min():.4e}, {Y_true_physical.max():.4e}]")
        
        # Create figure
        row_height = 5
        fig = plt.figure(figsize=(20, row_height * n_samples))
        
        for i in range(n_samples):
            idx = indices[i]
            
            # Get images (denormalized to MJy/sr)
            x_img = X_test_physical[i, :, :, 0]
            y_true_img = Y_true_physical[i, :, :, 0]
            y_pred_img = Y_pred_physical[i, :, :, 0]
            error_map = np.abs(y_true_img - y_pred_img)
            
            # Compute metrics on normalized data
            y_true_norm = self.Y_test[idx, :, :, 0]
            y_pred_norm = Y_pred[idx, :, :, 0]
            
            ssim_val = tf.image.ssim(
                self.Y_test[idx:idx+1], Y_pred[idx:idx+1], 
                max_val=self.ssim_max_val
            ).numpy()[0]
            mae_val = np.mean(np.abs(y_true_norm - y_pred_norm))
            
            # Create row
            row_fraction = 1.0 / n_samples
            top_pos = 0.97 - i * row_fraction
            bottom_pos = top_pos - (row_fraction * 0.85)
            
            gs = GridSpec(1, 4, figure=fig, left=0.05, right=0.95, 
                         top=top_pos, bottom=bottom_pos, wspace=0.3)
            
            # WISE Input
            ax1 = fig.add_subplot(gs[0])
            vmin, vmax = np.percentile(x_img, [1, 99])
            im1 = ax1.imshow(x_img, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
            ax1.set_title(f'WISE Input (14×14)', 
                         fontsize=10, fontweight='bold')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            
            # Model Output
            ax2 = fig.add_subplot(gs[1])
            vmin, vmax = np.percentile(y_pred_img, [1, 99])
            im2 = ax2.imshow(y_pred_img, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
            ax2.set_title(f'Model Output (64×64)', 
                         fontsize=10, fontweight='bold')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            
            # Ground Truth
            ax3 = fig.add_subplot(gs[2])
            vmin, vmax = np.percentile(y_true_img, [1, 99])
            im3 = ax3.imshow(y_true_img, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
            ax3.set_title(f'Spitzer Truth (64×64)]', 
                         fontsize=10, fontweight='bold')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            
            # Error Map
            ax4 = fig.add_subplot(gs[3])
            #vmin, vmax = np.percentile(error_map, [1, 99])
            im4 = ax4.imshow(error_map, cmap='hot', vmin=0, vmax=vmax, origin='lower')
            ax4.set_title(f'Absolute Error', 
                         fontsize=10, fontweight='bold')
            ax4.axis('off')
            plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
            
            # Metrics text
            metrics_text = f'Sample {i+1} | SSIM: {ssim_val:.4f} | MAE: {mae_val:.4e}'
            fig.text(0.5, top_pos + 0.01, metrics_text, ha='center', fontsize=11, 
                    fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Super-Resolution Results: WISE → Spitzer\n(Physical Units: MJy/sr)', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        output_path = self.output_dir / 'comparison_grid_v2.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ Comparison grid saved: {output_path}")
    
    def create_error_analysis(self):
        """Create error analysis plots."""
        self.logger.info("Creating error analysis...")
        
        Y_pred = self.model.predict(self.X_test, verbose=0)
        
        if isinstance(Y_pred, list):
            Y_pred = Y_pred[-1]
        
        if Y_pred.dtype == np.float16:
            Y_pred = Y_pred.astype(np.float32)
        
        errors = self.Y_test - Y_pred
        abs_errors = np.abs(errors)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Error distribution
        error_data = errors.flatten()
        err_p1, err_p99 = np.percentile(error_data, [0.5, 99.5])
        axes[0, 0].hist(error_data, bins=100, alpha=0.7, edgecolor='black', 
                       range=(err_p1, err_p99))
        axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {error_data.mean():.4f}')
        axes[0, 0].set_xlabel('Error (Normalized)', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Error Distribution', fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Absolute error distribution
        abs_error_data = abs_errors.flatten()
        abs_p99 = np.percentile(abs_error_data, 99.5)
        axes[0, 1].hist(abs_error_data, bins=100, alpha=0.7, color='orange', 
                       edgecolor='black', range=(0, abs_p99))
        axes[0, 1].axvline(abs_error_data.mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {abs_error_data.mean():.4f}')
        axes[0, 1].set_xlabel('Absolute Error (Normalized)', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Absolute Error Distribution', fontsize=13, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Predicted vs True scatter
        sample_size = min(10000, len(self.Y_test.flatten()))
        idx = np.random.choice(len(self.Y_test.flatten()), sample_size, replace=False)
        y_true_sample = self.Y_test.flatten()[idx]
        y_pred_sample = Y_pred.flatten()[idx]
        
        true_p1, true_p99 = np.percentile(y_true_sample, [0.5, 99.5])
        pred_p1, pred_p99 = np.percentile(y_pred_sample, [0.5, 99.5])
        data_min = min(true_p1, pred_p1)
        data_max = max(true_p99, pred_p99)
        
        axes[0, 2].scatter(y_true_sample, y_pred_sample, alpha=0.1, s=1, c='blue', 
                          rasterized=True)
        axes[0, 2].plot([data_min, data_max], [data_min, data_max], 'r--', 
                       linewidth=2, label='Perfect prediction')
        axes[0, 2].set_xlabel('True Values (Normalized)', fontsize=12)
        axes[0, 2].set_ylabel('Predicted Values (Normalized)', fontsize=12)
        axes[0, 2].set_title('Predicted vs True', fontsize=13, fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_xlim(data_min, data_max)
        axes[0, 2].set_ylim(data_min, data_max)
        axes[0, 2].set_aspect('equal', adjustable='box')
        
        # MAE per sample distribution
        mae_per_sample = np.mean(abs_errors, axis=(1, 2, 3))
        mae_p99 = np.percentile(mae_per_sample, 99.5)
        axes[1, 0].hist(mae_per_sample, bins=50, alpha=0.7, color='green', 
                       edgecolor='black', range=(0, mae_p99))
        axes[1, 0].axvline(mae_per_sample.mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {mae_per_sample.mean():.4f}')
        axes[1, 0].set_xlabel('MAE per Sample (Normalized)', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('MAE Distribution Across Samples', fontsize=13, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Spatial error pattern
        mean_abs_error_map = np.mean(abs_errors[:, :, :, 0], axis=0)
        im = axes[1, 1].imshow(mean_abs_error_map, cmap='hot', origin='lower')
        axes[1, 1].set_title(f'Mean Spatial Error Pattern\nAvg: {mean_abs_error_map.mean():.4f}', 
                            fontsize=13, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], label='Mean Absolute Error')
        
        # Residual plot
        residuals_sample = errors.flatten()[idx]
        res_p1, res_p99 = np.percentile(residuals_sample, [0.5, 99.5])
        
        axes[1, 2].scatter(y_pred_sample, residuals_sample, alpha=0.1, s=1, 
                          c='purple', rasterized=True)
        axes[1, 2].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 2].set_xlabel('Predicted Values (Normalized)', fontsize=12)
        axes[1, 2].set_ylabel('Residuals (Normalized)', fontsize=12)
        axes[1, 2].set_title('Residual Plot', fontsize=13, fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_ylim(res_p1, res_p99)
        
        plt.tight_layout()
        output_path = self.output_dir / 'error_analysis_v2.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ Error analysis saved: {output_path}")
    
    def create_metrics_summary(self, metrics: Dict, source_metrics: Dict):
        """Create metrics summary visualization."""
        self.logger.info("Creating metrics summary...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Global metrics
        metric_names = ['SSIM', 'Pearson\nCorr', '1-NRMSE']
        metric_values = [metrics['ssim'], metrics['pearson_r'], 1 - metrics['nrmse']]
        colors = ['steelblue', 'orange', 'green']
        
        bars = axes[0].bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('Value', fontsize=12)
        axes[0].set_title('Global Quality Metrics', fontsize=14, fontweight='bold')
        axes[0].set_ylim(0, 1.1)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, metric_values):
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(), 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Source metrics
        source_names = ['Peak\nAccuracy', 'Flux\nAccuracy', 'Contrast\nRatio']
        source_values = [
            1 - source_metrics['peak_flux_error_mean'],
            1 - source_metrics['integrated_flux_error_mean'],
            min(source_metrics['contrast_ratio_pred'] / (source_metrics['contrast_ratio_true'] + 1e-8), 1.5)
        ]
        
        bars = axes[1].bar(source_names, source_values, color='purple', alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('Value (1 = perfect)', fontsize=12)
        axes[1].set_title('Source-Focused Metrics', fontsize=14, fontweight='bold')
        axes[1].set_ylim(0, 1.5)
        axes[1].axhline(1.0, color='green', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, source_values):
            axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height(), 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Error metrics
        error_names = ['MSE', 'RMSE', 'MAE']
        error_values = [metrics['mse'], metrics['rmse'], metrics['mae']]
        
        bars = axes[2].bar(error_names, error_values, color='red', alpha=0.6, edgecolor='black')
        axes[2].set_ylabel('Value', fontsize=12)
        axes[2].set_title('Error Metrics (lower is better)', fontsize=14, fontweight='bold')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, error_values):
            axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height(), 
                        f'{val:.2e}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle('Evaluation Summary - Super-Resolution V2.1', fontsize=16, fontweight='bold')
        plt.tight_layout()
        output_path = self.output_dir / 'metrics_summary_v2.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ Metrics summary saved: {output_path}")
    
    def save_metrics(self, metrics: Dict, source_metrics: Dict):
        """Save all metrics to JSON."""
        output_path = self.output_dir / 'evaluation_metrics_v2.json'
        
        all_metrics = {
            'global_metrics': {},
            'source_metrics': {},
            'normalization': {
                'method': self.norm_method,
                'ssim_max_val': self.ssim_max_val
            }
        }
        
        # Convert numpy types to Python types
        for key, value in metrics.items():
            if isinstance(value, (np.float32, np.float64)):
                all_metrics['global_metrics'][key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                all_metrics['global_metrics'][key] = int(value)
            else:
                all_metrics['global_metrics'][key] = value
        
        for key, value in source_metrics.items():
            if isinstance(value, (np.float32, np.float64)):
                all_metrics['source_metrics'][key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                all_metrics['source_metrics'][key] = int(value)
            else:
                all_metrics['source_metrics'][key] = value
        
        with open(output_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        
        self.logger.info(f"✓ Metrics saved: {output_path}")
    
    def run(self):
        """Execute evaluation pipeline."""
        try:
            self.load_model_and_data()
            
            self.logger.info("Generating predictions...")
            Y_pred = self.model.predict(self.X_test, batch_size=32, verbose=1)
            
            if isinstance(Y_pred, list):
                self.logger.info(f"  Multi-output model detected ({len(Y_pred)} outputs)")
                Y_pred = Y_pred[-1]
            
            if Y_pred.dtype == np.float16:
                Y_pred = Y_pred.astype(np.float32)
            
            # Compute metrics
            global_metrics = self.compute_metrics(self.Y_test, Y_pred)
            source_metrics = self.compute_source_metrics(self.Y_test, Y_pred)
            
            # Print results
            self.print_metrics(global_metrics)
            self.print_comprehensive_metrics(global_metrics, source_metrics)
            
            # Save metrics
            self.save_metrics(global_metrics, source_metrics)
            
            # Create visualizations
            self.create_comparison_grid(n_samples=self.config['visualization']['n_samples'])
            self.create_error_analysis()
            self.create_metrics_summary(global_metrics, source_metrics)
            
            self.logger.info("="*70)
            self.logger.info("EVALUATION COMPLETED SUCCESSFULLY (V2.1)")
            self.logger.info(f"Results saved to: {self.output_dir}")
            self.logger.info("="*70)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}", exc_info=True)
            return False


def main():
    parser = argparse.ArgumentParser(description='Evaluate super-resolution model V2.1')
    parser.add_argument('--config', type=str, default='eval_config.json', 
                       help='Path to evaluation configuration file')
    args = parser.parse_args()
    
    evaluator = SuperResolutionEvaluator(args.config)
    success = evaluator.run()
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())