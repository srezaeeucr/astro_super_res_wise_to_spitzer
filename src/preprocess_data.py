"""
Data Preprocessing for Super-Resolution V2
===========================================
MAJOR UPDATE: Properly handles unWISE (DN) to Spitzer (MJy/sr) unit conversion.

Key changes from V1:
1. Converts unWISE DN → MJy/sr BEFORE normalization
2. Uses asinh normalization (handles negatives + dynamic range)
3. Does NOT clip negative values (they're valid background noise)
4. Stores complete denormalization parameters

Run this ONCE before training!
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DataPreprocessor:
    """Preprocess and split dataset for super-resolution training with proper unit handling."""
    
    def __init__(self, config_path: str):
        """Initialize preprocessor."""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_output_directory()
        
        self.logger.info("="*70)
        self.logger.info("Data Preprocessing Pipeline V2 (Unit-Corrected)")
        self.logger.info("="*70)
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _setup_logging(self):
        """Setup logging."""
        self.logger = logging.getLogger('Preprocessor')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_output_directory(self):
        """Create output directory."""
        output_dir = Path(self.config['output']['processed_data_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        
        # Create log file
        log_file = output_dir / f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Output directory: {output_dir}")
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load raw paired dataset."""
        self.logger.info("Loading dataset...")
        
        data_path = self.config['data']['input_dataset_path']
        data = np.load(data_path)
        
        X = data['X_wise']      # (N, 14, 14, 1) - unWISE in DN
        Y = data['Y_spitzer']   # (N, 64, 64, 1) - Spitzer in MJy/sr
        ra = data['ra']
        dec = data['dec']
        
        self.logger.info(f"✓ Loaded dataset:")
        self.logger.info(f"  X (WISE): {X.shape}")
        self.logger.info(f"  Y (Spitzer): {Y.shape}")
        self.logger.info(f"  Total samples: {len(X)}")
        
        # Check for NaN/Inf values
        self._check_data_quality(X, Y)
        
        # Clean data if requested
        if self.config.get('preprocessing', {}).get('remove_nan', True):
            X, Y, ra, dec = self._remove_invalid_samples(X, Y, ra, dec)
        
        return X, Y, ra, dec
    
    def _check_data_quality(self, X: np.ndarray, Y: np.ndarray):
        """Check for NaN and Inf values in data."""
        self.logger.info("\nData Quality Check:")
        self.logger.info("-" * 50)
        
        # Check X (WISE)
        x_nan = np.isnan(X).sum()
        x_inf = np.isinf(X).sum()
        x_nan_samples = np.any(np.isnan(X), axis=(1, 2, 3)).sum()
        x_neg = (X < 0).sum()
        
        self.logger.info(f"WISE (X) - Units: DN (MAGZP=22.5):")
        self.logger.info(f"  NaN pixels: {x_nan:,} ({100*x_nan/X.size:.2f}%)")
        self.logger.info(f"  Inf pixels: {x_inf:,}")
        self.logger.info(f"  Negative pixels: {x_neg:,} ({100*x_neg/X.size:.2f}%)")
        self.logger.info(f"  Samples with NaN: {x_nan_samples}")
        self.logger.info(f"  Range: [{np.nanmin(X):.4e}, {np.nanmax(X):.4e}]")
        
        # Check Y (Spitzer)
        y_nan = np.isnan(Y).sum()
        y_inf = np.isinf(Y).sum()
        y_nan_samples = np.any(np.isnan(Y), axis=(1, 2, 3)).sum()
        y_neg = (Y < 0).sum()
        
        self.logger.info(f"\nSpitzer (Y) - Units: MJy/sr:")
        self.logger.info(f"  NaN pixels: {y_nan:,} ({100*y_nan/Y.size:.2f}%)")
        self.logger.info(f"  Inf pixels: {y_inf:,}")
        self.logger.info(f"  Negative pixels: {y_neg:,} ({100*y_neg/Y.size:.2f}%)")
        self.logger.info(f"  Samples with NaN: {y_nan_samples}")
        self.logger.info(f"  Range: [{np.nanmin(Y):.4e}, {np.nanmax(Y):.4e}]")
        
        # Unit mismatch warning
        x_med = np.nanmedian(X[X > 0]) if np.any(X > 0) else 1
        y_med = np.nanmedian(Y[Y > 0]) if np.any(Y > 0) else 1
        ratio = x_med / y_med
        
        self.logger.info(f"\n⚠ Unit Check:")
        self.logger.info(f"  WISE median (positive): {x_med:.4e} DN")
        self.logger.info(f"  Spitzer median (positive): {y_med:.4e} MJy/sr")
        self.logger.info(f"  Ratio: {ratio:.1f}x")
        
        if ratio > 10:
            self.logger.warning(f"  → Units differ by ~{ratio:.0f}x! Will convert WISE DN → MJy/sr")
    
    def _remove_invalid_samples(self, X: np.ndarray, Y: np.ndarray, 
                                 ra: np.ndarray, dec: np.ndarray) -> Tuple:
        """Remove samples containing NaN or Inf values."""
        self.logger.info("\nRemoving invalid samples...")
        
        initial_count = len(X)
        
        # Find valid samples (no NaN or Inf in either X or Y)
        valid_x = ~np.any(np.isnan(X) | np.isinf(X), axis=(1, 2, 3))
        valid_y = ~np.any(np.isnan(Y) | np.isinf(Y), axis=(1, 2, 3))
        valid_mask = valid_x & valid_y
        
        # Filter data
        X_clean = X[valid_mask]
        Y_clean = Y[valid_mask]
        ra_clean = ra[valid_mask]
        dec_clean = dec[valid_mask]
        
        removed_count = initial_count - len(X_clean)
        
        self.logger.info(f"✓ Removed {removed_count} invalid samples ({100*removed_count/initial_count:.1f}%)")
        self.logger.info(f"  Remaining: {len(X_clean)} valid samples")
        
        if len(X_clean) == 0:
            raise ValueError("No valid samples remaining after NaN removal!")
        
        return X_clean, Y_clean, ra_clean, dec_clean
    
    # ============================================================
    # NEW: Unit Conversion Function
    # ============================================================
    def convert_unwise_to_MJy_sr(self, X: np.ndarray) -> np.ndarray:
        """
        Convert unWISE DN to MJy/sr to match Spitzer units.
        
        unWISE calibration:
        - MAGZP = 22.5 (Vega system)
        - Pixel scale = 2.75 arcsec
        
        Conversion chain: DN → Vega mag → AB mag → Jy → MJy/sr
        """
        self.logger.info("\n" + "="*50)
        self.logger.info("CONVERTING unWISE DN → MJy/sr")
        self.logger.info("="*50)
        
        # Get config values or use defaults
        wise_config = self.config.get('wise_calibration', {})
        magzp = wise_config.get('magzp', 22.5)
        vega_to_ab = wise_config.get('vega_to_ab_offset', 2.699)  # W1 band
        pixel_scale = wise_config.get('pixel_scale_arcsec', 2.75)
        
        self.logger.info(f"  MAGZP: {magzp}")
        self.logger.info(f"  Vega→AB offset: {vega_to_ab}")
        self.logger.info(f"  Pixel scale: {pixel_scale} arcsec")
        
        # Store original stats
        X_orig_min = np.nanmin(X)
        X_orig_max = np.nanmax(X)
        X_orig_median = np.nanmedian(X[X > 0]) if np.any(X > 0) else 0
        
        # Handle sign (negative values are valid background noise)
        sign = np.sign(X)
        abs_X = np.abs(X)
        
        # Avoid log(0) - use tiny value for zeros/near-zeros
        abs_X = np.clip(abs_X, 1e-10, None)
        
        # Step 1: DN → Vega magnitude
        # mag_vega = MAGZP - 2.5 * log10(DN)
        mag_vega = magzp - 2.5 * np.log10(abs_X)
        
        # Step 2: Vega → AB magnitude
        # W1 offset: Δm = 2.699 (from WISE documentation)
        mag_AB = mag_vega + vega_to_ab
        
        # Step 3: AB magnitude → flux density (Jy)
        # AB system: 0 mag = 3631 Jy
        flux_Jy = 3631.0 * np.power(10.0, -0.4 * mag_AB)
        
        # Step 4: Jy → MJy/sr (surface brightness)
        # Pixel solid angle: (pixel_scale / 206265)^2 steradians
        # 206265 arcsec = 1 radian
        pixel_sr = (pixel_scale / 206265.0) ** 2
        MJy_sr = (flux_Jy * 1e-6) / pixel_sr
        
        # Restore sign for background noise
        X_converted = sign * MJy_sr
        
        # Log conversion results
        X_conv_min = np.nanmin(X_converted)
        X_conv_max = np.nanmax(X_converted)
        X_conv_median = np.nanmedian(X_converted[X_converted > 0]) if np.any(X_converted > 0) else 0
        
        self.logger.info(f"\nConversion results:")
        self.logger.info(f"  Original (DN):    [{X_orig_min:.4e}, {X_orig_max:.4e}], median={X_orig_median:.4e}")
        self.logger.info(f"  Converted (MJy/sr): [{X_conv_min:.4e}, {X_conv_max:.4e}], median={X_conv_median:.4e}")
        self.logger.info(f"  Conversion factor (median): {X_conv_median/X_orig_median:.4e}")
        
        return X_converted.astype(np.float32)
    
    def compute_statistics(self, X: np.ndarray, Y: np.ndarray, label: str = ""):
        """Compute and log data statistics."""
        self.logger.info(f"\nDataset Statistics {label}:")
        self.logger.info("-" * 50)
        
        # WISE statistics
        self.logger.info("WISE (Input):")
        self.logger.info(f"  Shape: {X.shape}")
        self.logger.info(f"  Mean: {np.nanmean(X):.6e}")
        self.logger.info(f"  Std: {np.nanstd(X):.6e}")
        self.logger.info(f"  Min: {np.nanmin(X):.6e}")
        self.logger.info(f"  Max: {np.nanmax(X):.6e}")
        self.logger.info(f"  Median: {np.nanmedian(X):.6e}")
        self.logger.info(f"  % Negative: {100*(X<0).sum()/X.size:.2f}%")
        
        # Spitzer statistics
        self.logger.info("\nSpitzer (Target):")
        self.logger.info(f"  Shape: {Y.shape}")
        self.logger.info(f"  Mean: {np.nanmean(Y):.6e}")
        self.logger.info(f"  Std: {np.nanstd(Y):.6e}")
        self.logger.info(f"  Min: {np.nanmin(Y):.6e}")
        self.logger.info(f"  Max: {np.nanmax(Y):.6e}")
        self.logger.info(f"  Median: {np.nanmedian(Y):.6e}")
        self.logger.info(f"  % Negative: {100*(Y<0).sum()/Y.size:.2f}%")
    
    # ============================================================
    # NEW: Asinh Normalization (replaces old normalize_data)
    # ============================================================
    def normalize_data(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Normalize using asinh stretch - the standard for astronomical imaging.
        
        Both X and Y should be in the SAME units (MJy/sr) before this step!
        
        asinh(x) ≈ x for small x (linear regime)
        asinh(x) ≈ ln(2x) for large x (logarithmic compression)
        
        This handles:
        - Large dynamic range (faint to bright sources)
        - Negative values (background noise)
        - Preserves relative flux ratios
        """
        norm_method = self.config.get('normalization', {}).get('method', 'asinh')
        
        if norm_method == 'asinh':
            return self._normalize_asinh(X, Y)
        elif norm_method == 'minmax':
            return self._normalize_minmax(X, Y)
        else:
            self.logger.warning(f"Unknown method '{norm_method}', using asinh")
            return self._normalize_asinh(X, Y)
    
    def _normalize_asinh(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Asinh normalization - recommended for astronomical data.
        """
        self.logger.info("\n" + "="*50)
        self.logger.info("APPLYING ASINH NORMALIZATION")
        self.logger.info("="*50)
        
        # Compute softening parameters (controls linear→log transition)
        # Use median of positive values - this sets the "knee" of the curve
        X_soft = np.nanmedian(X[X > 0]) if np.any(X > 0) else 1.0
        Y_soft = np.nanmedian(Y[Y > 0]) if np.any(Y > 0) else 1.0
        
        self.logger.info(f"Softening parameters (median of positive values):")
        self.logger.info(f"  X (WISE): {X_soft:.6e} MJy/sr")
        self.logger.info(f"  Y (Spitzer): {Y_soft:.6e} MJy/sr")
        
        # Apply asinh transform
        # asinh(x/soft) gives ~linear behavior for |x| < soft
        # and ~logarithmic for |x| >> soft
        X_asinh = np.arcsinh(X / X_soft)
        Y_asinh = np.arcsinh(Y / Y_soft)
        
        # Scale to roughly [0, 1] using percentiles
        # We use 1st and 99th percentiles to be robust to outliers
        X_p01, X_p99 = np.nanpercentile(X_asinh, [1, 99])
        Y_p01, Y_p99 = np.nanpercentile(Y_asinh, [1, 99])
        
        self.logger.info(f"\nAsinh range (before scaling):")
        self.logger.info(f"  X: [{np.nanmin(X_asinh):.4f}, {np.nanmax(X_asinh):.4f}]")
        self.logger.info(f"  Y: [{np.nanmin(Y_asinh):.4f}, {np.nanmax(Y_asinh):.4f}]")
        self.logger.info(f"  X percentiles [1%, 99%]: [{X_p01:.4f}, {X_p99:.4f}]")
        self.logger.info(f"  Y percentiles [1%, 99%]: [{Y_p01:.4f}, {Y_p99:.4f}]")
        
        # Scale to [0, 1] range (values outside percentiles will be <0 or >1)
        X_norm = (X_asinh - X_p01) / (X_p99 - X_p01 + 1e-10)
        Y_norm = (Y_asinh - Y_p01) / (Y_p99 - Y_p01 + 1e-10)
        
        # Store all parameters needed for INVERSE transform
        norm_params = {
            'method': 'asinh',
            # Softening parameters
            'X_soft': float(X_soft),
            'Y_soft': float(Y_soft),
            # Percentile scaling parameters
            'X_p01': float(X_p01),
            'X_p99': float(X_p99),
            'Y_p01': float(Y_p01),
            'Y_p99': float(Y_p99),
            # Unit info
            'input_unit': 'MJy/sr',
            'wise_converted_from': 'DN',
        }
        
        self.logger.info(f"\n✓ Normalization complete:")
        self.logger.info(f"  X range: [{X_norm.min():.4f}, {X_norm.max():.4f}]")
        self.logger.info(f"  Y range: [{Y_norm.min():.4f}, {Y_norm.max():.4f}]")
        
        return X_norm.astype(np.float32), Y_norm.astype(np.float32), norm_params
    
    def _normalize_minmax(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Alternative: Robust min-max normalization using percentiles.
        Simpler than asinh but less robust to extreme values.
        """
        self.logger.info("\n" + "="*50)
        self.logger.info("APPLYING MIN-MAX NORMALIZATION")
        self.logger.info("="*50)
        
        # Use percentiles for robustness
        X_p01, X_p999 = np.nanpercentile(X, [0.1, 99.9])
        Y_p01, Y_p999 = np.nanpercentile(Y, [0.1, 99.9])
        
        self.logger.info(f"Percentile ranges [0.1%, 99.9%]:")
        self.logger.info(f"  X: [{X_p01:.6e}, {X_p999:.6e}]")
        self.logger.info(f"  Y: [{Y_p01:.6e}, {Y_p999:.6e}]")
        
        # Scale to [0, 1] - values outside percentiles can exceed [0, 1]
        X_norm = (X - X_p01) / (X_p999 - X_p01 + 1e-10)
        Y_norm = (Y - Y_p01) / (Y_p999 - Y_p01 + 1e-10)
        
        norm_params = {
            'method': 'minmax',
            'X_min': float(X_p01),
            'X_max': float(X_p999),
            'Y_min': float(Y_p01),
            'Y_max': float(Y_p999),
            'input_unit': 'MJy/sr',
            'wise_converted_from': 'DN',
        }
        
        self.logger.info(f"\n✓ Normalization complete:")
        self.logger.info(f"  X range: [{X_norm.min():.4f}, {X_norm.max():.4f}]")
        self.logger.info(f"  Y range: [{Y_norm.min():.4f}, {Y_norm.max():.4f}]")
        
        return X_norm.astype(np.float32), Y_norm.astype(np.float32), norm_params
    
    def split_data(self, X: np.ndarray, Y: np.ndarray, 
                   ra: np.ndarray, dec: np.ndarray) -> dict:
        """Split data into train and test sets."""
        self.logger.info("\nSplitting data...")
        
        test_size = self.config['split']['test_size']
        random_seed = self.config['split']['random_seed']
        
        # Perform split
        X_train, X_test, Y_train, Y_test, ra_train, ra_test, dec_train, dec_test = train_test_split(
            X, Y, ra, dec,
            test_size=test_size,
            random_state=random_seed,
            shuffle=True
        )
        
        train_size = len(X_train)
        test_size_actual = len(X_test)
        total = train_size + test_size_actual
        
        self.logger.info(f"✓ Data split:")
        self.logger.info(f"  Train: {train_size} samples ({100*train_size/total:.1f}%)")
        self.logger.info(f"  Test:  {test_size_actual} samples ({100*test_size_actual/total:.1f}%)")
        self.logger.info(f"  Random seed: {random_seed}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'Y_train': Y_train,
            'Y_test': Y_test,
            'ra_train': ra_train,
            'ra_test': ra_test,
            'dec_train': dec_train,
            'dec_test': dec_test
        }
    
    def save_splits(self, splits: dict, norm_params: dict):
        """Save train and test splits to separate files."""
        self.logger.info("\nSaving processed data...")
        
        # Save training set
        train_path = self.output_dir / 'train_data.npz'
        np.savez_compressed(
            train_path,
            X_train=splits['X_train'],
            Y_train=splits['Y_train'],
            ra_train=splits['ra_train'],
            dec_train=splits['dec_train']
        )
        train_size_mb = train_path.stat().st_size / (1024**2)
        self.logger.info(f"✓ Training set saved: {train_path} ({train_size_mb:.2f} MB)")
        
        # Save test set
        test_path = self.output_dir / 'test_data.npz'
        np.savez_compressed(
            test_path,
            X_test=splits['X_test'],
            Y_test=splits['Y_test'],
            ra_test=splits['ra_test'],
            dec_test=splits['dec_test']
        )
        test_size_mb = test_path.stat().st_size / (1024**2)
        self.logger.info(f"✓ Test set saved: {test_path} ({test_size_mb:.2f} MB)")
        
        # Save normalization parameters (NPZ)
        norm_path = self.output_dir / 'normalization_params.npz'
        np.savez(norm_path, **norm_params)
        self.logger.info(f"✓ Normalization params saved: {norm_path}")
        
        # Save as JSON for easy reading
        norm_json_path = self.output_dir / 'normalization_params.json'
        with open(norm_json_path, 'w') as f:
            json.dump(norm_params, f, indent=4)
        self.logger.info(f"✓ Normalization params (JSON): {norm_json_path}")
        
        # Save comprehensive metadata
        metadata = {
            'train_samples': int(len(splits['X_train'])),
            'test_samples': int(len(splits['X_test'])),
            'input_shape': list(splits['X_train'].shape[1:]),
            'output_shape': list(splits['Y_train'].shape[1:]),
            'test_split': self.config['split']['test_size'],
            'random_seed': self.config['split']['random_seed'],
            'preprocessing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'normalization': norm_params,
            'pipeline_version': 2,
            'notes': [
                'WISE data converted from DN to MJy/sr before normalization',
                'Both X and Y in same units (MJy/sr) before normalization',
                'Negative values preserved (valid background noise)',
                f"Normalization method: {norm_params['method']}"
            ]
        }
        
        metadata_path = self.output_dir / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        self.logger.info(f"✓ Metadata saved: {metadata_path}")
    
    def create_visualizations(self, splits: dict, X_orig: np.ndarray = None, 
                              X_converted: np.ndarray = None):
        """Create visualization of data splits and preprocessing steps."""
        self.logger.info("\nCreating visualizations...")
        
        # ============================================================
        # Plot 1: Train/Test Split Examples
        # ============================================================
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Train set - WISE example
        idx = np.random.randint(0, len(splits['X_train']))
        x_train_sample = splits['X_train'][idx, :, :, 0]
        vmin, vmax = np.percentile(x_train_sample, [1, 99])
        im1 = axes[0, 0].imshow(x_train_sample, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 0].set_title('Train - WISE Input\n(14×14 pixels, normalized)', fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        
        # Train set - Spitzer example
        y_train_sample = splits['Y_train'][idx, :, :, 0]
        vmin, vmax = np.percentile(y_train_sample, [1, 99])
        im2 = axes[0, 1].imshow(y_train_sample, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, 1].set_title('Train - Spitzer Target\n(64×64 pixels, normalized)', fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
        
        # Train set distribution
        axes[0, 2].hist(splits['Y_train'].flatten(), bins=100, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('Normalized Pixel Value')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Train Set - Pixel Distribution', fontweight='bold')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Test set - WISE example
        idx = np.random.randint(0, len(splits['X_test']))
        x_test_sample = splits['X_test'][idx, :, :, 0]
        vmin, vmax = np.percentile(x_test_sample, [1, 99])
        im3 = axes[1, 0].imshow(x_test_sample, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[1, 0].set_title('Test - WISE Input\n(14×14 pixels, normalized)', fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        
        # Test set - Spitzer example
        y_test_sample = splits['Y_test'][idx, :, :, 0]
        vmin, vmax = np.percentile(y_test_sample, [1, 99])
        im4 = axes[1, 1].imshow(y_test_sample, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[1, 1].set_title('Test - Spitzer Target\n(64×64 pixels, normalized)', fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
        
        # Test set distribution
        axes[1, 2].hist(splits['Y_test'].flatten(), bins=100, alpha=0.7, 
                       color='orange', edgecolor='black')
        axes[1, 2].set_xlabel('Normalized Pixel Value')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Test Set - Pixel Distribution', fontweight='bold')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Preprocessed Data - Train/Test Splits (V2)', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path = self.output_dir / 'data_splits_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"✓ Visualization saved: {output_path}")
        
        # ============================================================
        # Plot 2: Preprocessing Pipeline Diagnostics
        # ============================================================
        if X_orig is not None and X_converted is not None:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Row 1: Before conversion
            # WISE original (DN)
            x_flat = X_orig.flatten()
            x_p01, x_p99 = np.percentile(x_flat, [1, 99])
            axes[0, 0].hist(x_flat, bins=100, range=(x_p01, x_p99), alpha=0.7, color='blue')
            axes[0, 0].axvline(0, color='red', linestyle='--', label='Zero')
            axes[0, 0].set_xlabel('DN')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('WISE Original (DN)', fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].set_yscale('log')
            
            # WISE converted (MJy/sr)
            xc_flat = X_converted.flatten()
            xc_p01, xc_p99 = np.percentile(xc_flat, [1, 99])
            axes[0, 1].hist(xc_flat, bins=100, range=(xc_p01, xc_p99), alpha=0.7, color='purple')
            axes[0, 1].axvline(0, color='red', linestyle='--', label='Zero')
            axes[0, 1].set_xlabel('MJy/sr')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('WISE Converted (MJy/sr)', fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].set_yscale('log')
            
            # Conversion scatter
            sample_idx = np.random.choice(len(x_flat), min(10000, len(x_flat)), replace=False)
            axes[0, 2].scatter(x_flat[sample_idx], xc_flat[sample_idx], alpha=0.1, s=1)
            axes[0, 2].set_xlabel('Original DN')
            axes[0, 2].set_ylabel('Converted MJy/sr')
            axes[0, 2].set_title('Unit Conversion', fontweight='bold')
            axes[0, 2].grid(True, alpha=0.3)
            
            # Row 2: After normalization
            # WISE normalized
            xn_flat = splits['X_train'].flatten()
            axes[1, 0].hist(xn_flat, bins=100, alpha=0.7, color='blue')
            axes[1, 0].axvline(0, color='red', linestyle='--')
            axes[1, 0].axvline(1, color='green', linestyle='--')
            axes[1, 0].set_xlabel('Normalized Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('WISE Normalized', fontweight='bold')
            axes[1, 0].set_yscale('log')
            
            # Spitzer normalized
            yn_flat = splits['Y_train'].flatten()
            axes[1, 1].hist(yn_flat, bins=100, alpha=0.7, color='green')
            axes[1, 1].axvline(0, color='red', linestyle='--')
            axes[1, 1].axvline(1, color='green', linestyle='--')
            axes[1, 1].set_xlabel('Normalized Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Spitzer Normalized', fontweight='bold')
            axes[1, 1].set_yscale('log')
            
            # X vs Y correlation
            sample_idx = np.random.choice(len(splits['X_train']), 
                                          min(5000, len(splits['X_train'])), replace=False)
            x_means = splits['X_train'][sample_idx].mean(axis=(1, 2, 3))
            y_means = splits['Y_train'][sample_idx].mean(axis=(1, 2, 3))
            axes[1, 2].scatter(x_means, y_means, alpha=0.3, s=5)
            axes[1, 2].plot([0, 1], [0, 1], 'r--', label='1:1 line')
            axes[1, 2].set_xlabel('WISE Mean (normalized)')
            axes[1, 2].set_ylabel('Spitzer Mean (normalized)')
            axes[1, 2].set_title('Cutout Mean Correlation', fontweight='bold')
            axes[1, 2].legend()
            axes[1, 2].set_aspect('equal')
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.suptitle('Preprocessing Pipeline Diagnostics', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            output_path = self.output_dir / 'preprocessing_diagnostics.png'
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()
            self.logger.info(f"✓ Diagnostics saved: {output_path}")
        
        # ============================================================
        # Plot 3: Split Summary Pie Chart
        # ============================================================
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        labels = ['Training Set', 'Test Set']
        sizes = [len(splits['X_train']), len(splits['X_test'])]
        colors = ['#ff9999', '#66b3ff']
        explode = (0.05, 0)
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 14})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(16)
        
        ax.set_title(f'Dataset Split (V2)\nTotal: {sum(sizes)} samples', 
                    fontsize=16, fontweight='bold')
        
        legend_labels = [f'{label}: {size:,} samples' for label, size in zip(labels, sizes)]
        ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
        
        output_path = self.output_dir / 'split_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"✓ Split summary saved: {output_path}")
    
    def run(self):
        """Execute preprocessing pipeline."""
        try:
            # ============================================================
            # Step 1: Load raw data
            # ============================================================
            X, Y, ra, dec = self.load_data()
            
            # Store original for diagnostics
            X_original = X.copy()
            
            # Compute statistics on raw data
            self.compute_statistics(X, Y, label="(Raw - Different Units!)")
            
            # ============================================================
            # Step 2: Convert WISE DN → MJy/sr (NEW!)
            # ============================================================
            X_converted = self.convert_unwise_to_MJy_sr(X)
            
            # Verify units now match
            self.logger.info("\n--- After Unit Conversion ---")
            x_med = np.nanmedian(X_converted[X_converted > 0])
            y_med = np.nanmedian(Y[Y > 0])
            ratio = x_med / y_med
            self.logger.info(f"  WISE median: {x_med:.6e} MJy/sr")
            self.logger.info(f"  Spitzer median: {y_med:.6e} MJy/sr")
            self.logger.info(f"  Ratio: {ratio:.2f}x (should be ~1)")
            
            if 0.1 < ratio < 10:
                self.logger.info("  ✓ Units now comparable!")
            else:
                self.logger.warning(f"  ⚠ Ratio still off - check conversion")
            
            # ============================================================
            # Step 3: Normalize (asinh or minmax)
            # ============================================================
            X_norm, Y_norm, norm_params = self.normalize_data(X_converted, Y)
            
            # ============================================================
            # Step 4: Split
            # ============================================================
            splits = self.split_data(X_norm, Y_norm, ra, dec)
            
            # ============================================================
            # Step 5: Save
            # ============================================================
            self.save_splits(splits, norm_params)
            
            # ============================================================
            # Step 6: Visualize
            # ============================================================
            self.create_visualizations(splits, X_original[:1000], X_converted[:1000])
            
            # Final summary
            self.logger.info("\n" + "="*70)
            self.logger.info("PREPROCESSING V2 COMPLETED SUCCESSFULLY")
            self.logger.info("="*70)
            self.logger.info(f"Output directory: {self.output_dir}")
            self.logger.info(f"Training set: train_data.npz ({len(splits['X_train'])} samples)")
            self.logger.info(f"Test set: test_data.npz ({len(splits['X_test'])} samples)")
            self.logger.info(f"Normalization: {norm_params['method']}")
            self.logger.info("")
            self.logger.info("Key improvements over V1:")
            self.logger.info("  ✓ WISE converted from DN to MJy/sr")
            self.logger.info("  ✓ Both datasets in same units before normalization")
            self.logger.info("  ✓ Negative values preserved (valid noise)")
            self.logger.info("  ✓ Proper denormalization params saved")
            self.logger.info("")
            self.logger.info("Next steps:")
            self.logger.info("  1. Run: python train_super_resolution.py")
            self.logger.info("  2. Run: python evaluate_super_resolution.py")
            self.logger.info("="*70)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}", exc_info=True)
            return False


# ============================================================
# UTILITY FUNCTIONS FOR DENORMALIZATION
# (Use these in evaluate_super_resolution.py)
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
        raise ValueError(f"Unknown normalization method: {method}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess data for super-resolution training (V2)')
    parser.add_argument('--config', type=str, default='preprocess_config.json',
                       help='Path to preprocessing configuration file')
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(args.config)
    success = preprocessor.run()
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())