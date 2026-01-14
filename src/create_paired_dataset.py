"""
Spitzer-WISE Paired Dataset Creation Pipeline
==============================================
Creates paired cutouts for super-resolution training 

Author: Saeed Rezaee, PhD
Date: Jan, 2026
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy.stats import gaussian_kde
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u


class PairedDatasetCreator:
    """Creates paired Spitzer-WISE cutouts for super-resolution training."""
    
    def __init__(self, config_path: str):
        """
        Initialize the dataset creator.
        
        Parameters:
        -----------
        config_path : str
            Path to configuration JSON file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_output_directory()
        
        self.logger.info("="*70)
        self.logger.info("Spitzer-WISE Paired Dataset Creation Pipeline")
        self.logger.info("="*70)
        
        # Data containers
        self.spitzer_data = None
        self.spitzer_wcs = None
        self.wise_tiles = []
        self.catalog = None
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    def _setup_logging(self):
        """Configure logging with both file and console handlers."""
        log_config = self.config['logging']
        
        # Create logger
        self.logger = logging.getLogger('PairedDatasetCreator')
        self.logger.setLevel(getattr(logging, log_config['level']))
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_config['level']))
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (will be created after output directory is set up)
        self.file_handler = None
        
    def _setup_output_directory(self):
        """Create output directory structure."""
        output_dir = Path(self.config['output']['base_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.output_paths = {
            'base': output_dir,
            'data': output_dir / 'data',
            'visualizations': output_dir / 'visualizations',
            'logs': output_dir / 'logs'
        }
        
        for path in self.output_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        # Add file handler for logging
        log_file = self.output_paths['logs'] / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)
        
        self.logger.info(f"Output directory created: {output_dir}")
        self.logger.info(f"Log file: {log_file}")
    
    def load_spitzer_data(self):
        """Load Spitzer mosaic data."""
        self.logger.info("Loading Spitzer data...")
        spitzer_config = self.config['data']['spitzer']
        
        try:
            hdul = fits.open(spitzer_config['file_path'])
            self.spitzer_data = hdul[0].data
            self.spitzer_wcs = WCS(hdul[0].header)
            
            # NEW: Extract unit information from header
            self.spitzer_unit = hdul[0].header.get('BUNIT', 'MJy/sr')
            
            hdul.close()
            
            self.logger.info(f"✓ Loaded Spitzer data: {self.spitzer_data.shape}")
            self.logger.info(f"  Pixel scale: {spitzer_config['pixel_scale']} arcsec/pixel")
            self.logger.info(f"  Units: {self.spitzer_unit}")  # NEW
            
        except Exception as e:
            self.logger.error(f"Failed to load Spitzer data: {e}")
            raise
    
    def load_wise_data(self):
        """Load WISE tile data."""
        self.logger.info("Loading WISE tiles...")
        wise_config = self.config['data']['wise']
        
        for tile_file in wise_config['tile_files']:
            try:
                if not os.path.exists(tile_file):
                    self.logger.warning(f"  Tile not found: {tile_file}")
                    continue
                
                hdul = fits.open(tile_file)
                
                # NEW: Extract calibration info from header
                magzp = hdul[0].header.get('MAGZP', 22.5)
                
                self.wise_tiles.append({
                    'data': hdul[0].data,
                    'wcs': WCS(hdul[0].header),
                    'name': os.path.basename(tile_file),
                    'magzp': magzp  # NEW: Store zeropoint
                })
                hdul.close()
                self.logger.info(f"  ✓ Loaded: {os.path.basename(tile_file)} (MAGZP={magzp})")
                
            except Exception as e:
                self.logger.warning(f"  ✗ Failed to load {tile_file}: {e}")
        
        if len(self.wise_tiles) == 0:
            self.logger.error("No WISE tiles loaded!")
            raise ValueError("No WISE tiles available")
        
        self.logger.info(f"✓ Successfully loaded {len(self.wise_tiles)} WISE tiles")
        self.logger.info(f"  Pixel scale: {wise_config['pixel_scale']} arcsec/pixel")
        self.logger.info(f"  Units: DN (Vega nMgy, MAGZP=22.5)")  # NEW
    
    def load_catalog(self):
            """Load source catalog from FITS file using dynamic filters from config."""
            self.logger.info("Loading catalog...")
            cat_config = self.config['data']['catalog']
            
            try:
                from astropy.table import Table
                import numpy as np
                
                catalog_path = cat_config['file_path']
                self.logger.info(f"  Reading catalog from: {catalog_path}")
                
                # Load catalog
                self.catalog = Table.read(catalog_path)
                self.logger.info(f"  ✓ Initial load: {len(self.catalog)} sources")
                
                # Process Dynamic Filters
                filters = cat_config.get('filters', [])
                
                for f in filters:
                    col = f.get('column')
                    op = f.get('operator', '==') 
                    val = f.get('value')
                    
                    if col not in self.catalog.colnames:
                        self.logger.warning(f"  Skipping filter: Column '{col}' not found.")
                        continue

                    n_before = len(self.catalog)
                    
                    # Logical mapping
                    if op == "==":
                        mask = self.catalog[col] == val
                    elif op == ">":
                        mask = self.catalog[col] > val
                    elif op == "<":
                        mask = self.catalog[col] < val
                    elif op == ">=":
                        mask = self.catalog[col] >= val
                    elif op == "SNR_GT":
                        err_col = f.get('error_column')
                        min_flux = f.get('min_flux', 0.0)
                        
                        if err_col in self.catalog.colnames:
                            flux = self.catalog[col]
                            err = self.catalog[err_col]
                            valid = (np.isfinite(flux)) & (np.isfinite(err)) & (err > 0) & (flux > min_flux)
                            snr = np.zeros(len(self.catalog))
                            snr[valid] = flux[valid] / err[valid]
                            mask = snr >= val
                        else:
                            self.logger.warning(f"  Skipping SNR filter: '{err_col}' missing.")
                            continue
                    else:
                        self.logger.error(f"  Unknown operator: {op}")
                        continue
                    
                    self.catalog = self.catalog[mask]
                    self.logger.info(f"  Applied {col} {op} {val}: {n_before} → {len(self.catalog)}")

                # Verify coordinate columns
                required_cols = [cat_config['ra_column'], cat_config['dec_column']]
                for col in required_cols:
                    if col not in self.catalog.colnames:
                        raise ValueError(f"Missing required column: {col}")
                
                self.logger.info(f"✓ Catalog ready: {len(self.catalog)} sources")
                
            except Exception as e:
                self.logger.error(f"Failed to load catalog: {e}")
                raise
    
    def get_paired_cutout(self, ra: float, dec: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract paired cutouts at the same sky location.
        """
        cutout_config = self.config['cutouts']
        spitzer_size = cutout_config['spitzer_size_pixels']
        wise_size = cutout_config['wise_size_pixels']
        
        coord = SkyCoord(ra*u.deg, dec*u.deg)
        
        # --- SPITZER CUTOUT ---
        try:
            x_spit, y_spit = coord.to_pixel(self.spitzer_wcs)
            if not (0 <= x_spit < self.spitzer_data.shape[1] and 
                    0 <= y_spit < self.spitzer_data.shape[0]):
                return None, None
            
            spit_cut = Cutout2D(
                data=self.spitzer_data,
                position=(x_spit, y_spit),
                size=(spitzer_size, spitzer_size),
                wcs=self.spitzer_wcs
            )
            spitzer_cutout = spit_cut.data
            
            if spitzer_cutout.shape != (spitzer_size, spitzer_size):
                return None, None
                
        except Exception:
            return None, None
        
        # --- WISE CUTOUT ---
        wise_cutout = None
        for tile in self.wise_tiles:
            try:
                x_wise, y_wise = coord.to_pixel(tile['wcs'])
                if 0 <= x_wise < tile['data'].shape[1] and 0 <= y_wise < tile['data'].shape[0]:
                    wise_cut = Cutout2D(
                        data=tile['data'],
                        position=(x_wise, y_wise),
                        size=(wise_size, wise_size),
                        wcs=tile['wcs']
                    )
                    wise_cutout = wise_cut.data
                    
                    if wise_cutout.shape != (wise_size, wise_size):
                        continue
                    break
            except:
                continue
        
        if wise_cutout is None:
            return None, None
        
        return spitzer_cutout, wise_cutout
    
    def create_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Create paired dataset from catalog.
        """
        self.logger.info("Creating paired cutouts...")
        
        max_sources = self.config['processing']['max_sources']
        log_interval = self.config['processing']['log_interval']
        
        X_wise = []
        Y_spitzer = []
        sources_info = []
        
        n_processed = 0
        n_success = 0
        
        for i, row in enumerate(self.catalog[:max_sources]):
            if i % log_interval == 0:
                self.logger.info(f"  Processing {i}/{max_sources}...")
            
            n_processed += 1
            
            ra_col = self.config['data']['catalog']['ra_column']
            dec_col = self.config['data']['catalog']['dec_column']
            
            spit_cut, wise_cut = self.get_paired_cutout(
                row[ra_col], 
                row[dec_col]
            )
            
            if spit_cut is not None and wise_cut is not None:
                Y_spitzer.append(spit_cut.astype(np.float32))
                X_wise.append(wise_cut.astype(np.float32))
                
                sources_info.append({
                    'ra': row[ra_col],
                    'dec': row[dec_col],
                    'catalog_index': i
                })
                n_success += 1
        
        X_wise = np.array(X_wise)[..., None]
        Y_spitzer = np.array(Y_spitzer)[..., None]
        
        success_rate = 100 * n_success / n_processed if n_processed > 0 else 0
        
        self.logger.info(f"✓ Dataset creation complete:")
        self.logger.info(f"  Processed: {n_processed} sources")
        self.logger.info(f"  Successful: {n_success} pairs")
        self.logger.info(f"  Success rate: {success_rate:.1f}%")
        self.logger.info(f"  WISE shape: {X_wise.shape}")
        self.logger.info(f"  Spitzer shape: {Y_spitzer.shape}")
        
        return X_wise, Y_spitzer, sources_info
    
    def save_dataset(self, X_wise: np.ndarray, Y_spitzer: np.ndarray, 
                     sources_info: List[Dict]):
        """Save the paired dataset to disk with unit metadata."""
        self.logger.info("Saving dataset...")
        
        output_file = self.output_paths['data'] / self.config['output']['dataset_filename']
        
        # ============================================================
        # UPDATED: Save with comprehensive unit metadata
        # ============================================================
        np.savez_compressed(
            output_file,
            X_wise=X_wise,
            Y_spitzer=Y_spitzer,
            ra=np.array([s['ra'] for s in sources_info]),
            dec=np.array([s['dec'] for s in sources_info]),
            # Pixel scales
            wise_pixscale=self.config['data']['wise']['pixel_scale'],
            spitzer_pixscale=self.config['data']['spitzer']['pixel_scale'],
            # NEW: Unit information
            wise_unit='DN',
            wise_magzp=22.5,
            wise_vega_to_ab_offset=2.699,  # For W1 band
            spitzer_unit='MJy/sr',
            # NEW: Wavelength info
            wise_wavelength_um=3.4,
            spitzer_wavelength_um=3.6
        )
        
        file_size_mb = output_file.stat().st_size / (1024**2)
        self.logger.info(f"✓ Dataset saved to: {output_file}")
        self.logger.info(f"  File size: {file_size_mb:.2f} MB")
        self.logger.info(f"  WISE units: DN (MAGZP=22.5, Vega)")
        self.logger.info(f"  Spitzer units: MJy/sr")
        
        return output_file
    
    def create_alignment_visualization(self, X_wise: np.ndarray, Y_spitzer: np.ndarray,
                                       sources_info: List[Dict]):
        """Create visualization showing paired alignment examples."""
        self.logger.info("Creating alignment verification plots...")
        
        n_examples = self.config['visualization']['n_alignment_examples']
        
        fig, axes = plt.subplots(n_examples, 2, figsize=(10, 5*n_examples))
        
        if n_examples == 1:
            axes = np.array([axes])
        
        for i in range(n_examples):
            idx = np.random.randint(0, len(X_wise))
            
            # WISE (low-res)
            wise_img = X_wise[idx, :, :, 0]
            vmin_w, vmax_w = np.percentile(wise_img[np.isfinite(wise_img)], [1, 99])
            axes[i, 0].imshow(wise_img, origin='lower', cmap='gray', vmin=vmin_w, vmax=vmax_w)
            axes[i, 0].set_title(f'WISE (Input X) - {wise_img.shape[0]}×{wise_img.shape[1]} @ 2.75"/pix\n'
                                f'RA={sources_info[idx]["ra"]:.4f}°, Dec={sources_info[idx]["dec"]:.4f}°\n'
                                f'Units: DN (MAGZP=22.5)',  # NEW
                                fontsize=10)
            axes[i, 0].axis('off')
            
            # Spitzer (high-res)
            spit_img = Y_spitzer[idx, :, :, 0]
            vmin_s, vmax_s = np.percentile(spit_img[np.isfinite(spit_img)], [1, 99])
            axes[i, 1].imshow(spit_img, origin='lower', cmap='gray', vmin=vmin_s, vmax=vmax_s)
            axes[i, 1].set_title(f'Spitzer (Target Y) - {spit_img.shape[0]}×{spit_img.shape[1]} @ 0.6"/pix\n'
                                f'Same sky region: ~38.4" × 38.4"\n'
                                f'Units: MJy/sr',  # NEW
                                fontsize=10)
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.05)
        
        output_file = self.output_paths['visualizations'] / 'alignment_verification.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ Saved alignment verification: {output_file}")
    
    def create_coverage_visualization(self, X_wise: np.ndarray, Y_spitzer: np.ndarray,
                                      sources_info: List[Dict]):
        """Create comprehensive coverage visualization."""
        self.logger.info("Creating coverage visualization...")
        
        ra_cutouts = np.array([s['ra'] for s in sources_info])
        dec_cutouts = np.array([s['dec'] for s in sources_info])
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # --- LEFT: Full field with cutout boxes ---
        vmin, vmax = np.percentile(self.spitzer_data[np.isfinite(self.spitzer_data)], [0.5, 99.5])
        axes[0].imshow(self.spitzer_data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax,
                      aspect='auto', interpolation='nearest')
        
        cutout_size = self.config['cutouts']['spitzer_size_pixels']
        for ra, dec in zip(ra_cutouts, dec_cutouts):
            coord = SkyCoord(ra*u.deg, dec*u.deg)
            x, y = coord.to_pixel(self.spitzer_wcs)
            rect = Rectangle((x - cutout_size/2, y - cutout_size/2),
                           cutout_size, cutout_size,
                           linewidth=0.5, edgecolor='red', facecolor='none', alpha=0.3)
            axes[0].add_patch(rect)
        
        axes[0].set_title(f'COSMOS Field - Spitzer Ch1\n{len(ra_cutouts)} Cutout Locations',
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X Pixel', fontsize=12)
        axes[0].set_ylabel('Y Pixel', fontsize=12)
        
        coverage_text = (f'Total cutouts: {len(ra_cutouts)}\n'
                        f'Cutout size: {cutout_size}×{cutout_size} pix\n'
                        f'Field: {self.spitzer_data.shape[1]}×{self.spitzer_data.shape[0]} pix')
        axes[0].text(0.02, 0.98, coverage_text, transform=axes[0].transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # --- RIGHT: Sky coverage ---
        axes[1].scatter(ra_cutouts, dec_cutouts, c='red', s=5, alpha=0.5,
                       edgecolors='none', rasterized=True)
        
        ra_range = ra_cutouts.max() - ra_cutouts.min()
        dec_range = dec_cutouts.max() - dec_cutouts.min()
        
        bbox = Rectangle((ra_cutouts.min(), dec_cutouts.min()),
                        ra_range, dec_range,
                        linewidth=2, edgecolor='blue', facecolor='none', linestyle='--',
                        label=f'Coverage: {ra_range:.3f}° × {dec_range:.3f}°')
        axes[1].add_patch(bbox)
        
        cosmos_ra, cosmos_dec = 150.1, 2.2
        axes[1].plot(cosmos_ra, cosmos_dec, 'b*', markersize=20,
                    label='COSMOS Center', markeredgecolor='white', markeredgewidth=1)
        
        axes[1].set_xlabel('RA (degrees)', fontsize=12)
        axes[1].set_ylabel('Dec (degrees)', fontsize=12)
        axes[1].set_title(f'Sky Coverage - {len(ra_cutouts)} Training Cutouts',
                         fontsize=14, fontweight='bold')
        axes[1].legend(loc='upper right', fontsize=11)
        axes[1].grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        axes[1].set_aspect('equal', adjustable='box')
        axes[1].invert_xaxis()
        
        plt.tight_layout()
        output_file = self.output_paths['visualizations'] / 'coverage_map.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ Saved coverage map: {output_file}")
    
    def create_statistics_visualization(self, X_wise: np.ndarray, Y_spitzer: np.ndarray,
                                        sources_info: List[Dict]):
        """Create detailed statistics visualization."""
        self.logger.info("Creating statistics visualization...")
        
        ra_cutouts = np.array([s['ra'] for s in sources_info])
        dec_cutouts = np.array([s['dec'] for s in sources_info])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Density heatmap
        H, xedges, yedges = np.histogram2d(ra_cutouts, dec_cutouts, bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = axes[0, 0].imshow(H.T, origin='lower', extent=extent, cmap='hot',
                              aspect='auto', interpolation='gaussian')
        axes[0, 0].set_xlabel('RA (degrees)', fontsize=11)
        axes[0, 0].set_ylabel('Dec (degrees)', fontsize=11)
        axes[0, 0].set_title('Cutout Density Map', fontsize=12, fontweight='bold')
        axes[0, 0].invert_xaxis()
        plt.colorbar(im, ax=axes[0, 0], label='Number of cutouts')
        
        # RA distribution
        axes[0, 1].hist(ra_cutouts, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(ra_cutouts.mean(), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {ra_cutouts.mean():.3f}°')
        axes[0, 1].set_xlabel('RA (degrees)', fontsize=11)
        axes[0, 1].set_ylabel('Count', fontsize=11)
        axes[0, 1].set_title('RA Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Dec distribution
        axes[1, 0].hist(dec_cutouts, bins=50, color='forestgreen', edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(dec_cutouts.mean(), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {dec_cutouts.mean():.3f}°')
        axes[1, 0].set_xlabel('Dec (degrees)', fontsize=11)
        axes[1, 0].set_ylabel('Count', fontsize=11)
        axes[1, 0].set_title('Dec Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary text - UPDATED with unit info
        axes[1, 1].axis('off')
        ra_range = ra_cutouts.max() - ra_cutouts.min()
        dec_range = dec_cutouts.max() - dec_cutouts.min()
        
        summary_text = f"""
DATASET SUMMARY
{'='*40}

Training Samples: {len(ra_cutouts):,}

Input (WISE W1):
  • Shape: {X_wise.shape[1]}×{X_wise.shape[2]} pixels
  • Pixel scale: {self.config['data']['wise']['pixel_scale']}"/pixel
  • FOV: ~38.5" × 38.5"
  • Units: DN (MAGZP=22.5, Vega)
  • Wavelength: 3.4 μm

Target (Spitzer Ch1):
  • Shape: {Y_spitzer.shape[1]}×{Y_spitzer.shape[2]} pixels
  • Pixel scale: {self.config['data']['spitzer']['pixel_scale']}"/pixel
  • FOV: ~38.4" × 38.4"
  • Units: MJy/sr
  • Wavelength: 3.6 μm

Super-resolution: {Y_spitzer.shape[1] / X_wise.shape[1]:.1f}×

Coverage:
  • RA: {ra_cutouts.min():.3f}° - {ra_cutouts.max():.3f}°
  • Dec: {dec_cutouts.min():.3f}° - {dec_cutouts.max():.3f}°
  • Area: {ra_range * dec_range:.4f} deg²

{'='*40}
⚠ NOTE: Units differ! Convert WISE DN → MJy/sr
  before normalization in preprocessing step.
        """
        
        axes[1, 1].text(0.1, 0.95, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        output_file = self.output_paths['visualizations'] / 'dataset_statistics.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ Saved statistics: {output_file}")
    
    def run(self):
        """Execute the complete pipeline."""
        try:
            # Load data
            self.load_spitzer_data()
            self.load_wise_data()
            self.load_catalog()
            
            # Create dataset
            X_wise, Y_spitzer, sources_info = self.create_dataset()
            
            # Save dataset
            dataset_file = self.save_dataset(X_wise, Y_spitzer, sources_info)
            
            # Create visualizations
            self.create_alignment_visualization(X_wise, Y_spitzer, sources_info)
            self.create_coverage_visualization(X_wise, Y_spitzer, sources_info)
            self.create_statistics_visualization(X_wise, Y_spitzer, sources_info)
            
            # Final summary
            self.logger.info("="*70)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("="*70)
            self.logger.info(f"Dataset: {dataset_file}")
            self.logger.info(f"Visualizations: {self.output_paths['visualizations']}")
            self.logger.info(f"Logs: {self.output_paths['logs']}")
            self.logger.info("")
            self.logger.info("⚠ IMPORTANT: Run preprocessing with unit conversion!")
            self.logger.info("  WISE data is in DN, Spitzer in MJy/sr")
            self.logger.info("="*70)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Create paired Spitzer-WISE cutouts for super-resolution training'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to configuration JSON file (default: config.json)'
    )
    
    args = parser.parse_args()
    
    creator = PairedDatasetCreator(args.config)
    success = creator.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()