#  WISE-to-Spitzer Super-Resolution


**Deep learning super-resolution to enhance WISE 3.4μm images to Spitzer 3.6μm resolution**


This Figure presents examples of super-resolution results obtained on previously unseen test data, showing the model’s ability to generalize beyond the training distribution. Low-resolution WISE W1 (3.4 μm) input cutouts (14×14 pixels) are transformed into high-resolution predictions (64×64 pixels) and compared directly against Spitzer IRAC Channel 1 (3.6 μm) observations, which are used exclusively for evaluation.

From left to right, each row shows the WISE input, the model’s super-resolved output, the corresponding Spitzer ground truth, and the absolute error map. All panels are displayed in physical surface brightness units (MJy sr⁻¹). Quantitative image quality metrics, including the Structural Similarity Index (SSIM) and Mean Absolute Error (MAE), are reported for each example.

Despite the large resolution gap between WISE and Spitzer, the model accurately reconstructs compact sources, preserves relative photometric ordering, and recovers extended low-surface-brightness structure. Residual errors are primarily confined to the cores of bright sources, consistent with differences in point-spread functions and sub-pixel centering between the two instruments. These results demonstrate that the network learns physically meaningful mappings rather than memorizing training examples, supporting its applicability to enhancing archival WISE imaging in regions lacking high-resolution coverage.

<p align="center">
  <img src="images/eval.png" alt="WISE to Spitzer Super-Resolution" width="800"/>
</p>

---

## Project Overview

This project presents a deep learning approach to enhance WISE W1 (3.4μm) images to match Spitzer IRAC Ch1 (3.6μm) resolution, achieving **4.6× spatial super-resolution**.



---

## Architecture

### Enhanced Residual Channel Attention Network (Enhanced RCAN)

```
Input (14×14×1)
      │
      ▼
┌─────────────────────────────────────┐
│   Multi-Scale Feature Extraction    │
│   (3×3, 5×5, upsampled features)    │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│      4 Residual Groups × 8 RCAB     │
│   (Channel Attention + Skip Conn)   │
│         Total: 32 RCAB blocks       │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│      Progressive Upsampling         │
│   14→28→56→112→64 (center crop)     │
│      (Sub-pixel convolution)        │
└─────────────────────────────────────┘
      │
      ▼
Output (64×64×1)
```

### Key Components

- **Residual Channel Attention Blocks (RCAB)**: Squeeze-and-excitation attention with residual scaling
- **Progressive Upsampling**: Multi-stage 2× upsampling with intermediate supervision
- **Source-Focused Loss**: Weighted Huber + SSIM + Gradient loss


## Data

### Training Data

- **Field**: COSMOS (2 deg²)
- **Training Samples**: 168,226 paired cutouts
- **Test Samples**: 29,687 paired cutouts
- **Selection**: SNR > 5 in IRAC 3.6μm, IRAC CH1 mag <25, lp_type == 0 (i.e., only galaxies)

### Preprocessing Pipeline

```
WISE (DN)                          Spitzer (MJy/sr)
    │                                    │
    ▼                                    │
Convert to MJy/sr                        │
(DN → Vega mag → AB mag → Jy → MJy/sr)   │
    │                                    │
    ▼                                    ▼
    └──────────► Asinh Normalization ◄───┘
                        │
                        ▼
                 Normalized Data
                   (~[0, 1])
```

---


## Contact

**Saeed Rezaee, Ph.D.**  
University of California, Riverside  
📧 sreza003@ucr.edu

---

