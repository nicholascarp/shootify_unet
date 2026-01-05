# V2.2 Cloth Color Correction Model

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Deep learning model for correcting color degradation in virtual try on images using cloth references.**

**Performance:** 80% success rate | Color MAE 0.034 | PSNR 26.50 dB | 380K parameters | ~3s inference (CPU)

---
## **Installation**

### **Requirements**

```bash
Python >= 3.8
PyTorch >= 2.0
torchvision
PIL (Pillow)
numpy
pandas
tqdm
matplotlib
```

### **Setup**

```bash
# Clone repository
git clone <https://github.com/nicholascarp/shootify_unet.git>
cd shootify_unet

# Install dependencies
pip install torch torchvision
pip install pillow numpy pandas tqdm matplotlib

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

---
## **Quick Start**

**Complete workflow in 5 steps:**

```bash
# 1. Create manifests
python scripts/create_cloth_manifest.py

# 2. Validate degradation (optional but recommended)
python scripts/validate_degradation_comprehensive.py --num-samples 1000

# 3. Train model
python scripts/train.py \
    --train-manifest data/train_manifest.csv \
    --test-manifest data/test_manifest.csv \
    --epochs 15

# 4. Evaluate
python scripts/evaluate_model_minimal.py \
    --checkpoint outputs/model_best.pth \
    --test-manifest data/test_manifest.csv

# 5. Visualize results
python scripts/batch_inference_test.py \
    --checkpoint outputs/model_best.pth \
    --num-samples 15
```

**Expected time:** approximately 2 days (mostly training: 40 hours on CPU)

**Expected output:**
```
Step 1: data/train_manifest.csv (11,647 samples), data/test_manifest.csv (2,032 samples)
Step 2: EXCELLENT! Degradation works reliably (<1% failures)
Step 3: outputs/model_best.pth (380K params, ~1.5 MB)
Step 4: Color MAE: 0.034, PSNR: 26.50 dB, Success: 80%
Step 5: outputs/batch_inference_FIXED.png (7-column visualization)
```

---

## **Performance Summary**

| Metric | Value 
|--------|-------|
| **Success Rate** | 80% |
| **Color MAE** | 0.034 | 
| **PSNR** | 26.50 dB | 
| **Model Size** | 380K params | 
| **Inference Speed** | ~3s/image (CPU) |

**Comparison with previous versions:**

| Model | Color MAE | PSNR (dB) | Improvement |
|-------|-----------|-----------|-------------|
| V2 | 0.084 | 17.14 | Baseline |
| V2.1 | 0.068 | 18.81 | +19% |
| **V2.2** | **0.034** | **26.50** | **+50%** |

---


## **Key Features**

### **1. Lighting Aware Cloth Preprocessing** 
- Preserves lighting pattern from on model garment
- Combines color + lighting for realistic cloth reference

### **2. Two Component Degradation**
- **Multiplicative component**: Works on bright colors
- **Additive component**: Works on dark colors
- <1% failure rate on all colors

### **3. Minimal Architecture**
- **380K parameters** (vs 3.7M in standard U-Net)
- 7-channel input: RGB image + mask + RGB cloth
- Identity skip connection for structure preservation

### **4. Full Pipeline**
- Manifest Creation
- Training with validation
- Evaluation with correct metrics
- Batch inference
- Multiple validation tools

---

## 📁 **Project Structure**

```
.
├── src/
│   ├── data/
│   │   ├── dataset.py           # Dataset with lighting-aware cloth
│   │   └── degradation.py       # Two-component color degradation
│   ├── models/
│   │   └── fast_unet.py         # MinimalUNet architecture
│   └── losses/
│       └── simple_loss.py       # Simple structure loss
├── scripts/
│   ├── create_cloth_manifest.py # Create train/test manifests
│   ├── train.py                 # Training script
│   ├── evaluate_model_minimal.py # Evaluation (CORRECT metrics)
│   ├── batch_inference_test.py  # Batch inference with visualization
│   ├── test_cloth_sensitivity.py # Test if model uses cloth reference
│   ├── test_minimal_model.py    # Speed benchmarking
│   ├── test_degradation.py      # Visual degradation test
│   └── validate_degradation_comprehensive.py # Statistical validation
├── data/
│   ├── train_manifest.csv       # Training data manifest
│   └── test_manifest.csv        # Test data manifest
├── raw_data/
│   ├── train/
│   │   ├── images/              # On-model images
│   │   ├── cloth/               # Cloth references
│   │   └── image-parse-v3/      # Segmentation masks (PNG)
│   └── test/
│       └── (same structure)
└── outputs/
    └── (model checkpoints, visualizations, metrics)
```

---

## **Detailed Usage Guide**

### **1. Data Preparation**

#### **Expected Data Structure:**
```
raw_data/
├── train/
│   ├── images/              # On model JPG images
│   ├── cloth/               # Cloth JPG images  
│   └── image-parse-v3/      # Segmentation PNG masks (class 5 = upper garment)
└── test/
    └── (same structure)
```

#### **Create Manifests:**

```bash
python scripts/create_cloth_manifest.py
```


---

### **2. Training**

#### **Speed Test (Optional but Recommended):**

```bash
# Estimate training time before starting
python scripts/test_minimal_model.py
```

**Expected Output:**
```
Average: 3.25s per batch
Expected per epoch: 157.7 minutes (2.6 hours)
Total training (15 epochs): 39.4 hours (1.6 days)

Ok Fast enough to train!
```

#### **Train Model:**

```bash
python scripts/train.py \
    --train-manifest data/train_manifest.csv \
    --test-manifest data/test_manifest.csv \
    --epochs 15 \
    --batch-size 8 \
    --lr 1e-4
```

**Training Settings:**
- **Degradation strength:** 0.5 (50% color shift)
- **Cloth mismatch:** 30% during training (for robustness)
- **Image size:** 256×256
- **Optimizer:** Adam (lr=1e-4)

**Output:**
- `outputs/checkpoints/epoch_*.pth` (every epoch)
- `outputs/model_best.pth` (best validation loss)
- `outputs/training_history.json` (loss curves)

**Training Time:** ~40 hours on CPU (1.6 days)

---

### **3. Evaluation**

#### **Full Evaluation:**

```bash
python scripts/evaluate_model_minimal.py \
    --checkpoint outputs/model_best.pth \
    --test-manifest data/test_manifest.csv \
    --num-vis 10
```

**Output:**
```
======================================================================
FINAL EVALUATION RESULTS - V2.2 LIGHTING AWARE MODEL
======================================================================


Masked Metrics:
  MSE:  0.000567
  PSNR: 26.50 dB
  Color MAE: 0.034000

```

**Files Created:**
- `outputs/evaluation_v2_2_final/metrics.json`
- `outputs/evaluation_v2_2_final/evaluation_samples.png` (10-sample visualization)

#### **Batch Inference Visualization:**

```bash
python scripts/batch_inference_test.py \
    --checkpoint outputs/model_best.pth \
    --num-samples 15
```

**Output:**
- 7-column visualization showing complete pipeline
- Per-sample improvement percentages
- Color-coded success indicators (green/orange/red)
- Saved to: `outputs/batch_inference_FIXED.png`

**Grid Layout:**
```
[Original] [Raw Cloth] [Lighting-Aware Cloth] [Degraded] [Mask] [Corrected] [Error×5]
```

---

### **4. Validation Tools**

#### **Test Cloth Sensitivity:**

```bash
# Verify model uses cloth reference (not ignoring it)
python scripts/test_cloth_sensitivity.py
```

**Tests:** Model with 8 different solid-color cloths (red, blue, green, etc.)

**Expected Result:**
```
MODEL USES CLOTH REFERENCE!
   Outputs change significantly with cloth color!
   Color transfer learned successfully!
```

#### **Validate Degradation:**

```bash
# Comprehensive degradation validation (1000 samples)
python scripts/validate_degradation_comprehensive.py --num-samples 1000
```

**Output:**
- Statistics by color range (dark/medium/bright)
- Failure analysis (degradation < 0.05)
- 4-panel visualization plot
- Automated verdict

**Expected Result:**
```
EXCELLENT! Degradation works reliably (<1% failures)
   Safe to proceed with training!
```

#### **Visual Degradation Test:**

```bash
# Quick visual check (6 samples)
python scripts/test_degradation.py
```

**Output:** 6-sample grid showing lighting-aware cloth and degradation

---

## **Architecture**

### **Model: MinimalUNet**

```
Input: [degraded_image (3), mask (1), cloth_reference (3)] → 7 channels

Encoder:
  Conv1: 7 → 64   (256×256)
  Conv2: 64 → 128 (128×128)
  Conv3: 128 → 256 (64×64)
  Conv4: 256 → 512 (32×32) ← Bottleneck

Decoder:
  Up1: 512 → 256 (64×64)   + Skip from Conv3
  Up2: 256 → 128 (128×128) + Skip from Conv2
  Up3: 128 → 64  (256×256) + Skip from Conv1
  
Output: 64 → 3 channels (RGB corrected image)

Identity Skip: degraded_image → added to final output
```

**Total Parameters:** 379,875

### **Loss Function: Simple Structure Loss**

```python
loss = MSE(output, gt) + λ * MSE(output * mask, gt * mask)
```

- Global MSE: Entire image consistency
- Masked MSE: Garment region accuracy
- λ = 2.0: Higher weight on garment region



---

## **Future Improvements**

### **Potential Enhancements:**

1. **Attention Mechanisms**
   - Add self-attention to better capture cloth patterns
   - Could improve complex pattern transfer

2. **Multi-Scale Processing**
   - Process at multiple resolutions
   - Better detail preservation

3. **Data Augmentation Strategies**
   - Random Degradation Strength
   - Color Shift Direction Sampling
   - Lighting Variations
   - Geometric Augmentation

4. **Larger Datasets**
   - Current: 11,647 training samples
   - More data → Better generalization

5. **Loss Function Enhancements**
   - Perceptual Loss

---



## **License**

MIT License - See LICENSE file for details


---

## **Contact**

For questions or issues:
- Email: nicholas.carp@outlook.it


---

## **References**

1. **U-Net:** Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.

2. **VITON-HD:** Choi, S., Park, S., Lee, M., & Choo, J. (2021). VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization.

3. **Color Correction:** Deep learning approaches to color correction and enhancement in computer vision.

---


Last updated: January 2026
