#!/usr/bin/env python3
"""
Test model on IGPair dataset from HuggingFace (IMAGDressing/IGPair)
============================================================================
CORRECT IMPLEMENTATION:
  - Still-life cloth image → Reference (ground truth color)
  - On-model image → Target to correct (with simulated degradation)
  - Body mask → Segmentation of clothing region on model

This matches the actual use case:
  1. AI generates on-model image (with color artifacts)
  2. We have the original still-life cloth (correct colors)
  3. Model corrects the on-model image using cloth as reference

Dataset structure:
    IGPair_dataset/
    ├── clothes/        # Still-life flat garment images (REFERENCE)
    ├── [model images]  # On-model photos (TARGET) - via annotations
    ├── body_mask/      # Segmentation masks for on-model images
    ├── densepose/      # Dense pose for on-model
    ├── openpose/       # OpenPose keypoints
    └── IGPair.json     # Annotations linking cloth ↔ on-model pairs

Requirements:
    pip install huggingface_hub torch matplotlib tqdm pillow pandas pyarrow
    
    You need to accept the license at:
    https://huggingface.co/datasets/IMAGDressing/IGPair
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download, hf_hub_download
from tqdm import tqdm
import argparse

# Try to import from src (when running from repo root)
try:
    from src.models.fast_unet import MinimalUNet
    from src.data.degradation import ColorDegradation
    print("✅ Imported from src/")
except ImportError:
    print("⚠️  Could not import from src/, using local definitions")
    
    import torch.nn as nn
    
    class MinimalUNet(nn.Module):
        """Minimal UNet for color correction (380K params)"""
        def __init__(self, in_channels=7, out_channels=3, features=32):
            super().__init__()
            
            self.enc1 = self._block(in_channels, features)
            self.enc2 = self._block(features, features * 2)
            self.enc3 = self._block(features * 2, features * 4)
            self.pool = nn.MaxPool2d(2, 2)
            
            self.bottleneck = self._block(features * 4, features * 8)
            
            self.up3 = nn.ConvTranspose2d(features * 8, features * 4, 2, 2)
            self.dec3 = self._block(features * 8, features * 4)
            self.up2 = nn.ConvTranspose2d(features * 4, features * 2, 2, 2)
            self.dec2 = self._block(features * 4, features * 2)
            self.up1 = nn.ConvTranspose2d(features * 2, features, 2, 2)
            self.dec1 = self._block(features * 2, features)
            
            self.final = nn.Conv2d(features, out_channels, 1)
        
        def _block(self, in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        def forward(self, x, mask, cloth):
            x = torch.cat([x, mask, cloth], dim=1)
            
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            
            b = self.bottleneck(self.pool(e3))
            
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            
            return torch.sigmoid(self.final(d1))
    
    class ColorDegradation:
        """Apply realistic color degradation to simulate generation artifacts"""
        @staticmethod
        def apply_realistic_degradation(image, mask, strength=0.5):
            mask_3ch = mask.expand(3, -1, -1) if mask.dim() == 3 else mask.unsqueeze(0).expand(3, -1, -1)
            
            mult_factor = 1.0 + (torch.rand(3, 1, 1) - 0.5) * strength
            add_factor = (torch.rand(3, 1, 1) - 0.5) * strength * 0.3
            
            degraded = image.clone()
            degraded = degraded * mult_factor + add_factor
            
            result = image * (1 - mask_3ch) + degraded * mask_3ch
            return result.clamp(0, 1)


def create_lighting_aware_cloth(cloth, on_model, mask):
    """
    Apply lighting-aware preprocessing to cloth image.
    
    Takes the mean color from still-life cloth and applies
    the lighting pattern from the on-model image.
    """
    cloth_mean_color = cloth.mean(dim=[1, 2])
    
    on_model_luminance = 0.299 * on_model[0] + 0.587 * on_model[1] + 0.114 * on_model[2]
    
    mask_bool = mask.squeeze(0) > 0.5
    if mask_bool.sum() > 0:
        lum_values = on_model_luminance[mask_bool]
        lum_min, lum_max = lum_values.min(), lum_values.max()
        
        if lum_max > lum_min:
            normalized_lum = (on_model_luminance - lum_min) / (lum_max - lum_min + 1e-6)
            normalized_lum = normalized_lum * 1.0 + 0.3
        else:
            normalized_lum = torch.ones_like(on_model_luminance)
    else:
        normalized_lum = torch.ones_like(on_model_luminance)
    
    cloth_with_lighting = cloth_mean_color.view(3, 1, 1) * normalized_lum.unsqueeze(0)
    return cloth_with_lighting.clamp(0, 1)


def pil_to_tensor(img, size=256):
    """Convert PIL image to normalized tensor [0, 1]"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def load_segmentation_mask(mask_path, size=256):
    """
    Load segmentation mask from IGPair body_mask folder.
    
    IGPair uses SCHP for segmentation with ATR-like labels:
        4 = Upper-clothes
        5 = Skirt  
        6 = Pants
        7 = Dress
    """
    mask_img = Image.open(mask_path)
    mask_img = mask_img.resize((size, size), Image.NEAREST)
    mask_array = np.array(mask_img, dtype=np.uint8)
    
    # Try upper-clothes first (label 4)
    upper_mask = (mask_array == 4).astype(np.float32)
    
    # If empty, try other clothing labels
    if upper_mask.sum() < 100:
        for label in [7, 5, 6, 3]:  # dress, skirt, pants, coat
            alt_mask = (mask_array == label).astype(np.float32)
            if alt_mask.sum() > 100:
                upper_mask = alt_mask
                break
    
    # If still empty, use any non-background
    if upper_mask.sum() < 100:
        upper_mask = (mask_array > 0).astype(np.float32)
    
    return torch.from_numpy(upper_mask).unsqueeze(0)


def download_igpair_dataset(cache_dir="./igpair_cache"):
    """
    Download IGPair dataset from HuggingFace.
    """
    print("\n📥 Downloading IGPair dataset from HuggingFace...")
    print("   Repository: IMAGDressing/IGPair")
    print("   ⚠️  Make sure you've accepted the license at:")
    print("   https://huggingface.co/datasets/IMAGDressing/IGPair")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        local_dir = snapshot_download(
            repo_id="IMAGDressing/IGPair",
            repo_type="dataset",
            local_dir=cache_dir,
            allow_patterns=[
                "IGPair_dataset/IGPair.json/*",
                "IGPair_dataset/body_mask/*",
                "IGPair_dataset/clothes/*",
                "*.json",
                "*.parquet",
            ],
            ignore_patterns=["*.zip", "*.zip.*"],
        )
        print(f"✅ Downloaded to: {local_dir}")
        return local_dir
    except Exception as e:
        print(f"\n❌ Error downloading: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Visit https://huggingface.co/datasets/IMAGDressing/IGPair")
        print("   2. Click 'Access repository' and accept the license")
        print("   3. Run: huggingface-cli login")
        raise


def load_annotations(dataset_dir):
    """
    Load IGPair annotations that link cloth ↔ on-model pairs.
    
    Annotation format:
    {
        "text": "caption",
        "image_file": "path/to/on_model.jpg",  
        "cloth_file": "path/to/cloth.jpg",
        "cloth_type": "upper_body" | "dress" | "lower_body"
    }
    """
    dataset_dir = Path(dataset_dir)
    annotations = []
    
    # Look for JSON annotations
    json_dir = dataset_dir / "IGPair_dataset" / "IGPair.json"
    if json_dir.exists() and json_dir.is_dir():
        json_files = list(json_dir.glob("*.json"))
        print(f"   Found {len(json_files)} JSON files in IGPair.json/")
        for jf in json_files:
            try:
                with open(jf) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        annotations.extend(data)
                    elif isinstance(data, dict):
                        annotations.append(data)
            except Exception as e:
                print(f"   ⚠️ Error loading {jf.name}: {e}")
    
    # Also try parquet files (IGPair uses parquet format)
    parquet_files = list(dataset_dir.rglob("*.parquet"))
    if parquet_files:
        try:
            import pandas as pd
            print(f"   Found {len(parquet_files)} parquet files")
            for pf in parquet_files:
                df = pd.read_parquet(pf)
                for _, row in df.iterrows():
                    ann = row.to_dict()
                    if 'image_file' in ann and 'cloth_file' in ann:
                        annotations.append(ann)
        except ImportError:
            print("   ⚠️ pandas/pyarrow not installed, skipping parquet files")
        except Exception as e:
            print(f"   ⚠️ Error loading parquet: {e}")
    
    # Try direct JSON file
    direct_json = dataset_dir / "IGPair_dataset" / "IGPair.json"
    if direct_json.exists() and direct_json.is_file():
        try:
            with open(direct_json) as f:
                data = json.load(f)
                if isinstance(data, list):
                    annotations.extend(data)
        except:
            pass
    
    return annotations


def find_valid_pairs(dataset_dir, annotations, max_pairs=100):
    """
    Find valid pairs where both cloth and on-model images exist.
    
    Returns list of dicts:
        - cloth_path: Still-life cloth image (REFERENCE)
        - model_path: On-model image (TARGET to correct)
        - mask_path: Segmentation mask for on-model
        - cloth_type: upper_body, dress, lower_body
    """
    dataset_dir = Path(dataset_dir)
    valid_pairs = []
    
    # Base directories
    base_dir = dataset_dir / "IGPair_dataset"
    clothes_dir = base_dir / "clothes"
    mask_dir = base_dir / "body_mask"
    
    # Alternative paths
    if not clothes_dir.exists():
        clothes_dir = dataset_dir / "clothes"
    if not mask_dir.exists():
        mask_dir = dataset_dir / "body_mask"
    
    print(f"\n🔍 Searching for valid pairs...")
    print(f"   Clothes dir: {clothes_dir} (exists: {clothes_dir.exists()})")
    print(f"   Mask dir: {mask_dir} (exists: {mask_dir.exists()})")
    print(f"   Annotations loaded: {len(annotations)}")
    
    # Process annotations
    for ann in tqdm(annotations[:max_pairs * 10], desc="Scanning annotations"):
        cloth_file = ann.get('cloth_file', '')
        model_file = ann.get('image_file', '')
        cloth_type = ann.get('cloth_type', 'unknown')
        
        if not cloth_file or not model_file:
            continue
        
        # Build full paths (annotations may have relative or partial paths)
        cloth_candidates = [
            dataset_dir / cloth_file,
            base_dir / cloth_file,
            clothes_dir / Path(cloth_file).name,
            dataset_dir / "IGPair_dataset" / cloth_file,
        ]
        
        model_candidates = [
            dataset_dir / model_file,
            base_dir / model_file,
            dataset_dir / "IGPair_dataset" / model_file,
        ]
        
        # Find cloth path
        cloth_path = None
        for cp in cloth_candidates:
            if cp.exists():
                cloth_path = cp
                break
        
        # Find model path
        model_path = None
        for mp in model_candidates:
            if mp.exists():
                model_path = mp
                break
        
        if not cloth_path or not model_path:
            continue
        
        # Find mask for on-model image
        model_stem = model_path.stem
        mask_candidates = [
            mask_dir / f"{model_stem}.png",
            mask_dir / f"{model_stem}.jpg",
            mask_dir / model_path.name.replace('.jpg', '.png'),
        ]
        
        mask_path = None
        for mp in mask_candidates:
            if mp.exists():
                mask_path = mp
                break
        
        valid_pairs.append({
            'cloth_path': str(cloth_path),
            'model_path': str(model_path),
            'mask_path': str(mask_path) if mask_path else None,
            'cloth_type': cloth_type,
            'name': model_stem
        })
        
        if len(valid_pairs) >= max_pairs:
            break
    
    # If no pairs from annotations, try to match by filename patterns
    if len(valid_pairs) == 0 and clothes_dir.exists():
        print("   ⚠️ No pairs from annotations, trying filename matching...")
        cloth_images = list(clothes_dir.glob("*.jpg")) + list(clothes_dir.glob("*.png"))
        
        for cloth_img in cloth_images[:max_pairs]:
            # Look for corresponding mask
            mask_path = mask_dir / f"{cloth_img.stem}.png" if mask_dir.exists() else None
            if mask_path and not mask_path.exists():
                mask_path = None
            
            valid_pairs.append({
                'cloth_path': str(cloth_img),
                'model_path': str(cloth_img),  # Use same image as fallback
                'mask_path': str(mask_path) if mask_path else None,
                'cloth_type': 'upper_body',
                'name': cloth_img.stem,
                'is_fallback': True  # Mark as fallback (not true pair)
            })
    
    return valid_pairs


def test_on_pairs(pairs, model, device, num_samples=15, img_size=256):
    """
    Test model on paired cloth + on-model images.
    
    CORRECT FLOW:
    1. Load still-life cloth (REFERENCE - correct colors)
    2. Load on-model image (TARGET)
    3. Load segmentation mask for on-model
    4. Apply degradation to on-model (simulating AI generation artifacts)
    5. Run model: degraded_on_model + mask + cloth_ref → corrected_on_model
    6. Compare corrected vs original on-model
    """
    print(f"\n🧪 Testing on {min(num_samples, len(pairs))} paired samples")
    print("   Flow: Still-life cloth (ref) + Degraded on-model → Corrected on-model")
    
    results = []
    
    # Filter valid pairs
    valid_pairs = [p for p in pairs if p.get('cloth_path') and Path(p['cloth_path']).exists()]
    
    if len(valid_pairs) == 0:
        print("❌ No valid pairs found!")
        return results
    
    # Check if we have true pairs or fallback
    has_true_pairs = any(
        p.get('model_path') and 
        p['model_path'] != p['cloth_path'] and 
        Path(p['model_path']).exists() 
        for p in valid_pairs
    )
    
    if has_true_pairs:
        print("   ✅ Using TRUE paired data (cloth ≠ on-model)")
    else:
        print("   ⚠️ Using FALLBACK mode (cloth = on-model, self-reference test)")
    
    # Sample
    sample_indices = np.random.choice(
        len(valid_pairs),
        min(num_samples, len(valid_pairs)),
        replace=False
    )
    
    for idx in tqdm(sample_indices, desc="Processing pairs"):
        pair = valid_pairs[idx]
        
        try:
            # ============================================
            # 1. Load STILL-LIFE CLOTH (Reference)
            # ============================================
            cloth_img = Image.open(pair['cloth_path'])
            cloth_tensor = pil_to_tensor(cloth_img, img_size)
            
            # ============================================
            # 2. Load ON-MODEL IMAGE (Target)
            # ============================================
            if pair.get('model_path') and Path(pair['model_path']).exists():
                model_img = Image.open(pair['model_path'])
                on_model_tensor = pil_to_tensor(model_img, img_size)
                is_true_pair = pair['model_path'] != pair['cloth_path']
            else:
                # Fallback: use cloth as on-model (self-reference)
                on_model_tensor = cloth_tensor.clone()
                is_true_pair = False
            
            # ============================================
            # 3. Load SEGMENTATION MASK (for on-model)
            # ============================================
            if pair.get('mask_path') and Path(pair['mask_path']).exists():
                mask = load_segmentation_mask(pair['mask_path'], img_size)
            else:
                # Create center mask as fallback
                mask = torch.zeros(1, img_size, img_size)
                h_start, h_end = img_size // 4, 3 * img_size // 4
                w_start, w_end = img_size // 4, 3 * img_size // 4
                mask[0, h_start:h_end, w_start:w_end] = 1.0
            
            # Skip if mask too small
            if mask.mean() < 0.05:
                continue
            
            # ============================================
            # 4. Create LIGHTING-AWARE CLOTH reference
            # ============================================
            cloth_ref = create_lighting_aware_cloth(cloth_tensor, on_model_tensor, mask)
            
            # ============================================
            # 5. Apply DEGRADATION to on-model (simulate AI artifacts)
            # ============================================
            degraded = ColorDegradation.apply_realistic_degradation(
                on_model_tensor, mask, strength=0.5
            )
            
            # ============================================
            # 6. Run MODEL inference
            # ============================================
            with torch.no_grad():
                output = model(
                    degraded.unsqueeze(0).to(device),
                    mask.unsqueeze(0).to(device),
                    cloth_ref.unsqueeze(0).to(device)
                ).squeeze(0).cpu()
            
            # ============================================
            # 7. Calculate METRICS
            # ============================================
            mask_bool = mask.squeeze() > 0.5
            if mask_bool.sum() > 0:
                # MAE
                deg_mae = (degraded[:, mask_bool] - on_model_tensor[:, mask_bool]).abs().mean().item()
                final_mae = (output[:, mask_bool] - on_model_tensor[:, mask_bool]).abs().mean().item()
                
                # PSNR
                mse_deg = ((degraded[:, mask_bool] - on_model_tensor[:, mask_bool]) ** 2).mean().item()
                mse_final = ((output[:, mask_bool] - on_model_tensor[:, mask_bool]) ** 2).mean().item()
                psnr_deg = 10 * np.log10(1.0 / (mse_deg + 1e-10))
                psnr_final = 10 * np.log10(1.0 / (mse_final + 1e-10))
                
                improvement = ((deg_mae - final_mae) / deg_mae * 100) if deg_mae > 0 else 0
            else:
                deg_mae, final_mae = 0, 0
                psnr_deg, psnr_final = 0, 0
                improvement = 0
            
            results.append({
                'name': pair['name'],
                'cloth_type': pair.get('cloth_type', 'unknown'),
                'is_true_pair': is_true_pair,
                'cloth': cloth_tensor,           # Still-life (reference)
                'on_model': on_model_tensor,     # Original on-model
                'cloth_ref': cloth_ref,          # Lighting-aware reference
                'degraded': degraded,            # Degraded on-model
                'output': output,                # Corrected on-model
                'mask': mask,
                'deg_mae': deg_mae,
                'final_mae': final_mae,
                'psnr_deg': psnr_deg,
                'psnr_final': psnr_final,
                'improvement': improvement
            })
            
        except Exception as e:
            print(f"   ⚠️ Error processing {pair['name']}: {e}")
            continue
    
    return results


def visualize_results(results, output_path):
    """Create visualization showing the full pipeline"""
    if len(results) == 0:
        print("❌ No results to visualize!")
        return
    
    n_samples = len(results)
    # 7 columns: cloth, on-model, cloth_ref, degraded, mask, corrected, error
    fig, axes = plt.subplots(n_samples, 7, figsize=(28, 4 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    def to_img(t):
        return t.permute(1, 2, 0).numpy().clip(0, 1)
    
    for i, res in enumerate(results):
        pair_type = "TRUE PAIR" if res['is_true_pair'] else "SELF-REF"
        
        # Column 0: Still-life cloth (reference)
        axes[i, 0].imshow(to_img(res['cloth']))
        axes[i, 0].set_title(f"Still-Life Cloth\n(Reference)", fontsize=9)
        axes[i, 0].axis('off')
        
        # Column 1: Original on-model
        axes[i, 1].imshow(to_img(res['on_model']))
        axes[i, 1].set_title(f"On-Model Original\n({pair_type})", fontsize=9)
        axes[i, 1].axis('off')
        
        # Column 2: Lighting-aware cloth reference
        axes[i, 2].imshow(to_img(res['cloth_ref']))
        axes[i, 2].set_title(f"Lighting-Aware\nCloth Ref", fontsize=9)
        axes[i, 2].axis('off')
        
        # Column 3: Degraded on-model
        axes[i, 3].imshow(to_img(res['degraded']))
        axes[i, 3].set_title(f"Degraded\nMAE: {res['deg_mae']:.3f}", fontsize=9)
        axes[i, 3].axis('off')
        
        # Column 4: Mask
        axes[i, 4].imshow(res['mask'].squeeze().numpy(), cmap='gray')
        mask_pct = res['mask'].mean() * 100
        axes[i, 4].set_title(f"Mask\n({mask_pct:.1f}%)", fontsize=9)
        axes[i, 4].axis('off')
        
        # Column 5: Corrected output
        axes[i, 5].imshow(to_img(res['output']))
        axes[i, 5].set_title(f"Corrected\nMAE: {res['final_mae']:.3f}", fontsize=9)
        axes[i, 5].axis('off')
        
        # Column 6: Error map
        error = (res['output'] - res['on_model']).abs() * 5
        color = 'green' if res['improvement'] > 50 else 'orange' if res['improvement'] > 0 else 'red'
        
        axes[i, 6].imshow(to_img(error))
        axes[i, 6].set_title(f"Error (5x)\n{res['improvement']:.0f}% impr.", 
                           fontsize=9, color=color, fontweight='bold')
        axes[i, 6].axis('off')
        
        # Add row label
        axes[i, 0].set_ylabel(f"{res['name'][:15]}\n({res['cloth_type']})", 
                             fontsize=8, rotation=0, ha='right', va='center')
    
    plt.suptitle(
        '🧪 EXTERNAL TEST: IGPair Dataset (IMAGDressing)\n'
        'Still-Life Cloth (Reference) → Correct Degraded On-Model Image\n'
        'Trained: VITON-HD (Zalando) | Tested: IGPair (Internet Fashion)',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization: {output_path}")


def print_summary(results):
    """Print comprehensive summary"""
    print("\n" + "=" * 70)
    print("📊 EXTERNAL TEST SUMMARY - IGPair Dataset")
    print("=" * 70)
    
    if len(results) == 0:
        print("❌ No results!")
        return
    
    # Separate true pairs from fallback
    true_pairs = [r for r in results if r['is_true_pair']]
    fallback = [r for r in results if not r['is_true_pair']]
    
    print(f"\n📈 Dataset Breakdown:")
    print(f"   True pairs (cloth ≠ on-model): {len(true_pairs)}")
    print(f"   Fallback (self-reference): {len(fallback)}")
    
    # Overall stats
    improvements = [r['improvement'] for r in results]
    avg_improvement = np.mean(improvements)
    success_rate = sum(1 for imp in improvements if imp > 50) / len(improvements) * 100
    
    print(f"\n📈 Overall Results:")
    print(f"   Samples tested: {len(results)}")
    print(f"   Average improvement: {avg_improvement:.1f}%")
    print(f"   Success rate (>50%): {success_rate:.0f}%")
    
    print(f"\n📏 Color MAE (lower is better):")
    print(f"   Degraded: {np.mean([r['deg_mae'] for r in results]):.4f}")
    print(f"   Corrected: {np.mean([r['final_mae'] for r in results]):.4f}")
    
    print(f"\n📏 PSNR (higher is better):")
    psnr_deg = np.mean([r['psnr_deg'] for r in results])
    psnr_final = np.mean([r['psnr_final'] for r in results])
    print(f"   Degraded: {psnr_deg:.2f} dB")
    print(f"   Corrected: {psnr_final:.2f} dB")
    print(f"   Improvement: +{psnr_final - psnr_deg:.2f} dB")
    
    # By cloth type
    print(f"\n📊 By Cloth Type:")
    for ctype in ['upper_body', 'dress', 'lower_body', 'unknown']:
        type_results = [r for r in results if r['cloth_type'] == ctype]
        if type_results:
            type_imp = np.mean([r['improvement'] for r in type_results])
            print(f"   {ctype}: {len(type_results)} samples, {type_imp:.1f}% avg")
    
    # Per-sample details
    print(f"\n📋 Per-Sample Results:")
    for r in results:
        status = '✅' if r['improvement'] > 50 else '⚠️' if r['improvement'] > 0 else '❌'
        pair_mark = '🔗' if r['is_true_pair'] else '🔄'
        print(f"   {status}{pair_mark} {r['name'][:20]:20s} | "
              f"MAE: {r['deg_mae']:.3f}→{r['final_mae']:.3f} | "
              f"{r['improvement']:+.0f}%")
    
    # Verdict
    print("\n" + "=" * 70)
    print("🎯 GENERALIZATION VERDICT")
    print("=" * 70)
    
    if success_rate >= 70:
        print("✅ EXCELLENT generalization!")
        print("   Model transfers well from VITON-HD to IGPair")
    elif success_rate >= 50:
        print("⚠️  GOOD generalization with some edge cases")
    else:
        print("❌ LIMITED generalization - may need fine-tuning")


def main():
    parser = argparse.ArgumentParser(
        description='Test color correction on IGPair (paired cloth + on-model)'
    )
    parser.add_argument('--model_path', type=str,
                       default='outputs/cloth_model_lighting_aware/model_best.pth')
    parser.add_argument('--cache_dir', type=str, default='./igpair_cache')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--num_samples', type=int, default=15)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--skip_download', action='store_true')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🧪 EXTERNAL TEST - IGPair Dataset (CORRECT IMPLEMENTATION)")
    print("=" * 70)
    print("\n📋 Test Flow:")
    print("   1. Still-life cloth → Reference (ground truth colors)")
    print("   2. On-model image → Target (to be corrected)")
    print("   3. Apply degradation → Simulate AI generation artifacts")
    print("   4. Model corrects on-model using cloth reference")
    print("   5. Compare: corrected vs original on-model")
    
    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"\n🖥️  Device: {device}")
    
    # Download
    if not args.skip_download or not Path(args.cache_dir).exists():
        try:
            dataset_dir = download_igpair_dataset(args.cache_dir)
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            return
    else:
        dataset_dir = args.cache_dir
        print(f"\n📂 Using cached: {dataset_dir}")
    
    # Load annotations
    print("\n📂 Loading annotations...")
    annotations = load_annotations(dataset_dir)
    print(f"   Loaded {len(annotations)} annotations")
    
    # Find valid pairs
    pairs = find_valid_pairs(dataset_dir, annotations, max_pairs=args.num_samples * 3)
    print(f"✅ Found {len(pairs)} valid pairs")
    
    if len(pairs) == 0:
        print("\n❌ No pairs found!")
        return
    
    # Load model
    print(f"\n🏗️  Loading model: {args.model_path}")
    model = MinimalUNet(in_channels=7)
    
    if Path(args.model_path).exists():
        ckpt = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print("✅ Model loaded")
    else:
        print(f"⚠️  Model not found, using random weights")
    
    model = model.to(device)
    model.eval()
    
    # Run test
    results = test_on_pairs(pairs, model, device, args.num_samples, args.img_size)
    
    if len(results) == 0:
        print("\n❌ No results!")
        return
    
    # Output
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_path = Path(args.output_dir) / 'external_test_igpair.png'
    visualize_results(results, str(output_path))
    
    print_summary(results)
    
    # Save JSON
    results_path = Path(args.output_dir) / 'external_test_igpair_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'summary': {
                'num_samples': len(results),
                'true_pairs': sum(1 for r in results if r['is_true_pair']),
                'avg_improvement': float(np.mean([r['improvement'] for r in results])),
                'success_rate': float(sum(1 for r in results if r['improvement'] > 50) / len(results)),
            },
            'results': [{
                'name': r['name'],
                'cloth_type': r['cloth_type'],
                'is_true_pair': r['is_true_pair'],
                'deg_mae': r['deg_mae'],
                'final_mae': r['final_mae'],
                'improvement': r['improvement'],
            } for r in results]
        }, f, indent=2)
    print(f"✅ Results saved: {results_path}")


if __name__ == '__main__':
    main()
