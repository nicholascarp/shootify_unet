#!/usr/bin/env python3
"""
Create manifest CSV pairing on-model images with cloth images
Uses PNG masks directly (no conversion needed!)
"""
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def create_manifest_for_split(
    split_name: str,
    on_model_dir: str,
    cloth_dir: str,
    mask_dir: str,
    output_csv: str
):
    """
    Create manifest for a single split (train or test)
    Points directly to PNG masks - no NPY needed!
    """
    print(f"\n{'='*70}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'='*70}")
    
    on_model_dir = Path(on_model_dir)
    cloth_dir = Path(cloth_dir)
    mask_dir = Path(mask_dir)
    
    print(f"\n📂 Directories:")
    print(f"  On-model: {on_model_dir}")
    print(f"  Cloth: {cloth_dir}")
    print(f"  Masks: {mask_dir}")
    
    # Get all on-model images
    on_model_files = sorted(list(on_model_dir.glob('*.jpg')))
    
    print(f"\n🔍 Found {len(on_model_files)} on-model images")
    print(f"   Processing...")
    
    pairs = []
    missing_cloth = 0
    missing_mask = 0
    
    for on_model_path in tqdm(on_model_files, desc="   Pairing"):
        # Get filename without extension
        base_name = on_model_path.stem  # e.g., "00000_00"
        
        # Construct corresponding paths
        cloth_path = cloth_dir / f"{base_name}.jpg"
        
        # Mask as PNG (not NPY!) - CHANGED!
        mask_path = mask_dir / f"{base_name}.png"
        
        # Check if both exist
        if cloth_path.exists() and mask_path.exists():
            pairs.append({
                'image': str(on_model_path),
                'mask_path': str(mask_path),  # Note: 'mask_path' not 'mask_npy'
                'cloth_image': str(cloth_path)
            })
        else:
            if not cloth_path.exists():
                missing_cloth += 1
            if not mask_path.exists():
                missing_mask += 1
    
    # Report
    print(f"\n📊 Results:")
    print(f"  ✅ Complete pairs: {len(pairs)}")
    
    if missing_cloth > 0:
        print(f"  ⚠️  Missing cloth images: {missing_cloth}")
    
    if missing_mask > 0:
        print(f"  ⚠️  Missing masks: {missing_mask}")
    
    if len(pairs) == 0:
        print(f"\n❌ ERROR: No valid pairs found for {split_name}!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(pairs)
    
    # Save manifest
    df.to_csv(output_csv, index=False)
    
    print(f"\n💾 Saved manifest:")
    print(f"   {output_csv}")
    
    # Show sample
    print(f"\n📋 Sample entries:")
    sample_df = df.head(3)
    for idx, row in sample_df.iterrows():
        print(f"\n  Entry {idx + 1}:")
        print(f"    On-model: {Path(row['image']).name}")
        print(f"    Cloth: {Path(row['cloth_image']).name}")
        print(f"    Mask: {Path(row['mask_path']).name}")
    
    return df


def main():
    print("="*70)
    print("CREATING CLOTH MANIFESTS (PNG MASKS)")
    print("="*70)
    print("\n💡 Using PNG masks directly - no NPY conversion needed!")
    print("   Saves ~12 GB of disk space!\n")
    
    base_dir = Path('raw_data')
    output_dir = Path('data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process TRAIN split
    train_df = create_manifest_for_split(
        split_name='train',
        on_model_dir=base_dir / 'train' / 'images',
        cloth_dir=base_dir / 'train' / 'cloth',
        mask_dir=base_dir / 'train' / 'image-parse-v3',  # PNG masks!
        output_csv=output_dir / 'train_manifest.csv'
    )
    
    # Process TEST split
    test_df = create_manifest_for_split(
        split_name='test',
        on_model_dir=base_dir / 'test' / 'images',
        cloth_dir=base_dir / 'test' / 'cloth',
        mask_dir=base_dir / 'test' / 'image-parse-v3',  # PNG masks!
        output_csv=output_dir / 'test_manifest.csv'
    )
    
    # Final Summary
    print(f"\n{'='*70}")
    print("✅ MANIFEST CREATION COMPLETE")
    print(f"{'='*70}")
    
    if train_df is not None:
        print(f"\n📊 TRAIN: {len(train_df):,} pairs")
        print(f"   → data/train_manifest.csv")
    
    if test_df is not None:
        print(f"\n📊 TEST: {len(test_df):,} pairs")
        print(f"   → data/test_manifest.csv")
    
    if train_df is not None and test_df is not None:
        print(f"\n{'='*70}")
        print("🚀 READY TO TRAIN!")
        print(f"{'='*70}")
        print("\n💾 Disk space saved: ~12 GB (no NPY files needed!)")
        print("\nRun:")
        print("  python scripts/train.py \\")
        print("      --train-manifest data/train_manifest.csv \\")
        print("      --test-manifest data/test_manifest.csv \\")
        print("      --epochs 10 \\")
        print("      --batch-size 4 \\")
        print("      --output-dir outputs/cloth_reference_model")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()
