#!/usr/bin/env python3
"""
Data preparation script: Convert PNG masks to NPY and create proper manifests

This script:
1. Reads your existing manifests (with PNG masks)
2. Converts PNG masks to binary NPY files
3. Creates new manifests with correct format (image, mask_npy)
"""

import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


# Upper garment class IDs (standard for VITON-HD/CIHP parsing)
UPPER_CLASS_IDS = {5, 6, 7}  # upper-clothes, dress, coat


def read_indexed_png(path):
    """
    Read indexed/paletted PNG and return label map
    
    Args:
        path: Path to PNG file
    
    Returns:
        numpy array [H, W] with class labels
    """
    im = Image.open(path)
    
    if im.mode == "P":
        # Paletted image - get indices directly
        return np.array(im, dtype=np.uint8)
    
    # Fallback for RGB images (rare)
    arr = np.array(im.convert("RGB"))
    colors = arr.reshape(-1, 3)
    uniq, inv = np.unique(colors, axis=0, return_inverse=True)
    return inv.reshape(arr.shape[:2]).astype(np.int32)


def mask_from_labels(label_map, keep_ids):
    """
    Create binary mask from label map
    
    Args:
        label_map: [H, W] array with class labels
        keep_ids: Set of class IDs to keep
    
    Returns:
        Boolean array [H, W]
    """
    return np.isin(label_map, list(keep_ids))


def convert_manifest(
    input_manifest,
    output_manifest,
    output_mask_dir,
    image_dir=None,
    mask_dir=None
):
    """
    Convert manifest with PNG masks to NPY masks
    
    Args:
        input_manifest: Path to input CSV (with PNG masks)
        output_manifest: Path to output CSV (with NPY masks)
        output_mask_dir: Directory to save NPY masks
        image_dir: Optional base directory for images (to convert relative paths)
        mask_dir: Optional base directory for masks (to convert relative paths)
    """
    input_manifest = Path(input_manifest)
    output_manifest = Path(output_manifest)
    output_mask_dir = Path(output_mask_dir)
    
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Converting: {input_manifest.name}")
    print(f"{'='*70}")
    
    # Read input manifest
    with open(input_manifest, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Found {len(rows)} entries")
    
    # Process each row
    output_rows = []
    errors = []
    
    for row in tqdm(rows, desc="Converting masks"):
        try:
            # Get paths from row
            # Support both old format (id,image,onmodel_mask,...) and new format (image,parse,mask_npy)
            if 'image' in row:
                image_path = row['image']
            else:
                raise ValueError("No 'image' column found")
            
            # Try different mask column names
            if 'onmodel_mask' in row:
                mask_path = row['onmodel_mask']
            elif 'parse' in row:
                mask_path = row['parse']
            elif 'mask' in row:
                mask_path = row['mask']
            else:
                raise ValueError("No mask column found")
            
            # Convert Google Drive paths to local paths if needed
            if '/content/drive/MyDrive/' in image_path and image_dir:
                # Extract just the filename
                image_filename = Path(image_path).name
                image_path = str(Path(image_dir) / image_filename)
            
            if '/content/drive/MyDrive/' in mask_path and mask_dir:
                # Extract just the filename
                mask_filename = Path(mask_path).name
                mask_path = str(Path(mask_dir) / mask_filename)
            
            # Check if files exist
            image_path = Path(image_path)
            mask_path = Path(mask_path)
            
            if not image_path.exists():
                errors.append(f"Image not found: {image_path}")
                continue
            
            if not mask_path.exists():
                errors.append(f"Mask not found: {mask_path}")
                continue
            
            # Read PNG mask and convert to binary NPY
            label_map = read_indexed_png(mask_path)
            binary_mask = mask_from_labels(label_map, UPPER_CLASS_IDS)
            
            # Save as NPY
            mask_stem = mask_path.stem
            output_npy_path = output_mask_dir / f"{mask_stem}.npy"
            np.save(output_npy_path, binary_mask)
            
            # Add to output
            output_rows.append({
                'image': str(image_path.resolve()),
                'mask_npy': str(output_npy_path.resolve())
            })
            
        except Exception as e:
            errors.append(f"Error processing {row.get('id', 'unknown')}: {e}")
    
    # Write output manifest
    with open(output_manifest, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'mask_npy'])
        writer.writeheader()
        writer.writerows(output_rows)
    
    # Print summary
    print(f"\n✅ Conversion complete!")
    print(f"   Input: {len(rows)} entries")
    print(f"   Output: {len(output_rows)} entries")
    print(f"   Errors: {len(errors)}")
    print(f"   Manifest saved to: {output_manifest}")
    print(f"   Masks saved to: {output_mask_dir}")
    
    if errors and len(errors) < 10:
        print(f"\n⚠️  Errors:")
        for err in errors:
            print(f"   - {err}")
    elif errors:
        print(f"\n⚠️  {len(errors)} errors occurred (showing first 10):")
        for err in errors[:10]:
            print(f"   - {err}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert manifests from PNG masks to NPY masks'
    )
    parser.add_argument(
        '--input-manifest',
        type=str,
        required=True,
        help='Input manifest CSV with PNG masks'
    )
    parser.add_argument(
        '--output-manifest',
        type=str,
        required=True,
        help='Output manifest CSV with NPY masks'
    )
    parser.add_argument(
        '--output-mask-dir',
        type=str,
        required=True,
        help='Directory to save NPY mask files'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Base directory for images (optional, for path conversion)'
    )
    parser.add_argument(
        '--mask-dir',
        type=str,
        help='Base directory for PNG masks (optional, for path conversion)'
    )
    
    args = parser.parse_args()
    
    convert_manifest(
        input_manifest=args.input_manifest,
        output_manifest=args.output_manifest,
        output_mask_dir=args.output_mask_dir,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir
    )
    
    print(f"\n{'='*70}")
    print("🎉 ALL DONE!")
    print(f"{'='*70}")
    print(f"\nYou can now use the new manifest for training:")
    print(f"  {args.output_manifest}")


if __name__ == '__main__':
    main()
