#!/usr/bin/env python3
"""
Smart inference that automatically picks a test image from manifest
"""

import sys
import csv
import argparse
from pathlib import Path

# Run the regular inference with auto-selected image
def main():
    parser = argparse.ArgumentParser(description='Run inference on a test image from manifest')
    parser.add_argument('--checkpoint', type=str, default='outputs/model.pth', help='Model checkpoint')
    parser.add_argument('--test-manifest', type=str, default='data/test_manifest.csv', help='Test manifest')
    parser.add_argument('--index', type=int, default=0, help='Which test image to use (default: 0 = first)')
    parser.add_argument('--output', type=str, help='Output path (optional)')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    
    args = parser.parse_args()
    
    # Read manifest
    with open(args.test_manifest, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if args.index >= len(rows):
        print(f"❌ Error: Index {args.index} out of range (manifest has {len(rows)} images)")
        sys.exit(1)
    
    # Get the image
    row = rows[args.index]
    image_path = row['image']
    mask_path = row['mask_npy']
    
    # Create output path if not specified
    if args.output:
        output_path = args.output
    else:
        image_name = Path(image_path).stem
        output_path = f'outputs/corrected_{image_name}.jpg'
    
    # Build command
    cmd_parts = [
        'python', 'scripts/inference.py',
        '--checkpoint', args.checkpoint,
        '--degraded', image_path,
        '--mask', mask_path,
        '--reference', image_path,
        '--output', output_path
    ]
    
    if args.visualize:
        cmd_parts.append('--visualize')
    
    # Print info
    print("="*70)
    print("AUTO INFERENCE")
    print("="*70)
    print(f"📊 Using test image {args.index + 1}/{len(rows)}")
    print(f"🖼️  Image: {Path(image_path).name}")
    print(f"📁 Output: {output_path}")
    print("="*70)
    
    # Run inference
    import subprocess
    subprocess.run(cmd_parts)

if __name__ == '__main__':
    main()
