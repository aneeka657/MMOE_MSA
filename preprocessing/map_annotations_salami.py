#!/usr/bin/env python3
"""
Convert Beatles 7-class text annotations to .npy format
CORRECTED: Outputs 2D array format expected by preprocessing
"""

import os
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm

# Paths
INPUT_ANNOTATION_DIR = "/Scratch/repository/iahmad/salami_7class_annotations"
OUTPUT_ANNOTATION_DIR = "/Scratch/repository/iahmad/salami-mapping"

# ============================================================================
# FILENAME MATCHING
# ============================================================================
_filename_cache = {}

def build_filename_cache(annotation_dir):
    """Build cache of all annotation files."""
    global _filename_cache
    
    if _filename_cache:
        return
    
    print("🔍 Building filename cache...")
    
    all_files = []
    for ext in ['*.txt', '*.lab']:
        all_files.extend(glob.glob(os.path.join(annotation_dir, ext)))
        all_files.extend(glob.glob(os.path.join(annotation_dir, '**', ext), recursive=True))
    
    for filepath in all_files:
        filename = Path(filepath).stem
        variations = [
            filename, filename.lower(),
            filename.replace("'", ""),
            filename.replace("'", "'"),
            filename.replace("'", ""),
        ]
        
        for var in variations:
            _filename_cache[var] = filepath
            _filename_cache[var.lower()] = filepath
    
    print(f"✅ Cached {len(all_files)} files")

# ============================================================================
# CONVERT TEXT TO NPY (CORRECTED FORMAT)
# ============================================================================

def read_text_annotation(filepath):
    """
    Read text annotation file and convert to 2D numpy array.
    
    Input format (text file):
        0.0    intro
        15.5   verse
        45.2   chorus
        ...
    
    Output format (2D numpy array):
        array([[ 0.0, 'intro'],
               [15.5, 'verse'],
               [45.2, 'chorus'],
               ...], dtype=object)
        
    Shape: (n_segments, 2)
    - Column 0: onset time (float) in seconds
    - Column 1: section label (string)
    """
    segments = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Parse: timestamp\tlabel or timestamp label
                parts = line.split('\t') if '\t' in line else line.split(maxsplit=1)
                
                if len(parts) >= 2:
                    try:
                        time = float(parts[0])
                        label = parts[1].strip()
                        
                        # Skip 'end' markers
                        if label.lower() != 'end':
                            segments.append([time, label])
                    except ValueError:
                        continue
        
        if not segments:
            return None
        
        # Convert to 2D numpy array with dtype=object
        # This allows mixed types (float + string)
        annotation_array = np.array(segments, dtype=object)
        
        return annotation_array
    
    except Exception as e:
        print(f"⚠️ Error reading {filepath}: {e}")
        return None

def convert_all_annotations(input_dir, output_dir):
    """Convert all text annotations to .npy format."""
    
    print(f"\n{'='*70}")
    print("CONVERTING TEXT ANNOTATIONS TO .NPY FORMAT")
    print(f"{'='*70}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Build cache for filename matching
    build_filename_cache(input_dir)
    
    # Find all text annotation files
    annotation_files = []
    for ext in ['*.txt', '*.lab']:
        annotation_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    print(f"Found {len(annotation_files)} annotation files\n")
    
    converted = 0
    failed = 0
    
    for input_path in tqdm(annotation_files, desc="Converting annotations"):
        song_id = Path(input_path).stem
        output_path = os.path.join(output_dir, f"{song_id}.npy")
        
        try:
            # Read and convert
            annotation_array = read_text_annotation(input_path)
            
            if annotation_array is not None and len(annotation_array) > 0:
                # Verify shape
                assert annotation_array.shape[1] == 2, f"Expected 2 columns, got {annotation_array.shape}"
                
                # Save as .npy
                np.save(output_path, annotation_array, allow_pickle=True)
                converted += 1
            else:
                print(f"⚠️ Empty annotation: {song_id}")
                failed += 1
                
        except Exception as e:
            print(f"❌ Error converting {song_id}: {e}")
            failed += 1
    
    print(f"\n{'='*70}")
    print("CONVERSION SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Successfully converted: {converted}/{len(annotation_files)}")
    print(f"❌ Failed: {failed}/{len(annotation_files)}")
    print(f"\n📁 Output directory: {output_dir}/")

def verify_npy_files(output_dir, sample_count=5):
    """Verify the converted .npy files match expected format."""
    
    print(f"\n{'='*70}")
    print("VERIFYING NPY FILES")
    print(f"{'='*70}\n")
    
    npy_files = glob.glob(os.path.join(output_dir, '*.npy'))
    
    if not npy_files:
        print("❌ No .npy files found!")
        return
    
    print(f"Found {len(npy_files)} .npy files\n")
    print(f"Checking first {sample_count} files:\n")
    
    for npy_file in sorted(npy_files)[:sample_count]:
        song_id = Path(npy_file).stem
        
        try:
            data = np.load(npy_file, allow_pickle=True)
            
            print(f"📄 {song_id}")
            print(f"   Shape: {data.shape} (expected: (n_segments, 2))")
            print(f"   Dtype: {data.dtype} (expected: object)")
            
            # Verify format
            assert data.shape[1] == 2, f"Expected 2 columns!"
            assert isinstance(data[0, 0], (float, np.floating)), "Column 0 should be float (time)"
            assert isinstance(data[0, 1], str), "Column 1 should be string (label)"
            
            print(f"   ✅ Format correct!")
            print(f"   First 3 segments:")
            
            for i in range(min(3, len(data))):
                print(f"      {i+1}. Time={float(data[i, 0]):6.2f}s, Label='{data[i, 1]}'")
            print()
            
        except Exception as e:
            print(f"❌ Error verifying {song_id}: {e}\n")
    
    print(f"✅ Verification complete!")
    
    # Additional check: simulate preprocessing read
    print(f"\n{'='*70}")
    print("SIMULATING PREPROCESSING READ")
    print(f"{'='*70}\n")
    
    test_file = npy_files[0]
    print(f"Testing with: {Path(test_file).stem}\n")
    
    try:
        segments = np.load(test_file, allow_pickle=True)
        
        # This is what preprocessing does:
        print(f"segments.shape: {segments.shape}")
        print(f"segments.shape[1]: {segments.shape[1]}")
        
        # Convert to structured array (preprocessing format)
        dtype = [('onset', np.float32), ('section', 'U40')]
        formatted_segments = []
        
        for i in range(len(segments)):
            time_val = float(segments[i, 0])
            label_val = str(segments[i, 1])
            formatted_segments.append((time_val, label_val))
        
        formatted_annotations = np.array(formatted_segments, dtype=dtype)
        
        print(f"\n✅ Successfully converted to preprocessing format:")
        print(f"   dtype: {formatted_annotations.dtype}")
        print(f"   First segment: onset={formatted_annotations[0]['onset']}, section='{formatted_annotations[0]['section']}'")
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")

def main():
    """Main execution."""
    
    print(f"\n{'='*70}")
    print("BEATLES ANNOTATION CONVERSION: TEXT → NPY")
    print(f"{'='*70}")
    print(f"Input:  {INPUT_ANNOTATION_DIR}/")
    print(f"Output: {OUTPUT_ANNOTATION_DIR}/")
    print(f"\nFormat: 2D array (n_segments, 2)")
    print(f"  - Column 0: Time in seconds (float)")
    print(f"  - Column 1: Label (string)")
    
    # Convert all annotations
    convert_all_annotations(INPUT_ANNOTATION_DIR, OUTPUT_ANNOTATION_DIR)
    
    # Verify conversion
    verify_npy_files(OUTPUT_ANNOTATION_DIR, sample_count=5)
    
    print(f"\n{'='*70}")
    print("✨ CONVERSION COMPLETE!")
    print(f"{'='*70}\n")
    print(f"Next steps:")
    print(f"1. ✅ .npy files are in 2D array format (n_segments, 2)")
    print(f"2. ✅ Times are in SECONDS (not frames)")
    print(f"3. ➡️  Run preprocessing - it will convert to frames internally")

if __name__ == '__main__':
    INPUT_ANNOTATION_DIR = "/Scratch/repository/iahmad/salami_7class_annotations"
    OUTPUT_ANNOTATION_DIR = "/Scratch/repository/iahmad/salami-mapping"
    
    main()