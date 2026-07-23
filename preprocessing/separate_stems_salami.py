#!/usr/bin/env python3
"""
Complete SALAMI Demixing Script
Checks existing stems and processes only missing ones
"""

import os
import glob
from pathlib import Path
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import time
from tqdm import tqdm

# Configuration
SALAMI_IDS = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 306, 307, 308, 309, 310, 311, 
              312, 314, 315, 316, 317, 318, 319, 320, 322, 323, 324, 325, 326, 327, 328, 330, 
              331, 332, 333, 334, 335, 336, 338, 339, 340, 341, 342, 343, 344, 346, 347, 348, 
              349, 350, 351, 352, 354, 355, 356, 357, 358, 359, 360, 362, 363, 364, 365, 366, 
              367, 368, 370, 371, 372, 373, 374, 376, 378, 379, 380, 381, 382, 383, 384, 386, 
              387, 388, 389, 390, 391, 392, 394, 395, 396, 397, 398, 399, 400, 532, 533, 534, 
              535, 536, 538, 539, 540, 541, 542, 543, 544, 546, 547, 548, 549, 550, 551, 552, 
              554, 555, 556, 557, 558, 559, 562, 563, 629, 630, 631, 632, 634, 635, 636, 637, 
              638, 639, 640, 642, 643, 644, 645, 646, 647, 648, 650, 651, 652, 653, 654, 655, 
              656, 658, 659, 660, 661, 662, 663, 664, 666, 667, 668, 669, 670, 671, 672, 674, 
              675, 676, 677, 678, 679, 680, 682, 683, 684, 685, 686, 687, 688, 690, 691, 692, 
              693, 694, 695, 696, 698, 699, 700, 701, 702, 703, 704, 706, 707, 708, 709, 710, 
              711, 712, 714, 715, 716, 717, 718, 719, 720, 722, 723, 724, 1570, 1571, 1572, 
              1573, 1574, 1575, 1576, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1586, 1587, 
              1588, 1589, 1590, 1591, 1592, 1594, 1595, 1596, 1597, 1598, 1600, 1602, 1603, 
              1604, 1605, 1607, 1608, 1610, 1611, 1613, 1615, 1616, 1618, 1619, 1620, 1622, 
              1624, 1626, 1627, 1628, 1629, 1630, 1631, 1634, 1635, 1640, 1642, 1647, 1648, 
              1652, 1653, 1654]

# Paths
INPUT_AUDIO_DIR = "/Scratch/repository/iahmad/salami_pop"
OUTPUT_BASE_DIR = "/Scratch/repository/iahmad/salami-demucs"

STEM_DIRS = {
    'drums': os.path.join(OUTPUT_BASE_DIR, 'drums'),
    'vocals': os.path.join(OUTPUT_BASE_DIR, 'vocals'),
    'bass': os.path.join(OUTPUT_BASE_DIR, 'bass'),
    'other': os.path.join(OUTPUT_BASE_DIR, 'others')
}

# Create output directories
for stem_dir in STEM_DIRS.values():
    os.makedirs(stem_dir, exist_ok=True)

def check_existing_stems(song_ids):
    """Check which stems already exist for each song."""
    
    print(f"\n{'='*80}")
    print("CHECKING EXISTING STEMS")
    print(f"{'='*80}\n")
    
    status = {}
    
    for song_id in song_ids:
        status[song_id] = {
            'audio_exists': False,
            'audio_path': None,
            'stems': {
                'drums': False,
                'vocals': False,
                'bass': False,
                'other': False
            },
            'missing_stems': []
        }
        
        # Check if audio file exists
        for ext in ['.mp3', '.wav', '.flac']:
            audio_path = os.path.join(INPUT_AUDIO_DIR, f"{song_id}{ext}")
            if os.path.exists(audio_path):
                status[song_id]['audio_exists'] = True
                status[song_id]['audio_path'] = audio_path
                break
        
        # Check stems
        for stem_name, stem_dir in STEM_DIRS.items():
            stem_path = os.path.join(stem_dir, f"{song_id}.wav")
            if os.path.exists(stem_path):
                status[song_id]['stems'][stem_name] = True
            else:
                status[song_id]['missing_stems'].append(stem_name)
    
    # Summary
    has_audio = sum(1 for s in status.values() if s['audio_exists'])
    no_audio = len(song_ids) - has_audio
    
    complete = sum(1 for s in status.values() if len(s['missing_stems']) == 0 and s['audio_exists'])
    partial = sum(1 for s in status.values() if 0 < len(s['missing_stems']) < 4 and s['audio_exists'])
    none = sum(1 for s in status.values() if len(s['missing_stems']) == 4 and s['audio_exists'])
    
    print(f"📊 Audio Files Status:")
    print(f"   ✅ Has audio: {has_audio}/{len(song_ids)}")
    print(f"   ❌ No audio: {no_audio}/{len(song_ids)}")
    
    print(f"\n📊 Stem Completion Status:")
    print(f"   ✅ Complete (all 4 stems): {complete}")
    print(f"   ⚠️  Partial (some stems): {partial}")
    print(f"   ❌ None (no stems): {none}")
    
    # Detailed breakdown by stem
    print(f"\n📊 Individual Stem Status:")
    for stem_name in ['drums', 'vocals', 'bass', 'other']:
        existing = sum(1 for s in status.values() if s['stems'][stem_name])
        print(f"   {stem_name:8s}: {existing}/{has_audio} exist")
    
    # Songs needing processing
    need_processing = [
        sid for sid, info in status.items() 
        if info['audio_exists'] and len(info['missing_stems']) > 0
    ]
    
    print(f"\n🎯 Songs needing processing: {len(need_processing)}")
    
    if no_audio > 0:
        no_audio_ids = [sid for sid, info in status.items() if not info['audio_exists']]
        print(f"\n⚠️  Songs without audio files: {no_audio_ids[:20]}")
        if len(no_audio_ids) > 20:
            print(f"   ... and {len(no_audio_ids) - 20} more")
    
    return status, need_processing

def process_missing_stems(status, need_processing):
    """Process only missing stems for songs that need it."""
    
    if not need_processing:
        print(f"\n✅ All stems already exist! Nothing to process.")
        return
    
    print(f"\n{'='*80}")
    print(f"PROCESSING {len(need_processing)} SONGS WITH MISSING STEMS")
    print(f"{'='*80}\n")
    
    # Check GPU
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        device = torch.device('cuda:2')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(2)}")
    else:
        device = torch.device('cpu')
        print(f"⚠️  Using CPU (will be slower)")
    
    # Load model
    print(f"\n📥 Loading htdemucs_ft model...")
    model = get_model('htdemucs_ft')
    model.eval()
    model = model.to(device)
    print(f"✅ Model loaded")
    print(f"🎯 Source order: {model.sources}")  # ['drums', 'bass', 'other', 'vocals']
    
    # Process each song
    processed = 0
    failed = 0
    
    for i, song_id in enumerate(tqdm(need_processing, desc="🎵 Processing songs"), 1):
        info = status[song_id]
        audio_path = info['audio_path']
        missing_stems = info['missing_stems']
        
        try:
            print(f"\n[{i}/{len(need_processing)}] Song {song_id}")
            print(f"   Audio: {Path(audio_path).name}")
            print(f"   Missing: {', '.join(missing_stems)}")
            
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert mono to stereo
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            
            # Resample if needed
            if sr != model.samplerate:
                resampler = torchaudio.transforms.Resample(sr, model.samplerate)
                waveform = resampler(waveform)
                sr = model.samplerate
            
            # Normalize
            peak = torch.max(torch.abs(waveform))
            if peak > 0:
                waveform = waveform / peak * 0.95
            
            waveform = waveform.to(device)
            
            # Separate
            with torch.no_grad():
                sources = apply_model(model, waveform.unsqueeze(0), split=True)
            
            sources = sources.squeeze(0).cpu()
            
            # Save only missing stems
            # Source order: drums=0, bass=1, other=2, vocals=3
            stem_indices = {'drums': 0, 'bass': 1, 'other': 2, 'vocals': 3}
            
            for stem_name in missing_stems:
                stem_idx = stem_indices[stem_name]
                stem_audio = sources[stem_idx]
                
                # Ensure stereo
                if stem_audio.ndim == 1:
                    stem_audio = stem_audio.unsqueeze(0)
                
                # Save
                output_path = os.path.join(STEM_DIRS[stem_name], f"{song_id}.wav")
                torchaudio.save(
                    output_path,
                    stem_audio,
                    sr,
                    encoding="PCM_S",
                    bits_per_sample=24
                )
            
            print(f"   ✅ Saved: {', '.join(missing_stems)}")
            processed += 1
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed += 1
            continue
    
    print(f"\n{'='*80}")
    print("🎉 PROCESSING COMPLETE!")
    print(f"{'='*80}\n")
    print(f"✅ Successfully processed: {processed}")
    print(f"❌ Failed: {failed}")
    print(f"\n📁 Output directories:")
    for stem_name, stem_dir in STEM_DIRS.items():
        count = len(glob.glob(os.path.join(stem_dir, "*.wav")))
        print(f"   {stem_name:8s}: {count} files in {stem_dir}")

def verify_completeness(song_ids):
    """Final verification that all stems exist."""
    
    print(f"\n{'='*80}")
    print("FINAL VERIFICATION")
    print(f"{'='*80}\n")
    
    complete_count = 0
    incomplete = []
    
    for song_id in song_ids:
        # Check if audio exists
        audio_exists = False
        for ext in ['.mp3', '.wav', '.flac']:
            if os.path.exists(os.path.join(INPUT_AUDIO_DIR, f"{song_id}{ext}")):
                audio_exists = True
                break
        
        if not audio_exists:
            continue
        
        # Check all stems
        all_stems_exist = True
        for stem_dir in STEM_DIRS.values():
            if not os.path.exists(os.path.join(stem_dir, f"{song_id}.wav")):
                all_stems_exist = False
                break
        
        if all_stems_exist:
            complete_count += 1
        else:
            incomplete.append(song_id)
    
    print(f"✅ Complete songs (all 4 stems): {complete_count}")
    
    if incomplete:
        print(f"⚠️  Incomplete songs: {len(incomplete)}")
        print(f"   IDs: {incomplete[:20]}")
        if len(incomplete) > 20:
            print(f"   ... and {len(incomplete) - 20} more")
    else:
        print(f"🎉 ALL SONGS HAVE COMPLETE STEMS!")

def main():
    """Main execution."""
    
    print(f"\n{'='*80}")
    print("SALAMI COMPLETE DEMIXING PIPELINE")
    print(f"{'='*80}\n")
    print(f"Total song IDs to process: {len(SALAMI_IDS)}")
    print(f"Input audio: {INPUT_AUDIO_DIR}")
    print(f"Output base: {OUTPUT_BASE_DIR}")
    
    # Step 1: Check existing stems
    status, need_processing = check_existing_stems(SALAMI_IDS)
    
    # Step 2: Process missing stems
    if need_processing:
        proceed = input(f"\n🎯 Process {len(need_processing)} songs? [y/N]: ")
        if proceed.lower() == 'y':
            process_missing_stems(status, need_processing)
        else:
            print("❌ Processing cancelled")
            return
    
    # Step 3: Final verification
    verify_completeness(SALAMI_IDS)
    
    print(f"\n{'='*80}")
    print("✨ PIPELINE COMPLETE!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()