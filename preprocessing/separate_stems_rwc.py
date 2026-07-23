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
SALAMI_IDS = ["RM-P002",
        "RM-P003",
        "RM-P006",
        "RM-P007",
        "RM-P008",
        "RM-P009",
        "RM-P010",
        "RM-P011",
        "RM-P013",
        "RM-P016",
        "RM-P017",
        "RM-P019",
        "RM-P020",
        "RM-P021",
        "RM-P022",
        "RM-P023",
        "RM-P024",
        "RM-P025",
        "RM-P027",
        "RM-P031",
        "RM-P033",
        "RM-P034",
        "RM-P035",
        "RM-P037",
        "RM-P038",
        "RM-P039",
        "RM-P040",
        "RM-P041",
        "RM-P042",
        "RM-P043",
        "RM-P044",
        "RM-P045",
        "RM-P046",
        "RM-P047",
        "RM-P048",
        "RM-P049",
        "RM-P050",
        "RM-P051",
        "RM-P052",
        "RM-P053",
        "RM-P056",
        "RM-P057",
        "RM-P059",
        "RM-P060",
        "RM-P061",
        "RM-P062",
        "RM-P063",
        "RM-P064",
        "RM-P066",
        "RM-P067",
        "RM-P068",
        "RM-P069",
        "RM-P071",
        "RM-P073",
        "RM-P074",
        "RM-P075",
        "RM-P077",
        "RM-P079",
        "RM-P080",
        "RM-P081",
        "RM-P083",
        "RM-P084",
        "RM-P086",
        "RM-P088",
        "RM-P091",
        "RM-P092",
        "RM-P093",
        "RM-P097",
        "RM-P099",
        "RM-P100",
        "RM-P001",
        "RM-P004",
        "RM-P005",
        "RM-P012",
        "RM-P014",
        "RM-P015",
        "RM-P018",
        "RM-P026",
        "RM-P028",
        "RM-P029",
        "RM-P030",
        "RM-P032",
        "RM-P036",
        "RM-P054",
        "RM-P055",
        "RM-P058",
        "RM-P065",
        "RM-P070",
        "RM-P072",
        "RM-P076",
        "RM-P078",
        "RM-P082",
        "RM-P085",
        "RM-P087",
        "RM-P089",
        "RM-P090",
        "RM-P094",
        "RM-P095",
        "RM-P096",
        "RM-P098"]

# Paths
INPUT_AUDIO_DIR = "/Scratch/repository/iahmad/RWC-audio"
OUTPUT_BASE_DIR = "/Scratch/repository/iahmad/RWC-demucs"

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
            stem_path = os.path.join(stem_dir, f"{song_id}.mp3")
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
                output_path = os.path.join(STEM_DIRS[stem_name], f"{song_id}.mp3")
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
        count = len(glob.glob(os.path.join(stem_dir, "*.mp3")))
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