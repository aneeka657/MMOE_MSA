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
SALAMI_IDS = ["11_-_Do_You_Want_To_Know_A_Secret",
        "01_-_Birthday",
        "01_-_Taxman",
        "04_-_I_Need_You",
        "06_-_Youre_Going_To_Lose_That_Girl",
        "04_-_Getting_Better",
        "07_-_Michelle",
        "10_-_You_Really_Got_A_Hold_On_Me",
        "03_-_Youve_Got_To_Hide_Your_Love_Away",
        "13_-_Good_Night",
        "11_-_In_My_Life",
        "01_-_No_Reply",
        "08_-_Revolution_1",
        "09_-_Hold_Me_Tight",
        "17_-_Her_Majesty",
        "09_-_Its_Only_Love",
        "12_-_You_Cant_Do_That",
        "10_-_Lovely_Rita",
        "07_-_She_Said_She_Said",
        "05_-_Little_Child",
        "06_-_Till_There_Was_You",
        "15_-_Why_Dont_We_Do_It_In_The_Road",
        "12_-_Piggies",
        "03_-_Lucy_In_The_Sky_With_Diamonds",
        "05_-_Dig_It",
        "10_-_Things_We_Said_Today",
        "04_-_Blue_Jay_Way",
        "12_-_A_Taste_Of_Honey",
        "03_-_Babys_In_Black",
        "04_-_Oh!_Darling",
        "06_-_I_Want_You",
        "03_-_Across_the_Universe",
        "13_-_She_Came_In_Through_The_Bathroom_Window",
        "06_-_Helter_Skelter",
        "05_-_Another_Girl",
        "03_-_Glass_Onion",
        "06_-_Tell_Me_Why",
        "10_-_Im_So_Tired",
        "06_-_Yellow_Submarine",
        "16_-_The_End",
        "11_-_All_You_Need_Is_Love",
        "01_-_Sgt._Peppers_Lonely_Hearts_Club_Band",
        "06_-_Ask_Me_Why",
        "05_-_Your_Mother_Should_Know",
        "08_-_Ive_Got_A_Feeling",
        "01_-_I_Saw_Her_Standing_There",
        "08_-_Happiness_is_a_Warm_Gun",
        "06_-_Let_It_Be",
        "08_-_Eight_Days_a_Week",
        "06_-_The_Word",
        "10_-_Honey_Dont",
        "14_-_Money",
        "03_-_You_Wont_See_Me",
        "13_-_If_I_Needed_Someone",
        "08_-_Because",
        "07_-_Ticket_To_Ride",
        "14_-_Golden_Slumbers",
        "17_-_Julia",
        "08_-_Roll_Over_Beethoven",
        "14_-_Dont_Pass_Me_By",
        "12_-_Devil_In_Her_Heart",
        "01_-_It_Wont_Be_Long",
        "13_-_Got_To_Get_You_Into_My_Life",
        "13_-_Yesterday",
        "13_-_Theres_A_Place",
        "14_-_Twist_And_Shout",
        "12_-_Revolution_9",
        "09_-_Penny_Lane",
        "02_-_Misery",
        "10_-_Baby_Youre_A_Rich_Man",
        "13_-_Not_A_Second_Time",
        "03_-_All_My_Loving",
        "08_-_Within_You_Without_You",
        "02_-_Yer_Blues",
        "09_-_Girl",
        "07_-_While_My_Guitar_Gently_Weeps",
        "13_-_Ill_Be_Back",
        "04_-_Nowhere_Man",
        "09_-_And_Your_Bird_Can_Sing",
        "13_-_What_Youre_Doing",
        "05_-_Think_For_Yourself",
        "12_-_I_Want_To_Tell_You",
        "12_-_I_Dont_Want_to_Spoil_the_Party",
        "05_-_Octopuss_Garden",
        "11_-_Black_Bird",
        "07_-_Maggie_Mae",
        "01_-_Magical_Mystery_Tour",
        "02_-_Im_a_Loser",
        "11_-_Tell_Me_What_You_See",
        "07_-_Please_Mister_Postman",
        "03_-_Flying",
        "09_-_Words_of_Love",
        "11_-_Cry_Baby_Cry",
        "08_-_Good_Day_Sunshine",
        "14_-_Run_For_Your_Life",
        "08_-_What_Goes_On",
        "13_-_A_Day_In_The_Life",
        "06_-_I_Am_The_Walrus",
        "04_-_Everybodys_Got_Something_To_Hide_Except_Me_and_My_Monkey",
        "10_-_Im_Looking_Through_You",
        "14_-_Everybodys_Trying_to_Be_My_Baby",
        "02_-_I_Should_Have_Known_Better",
        "10_-_Savoy_Truffle",
        "02_-_Something",
        "02_-_Norwegian_Wood_(This_Bird_Has_Flown)",
        "12_-_Get_Back",
        "02_-_Dear_Prudence",
        "05_-_Sexy_Sadie",
        "08_-_Any_Time_At_All",
        "01_-_Drive_My_Car",
        "07_-_Kansas_City-_Hey,_Hey,_Hey,_Hey",
        "12_-_Wait",
        "01_-_A_Hard_Days_Night",
        "04_-_Dont_Bother_Me",
        "12_-_Sgt._Peppers_Lonely_Hearts_Club_Band_(Reprise)",
        "05_-_Boys",
        "08_-_Act_Naturally",
        "01_-_Back_in_the_USSR",
        "07_-_Hello_Goodbye",
        "05_-_Ill_Follow_the_Sun",
        "01_-_Help!",
        "10_-_You_Like_Me_Too_Much",
        "03_-_If_I_Fell",
        "07_-_Being_For_The_Benefit_Of_Mr._Kite!",
        "05_-_Fixing_A_Hole",
        "02_-_With_A_Little_Help_From_My_Friends",
        "04_-_Im_Happy_Just_To_Dance_With_You",
        "06_-_Mr._Moonlight",
        "11_-_Doctor_Robert",
        "10_-_The_Long_and_Winding_Road",
        "06_-_The_Continuing_Story_of_Bungalow_Bill",
        "09_-_Ill_Cry_Instead",
        "03_-_Im_Only_Sleeping",
        "11_-_Good_Morning_Good_Morning",
        "13_-_Rocky_Raccoon",
        "10_-_Baby_Its_You",
        "03_-_Anna_(Go_To_Him)",
        "14_-_Dizzy_Miss_Lizzy",
        "11_-_I_Wanna_Be_Your_Man",
        "07_-_Please_Please_Me",
        "03_-_Mother_Natures_Son",
        "11_-_For_You_Blue",
        "09_-_Honey_Pie",
        "07_-_Long_Long_Long",
        "04_-_Rock_and_Roll_Music",
        "12_-_Ive_Just_Seen_a_Face",
        "09_-_You_Never_Give_Me_Your_Money",
        "02_-_The_Night_Before",
        "03_-_Maxwells_Silver_Hammer",
        "08_-_Love_Me_Do",
        "12_-_Polythene_Pam",
        "04_-_Chains",
        "05_-_Wild_Honey_Pie",
        "02_-_The_Fool_On_The_Hill",
        "11_-_Every_Little_Thing",
        "16_-_I_Will",
        "04_-_Ob-La-Di,_Ob-La-Da",
        "14_-_Tomorrow_Never_Knows",
        "09_-_Martha_My_Dear",
        "07_-_Cant_Buy_Me_Love",
        "15_-_Carry_That_Weight",
        "07_-_Here_Comes_The_Sun",
        "01_-_Come_Together",
        "11_-_When_I_Get_Home",
        "09_-_P._S._I_Love_You",
        "08_-_Strawberry_Fields_Forever",
        "02_-_All_Ive_Got_To_Do",
        "10_-_For_No_One",
        "09_-_One_After_909",
        "04_-_Love_You_To",
        "02_-_Dig_a_Pony",
        "02_-_Eleanor_Rigby",
        "04_-_I_Me_Mine",
        "05_-_And_I_Love_Her",
        "06_-_Shes_Leaving_Home",
        "09_-_When_Im_Sixty-Four",
        "10_-_Sun_King",
        "11_-_Mean_Mr_Mustard",
        "01_-_Two_of_Us",
        "05_-_Here,_There_And_Everywhere"]

# Paths
INPUT_AUDIO_DIR = "/Scratch/repository/iahmad/beatles"
OUTPUT_BASE_DIR = "/Scratch/repository/iahmad/beatles-demucs"

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
