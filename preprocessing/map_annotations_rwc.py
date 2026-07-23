import os
import numpy as np
from tqdm import tqdm

# === ICASSP 2022 Mapping Function ===
# def map_label(raw_label):
#     label = raw_label.lower()
#     if label == "end":
#         return "end"  # optional: can skip these in actual processing
#     substrings = [
#         ("silence", "silence"), ("pre-chorus", "verse"),
#         ("prechorus", "verse"), ("refrain", "chorus"),
#         ("chorus", "chorus"), ("theme", "chorus"),
#         ("stutter", "chorus"), ("verse", "verse"),
#         ("rap", "verse"), ("section", "verse"),
#         ("slow", "verse"), ("build", "verse"),
#         ("dialog", "verse"), ("intro", "intro"),
#         ("fadein", "intro"), ("opening", "intro"),
#         ("bridge", "bridge"), ("trans", "bridge"),
#         ("out", "outro"), ("coda", "outro"),
#         ("ending", "outro"), ("break", "inst"),
#         ("inst", "inst"), ("interlude", "inst"),
#         ("impro", "inst"), ("solo", "inst")
#     ]
#     for substr, mapped in substrings:
#         if substr in label:
#             return mapped
#     return "inst"  # fallback

def map_label(raw_label):
    label = raw_label.lower().strip()
    
    # Exact matches first
    if label in ["end", "End"]:
        return "end"
    
    # Handle "nothing" - RWC specific
    if label == "nothing":
        return "silence"
    
    substrings = [
        # Silence
        ("silence", "silence"),
        # Intro
        ("intro", "intro"),
        ("fadein", "intro"),
        ("opening", "intro"),
        # Pre/Post chorus BEFORE chorus
        ("pre-chorus", "verse"),    # ← BEFORE chorus
        ("prechorus", "verse"),     # ← BEFORE chorus
        ("post-chorus", "outro"),   # ← BEFORE chorus
        ("postchorus", "outro"),    # ← BEFORE chorus
        # Chorus
        ("refrain", "chorus"),
        ("chorus", "chorus"),
        ("theme", "chorus"),
        ("stutter", "chorus"),
        # Verse
        ("verse", "verse"),
        ("rap", "verse"),
        ("section", "verse"),
        ("slow", "verse"),
        ("build", "verse"),
        ("dialog", "verse"),
        # Bridge
        ("bridge", "bridge"),
        ("trans", "bridge"),
        # Outro - ending BEFORE out
        ("ending", "outro"),        # ← BEFORE out
        ("out", "outro"),
        ("coda", "outro"),
        # Instrumental
        ("break", "inst"),
        ("inst", "inst"),
        ("interlude", "inst"),
        ("impro", "inst"),
        ("solo", "inst"),
    ]
    
    for substr, mapped in substrings:
        if substr in label:
            return mapped
    
    return "inst"  # fallback
# === Label Conversion for SALAMI ===
def convert_salami_labels(base_dir, output_dir):
    # os.makedirs(output_dir, exist_ok=True)
    # subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print("base dir..........")
    print(base_dir)
    files = [f for f in os.listdir(base_dir) if f.endswith(".txt")]

    processed = 0
    for fname in tqdm(files, desc="🔄 Converting to 10-class labels"):
        input_path = os.path.join(base_dir, fname)
        output_path = os.path.join(output_dir, fname.replace(".txt", ".npy"))

        segments = []
        with open(input_path, 'r') as f:
            for line in f:
                try:
                    time_str, raw_label = line.strip().split(maxsplit=1)
                    time = float(time_str)
                    mapped = map_label(raw_label)
                    if mapped == "end":
                        continue
                    segments.append((time, mapped))
                except Exception as e:
                    print(f"⚠️ Error in {fname}: {e} | Line: {line}")
                    

        # output_path = os.path.join(output_dir, f"{fname}.npy")
        output_path = os.path.join(output_dir, fname.replace(".txt", ".npy"))
        np.save(output_path, np.array(segments, dtype=object))
        processed += 1

    print(f"\n✅ Completed label conversion for {processed} files.")

# Example usage:
base_salami_annotation_path = "/Scratch/repository/iahmad/rwc-annotations"
output_mapped_dir = "/Scratch/repository/iahmad/rwc-mapped-annotation"
convert_salami_labels(base_salami_annotation_path, output_mapped_dir)
