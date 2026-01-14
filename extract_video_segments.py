#!/usr/bin/env python3
"""
Extract video segments for each utterance from IEMOCAP subset.
Uses ffmpeg to cut videos based on start_time and end_time from metadata.
"""

import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def extract_video_segment(utterance: dict, base_path: Path) -> tuple:
    """
    Extract a video segment for a single utterance.
    Returns (utterance_id, success, error_message)
    """
    utterance_id = utterance['utterance_id']
    emotion = utterance['emotion']
    video_path = utterance.get('video_path')
    start_time = utterance['start_time']
    end_time = utterance['end_time']
    
    if not video_path:
        return (utterance_id, False, "No video path")
    
    # Normalize video path - extract filename and look in videos/ folder
    video_filename = Path(video_path).name
    video_file = base_path / "videos" / video_filename
    
    if not video_file.exists():
        # Try the original path as fallback
        video_file = base_path / video_path
        if not video_file.exists():
            return (utterance_id, False, f"Video not found: {video_filename}")
    
    # Create output directory
    output_dir = base_path / emotion / "video"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file
    output_file = output_dir / f"{utterance_id}.mp4"
    
    if output_file.exists():
        return (utterance_id, True, "Already exists")
    
    # Calculate duration
    duration = end_time - start_time
    
    # ffmpeg command
    cmd = [
        'ffmpeg',
        '-y',                      # Overwrite
        '-ss', str(start_time),    # Start time (before -i for faster seeking)
        '-i', str(video_file),     # Input video
        '-t', str(duration),       # Duration
        '-c:v', 'libx264',         # Video codec
        '-c:a', 'aac',             # Audio codec
        '-preset', 'slow',         # Better compression
        '-crf', '18',              # High quality (lower = better)
        '-loglevel', 'error',      # Only show errors
        str(output_file)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return (utterance_id, False, result.stderr[:200])
        return (utterance_id, True, None)
    except subprocess.TimeoutExpired:
        return (utterance_id, False, "Timeout")
    except Exception as e:
        return (utterance_id, False, str(e))


def main():
    # Parse arguments
    base_path = Path('/home/z23/Code/PM/iemocap_subset')
    metadata_file = base_path / 'metadata.json'
    
    if not metadata_file.exists():
        print(f"Error: {metadata_file} not found")
        return 1
    
    # Load metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    utterances = metadata['utterances']
    total = len(utterances)
    
    print(f"Extracting video segments for {total} utterances...")
    print(f"Output format: MP4 (H.264 + AAC)")
    print()
    
    # Process with thread pool (I/O bound, ffmpeg runs separately)
    success_count = 0
    error_count = 0
    errors = []
    
    # Use 4 workers to avoid overwhelming the system
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(extract_video_segment, u, base_path): u 
            for u in utterances
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            utterance_id, success, error = future.result()
            
            if success:
                success_count += 1
            else:
                error_count += 1
                errors.append((utterance_id, error))
            
            # Progress
            if i % 100 == 0 or i == total:
                print(f"  Progress: {i}/{total} ({success_count} OK, {error_count} errors)")
    
    print()
    print(f"Extraction complete!")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    
    if errors and len(errors) <= 10:
        print("\nErrors:")
        for uid, err in errors:
            print(f"  {uid}: {err}")
    elif errors:
        print(f"\nFirst 10 errors:")
        for uid, err in errors[:10]:
            print(f"  {uid}: {err}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
