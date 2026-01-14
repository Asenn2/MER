#!/usr/bin/env python3
"""
IEMOCAP Subset Extractor for Transfer Learning
Extracts audio, video, and text for 6 emotions with balanced classes.
"""

import argparse
import csv
import json
import os
import re
import shutil
import tarfile
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Utterance:
    """Represents a single utterance with all modalities."""
    utterance_id: str
    session: str
    speaker: str
    emotion: str
    transcription: str
    audio_path: str
    video_path: Optional[str]
    text_path: str
    start_time: float = 0.0
    end_time: float = 0.0
    audio_size: int = 0
    video_size: int = 0


# Emotion code mapping
EMOTION_MAP = {
    'ang': 'angry',
    'hap': 'happy', 
    'exc': 'excited',
    'sad': 'sad',
    'neu': 'neutral',
    'fru': 'frustrated',
    'fea': 'fearful',
    'sur': 'surprised',
    'dis': 'disgusted',
    'oth': 'other',
    'xxx': 'other'
}

# Target emotions (base 5 + 1 to be determined)
BASE_EMOTIONS = {'angry', 'happy', 'sad', 'neutral', 'frustrated'}
CANDIDATE_EMOTIONS = {'excited', 'fearful'}

TARGET_SIZE_GB = 10.0
TARGET_SIZE_BYTES = int(TARGET_SIZE_GB * 1024 * 1024 * 1024)


def parse_emotion_label(label: str) -> Optional[str]:
    """Parse emotion label from annotation file."""
    label = label.strip().lower()
    # Handle compound emotions like "hap; exc"
    if ';' in label:
        label = label.split(';')[0].strip()
    return EMOTION_MAP.get(label[:3])


def parse_evaluation_file(content: str) -> Dict[str, Tuple[str, float, float]]:
    """
    Parse IEMOCAP evaluation file to extract emotion labels and timestamps.
    Returns dict: utterance_id -> (emotion, start_time, end_time)
    """
    results = {}
    
    # Pattern to match utterance lines with timestamps
    # Format: [start - end] utterance_id emotion [V, A, D]
    pattern = r'\[([\d\.]+) - ([\d\.]+)\]\s+(\S+)\s+(\w+)\s+\[[\d\., ]+\]'
    
    lines = content.split('\n')
    for line in lines:
        match = re.match(pattern, line)
        if match:
            start_time = float(match.group(1))
            end_time = float(match.group(2))
            utterance_id = match.group(3)
            emotion_code = match.group(4).lower()
            emotion = parse_emotion_label(emotion_code)
            
            if emotion:
                results[utterance_id] = (emotion, start_time, end_time)
    
    return results


def parse_transcription_file(content: str) -> Dict[str, str]:
    """
    Parse IEMOCAP transcription file.
    Format: Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me.
    Returns dict: utterance_id -> transcription
    """
    results = {}
    
    # Pattern: utterance_id [timestamps]: transcription
    pattern = r'^(\S+)\s+\[[\d\.-]+\]:\s*(.*)$'
    
    for line in content.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            utterance_id = match.group(1)
            transcription = match.group(2).strip()
            results[utterance_id] = transcription
    
    return results


def scan_archive(tar_path: str) -> Tuple[Dict[str, List[Utterance]], Dict[str, int]]:
    """
    Scan the tar.gz archive to collect all utterances and their metadata.
    Returns: (utterances_by_emotion, emotion_counts)
    """
    print(f"Scanning archive: {tar_path}")
    print("This may take a few minutes...")
    
    utterances_by_emotion: Dict[str, List[Utterance]] = defaultdict(list)
    
    # Track files by utterance_id
    audio_files: Dict[str, Tuple[str, int]] = {}  # id -> (path, size)
    video_files: Dict[str, Tuple[str, int]] = {}
    transcriptions: Dict[str, str] = {}  # id -> transcription
    annotations: Dict[str, Tuple[str, float, float]] = {}  # id -> (emotion, start_time, end_time)
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        members = tar.getmembers()
        total = len(members)
        
        for idx, member in enumerate(members):
            if idx % 1000 == 0:
                print(f"  Scanning: {idx}/{total} files...")
            
            name = member.name
            
            # Audio files: .../sentences/wav/.../*.wav
            if '/sentences/wav/' in name and name.endswith('.wav'):
                utterance_id = Path(name).stem
                audio_files[utterance_id] = (name, member.size)
            
            # Video files: SessionX/dialog/avi/DivX/*.avi
            elif '/dialog/avi/DivX/' in name and name.endswith('.avi'):
                # Video is per-dialog, we'll handle this separately
                video_files[Path(name).stem] = (name, member.size)
            
            # Evaluation files: .../dialog/EmoEvaluation/*.txt
            elif '/EmoEvaluation/' in name and name.endswith('.txt') and 'Categorical' not in name:
                try:
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode('utf-8', errors='ignore')
                        parsed = parse_evaluation_file(content)
                        annotations.update(parsed)
                except:
                    pass
            
            # Transcription files: .../dialog/transcriptions/*.txt
            elif '/transcriptions/' in name and name.endswith('.txt'):
                try:
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode('utf-8', errors='ignore')
                        parsed = parse_transcription_file(content)
                        transcriptions.update(parsed)
                except:
                    pass
    
    print(f"  Found {len(audio_files)} audio files, {len(video_files)} video files, {len(annotations)} annotations, {len(transcriptions)} transcriptions")
    
    # Build utterance objects
    for utterance_id, (audio_path, audio_size) in audio_files.items():
        if utterance_id not in annotations:
            continue
        
        emotion, start_time, end_time = annotations[utterance_id]
        transcription = transcriptions.get(utterance_id, "")
        
        # Extract session and speaker from utterance_id
        # Format: Ses01F_impro01_F000
        parts = utterance_id.split('_')
        session = parts[0][:5] if parts else "unknown"
        speaker = parts[-1][0] if parts else "unknown"
        
        # Find corresponding video (dialog level)
        dialog_id = '_'.join(parts[:-1]) if len(parts) > 1 else utterance_id
        video_info = video_files.get(dialog_id)
        
        utterance = Utterance(
            utterance_id=utterance_id,
            session=session,
            speaker=speaker,
            emotion=emotion,
            transcription=transcription,
            audio_path=audio_path,
            video_path=video_info[0] if video_info else None,
            text_path=f"{dialog_id}.txt",
            start_time=start_time,
            end_time=end_time,
            audio_size=audio_size,
            video_size=video_info[1] if video_info else 0
        )
        
        utterances_by_emotion[emotion].append(utterance)
    
    # Count and summarize
    emotion_counts = {e: len(u) for e, u in utterances_by_emotion.items()}
    
    print("\nEmotion distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        total_audio = sum(u.audio_size for u in utterances_by_emotion[emotion])
        total_video = sum(u.video_size for u in utterances_by_emotion[emotion])
        print(f"  {emotion}: {count} utterances ({total_audio / 1024 / 1024:.1f} MB audio, {total_video / 1024 / 1024:.1f} MB video)")
    
    return utterances_by_emotion, emotion_counts


def select_sixth_emotion(emotion_counts: Dict[str, int]) -> str:
    """
    Select between 'excited' and 'fearful' based on class balance.
    Returns the emotion that provides better balance with base emotions.
    """
    base_counts = {e: emotion_counts.get(e, 0) for e in BASE_EMOTIONS}
    min_base = min(base_counts.values())
    
    excited_count = emotion_counts.get('excited', 0)
    fearful_count = emotion_counts.get('fearful', 0)
    
    print(f"\nSelecting 6th emotion:")
    print(f"  Base minimum count: {min_base}")
    print(f"  excited: {excited_count}, fearful: {fearful_count}")
    
    # Choose the one closest to min_base (for better balance)
    # If both are above min_base, choose the one with more samples
    if fearful_count >= min_base * 0.8 and fearful_count <= excited_count:
        selected = 'fearful'
    elif excited_count >= min_base * 0.5:
        selected = 'excited'
    else:
        selected = 'fearful' if fearful_count > excited_count else 'excited'
    
    print(f"  Selected: {selected}")
    return selected


def balance_classes(
    utterances_by_emotion: Dict[str, List[Utterance]],
    target_emotions: Set[str],
    target_size_bytes: int
) -> Dict[str, List[Utterance]]:
    """
    Balance classes to have equal samples and fit within target size.
    """
    # Filter to target emotions only
    filtered = {e: u for e, u in utterances_by_emotion.items() if e in target_emotions}
    
    if not filtered:
        return {}
    
    # Find minimum class size
    min_count = min(len(u) for u in filtered.values())
    print(f"\nBalancing classes to {min_count} samples each")
    
    # Calculate average size per utterance (audio only, videos extracted separately)
    total_audio = sum(sum(u.audio_size for u in utts) for utts in filtered.values())
    total_count = sum(len(u) for u in filtered.values())
    avg_size = total_audio / total_count if total_count > 0 else 0
    
    # Estimate samples per class to fit target size
    estimated_total = min_count * len(target_emotions)
    estimated_size = estimated_total * avg_size
    
    if estimated_size > target_size_bytes:
        # Need to reduce samples per class
        samples_per_class = int(target_size_bytes / (avg_size * len(target_emotions)))
        samples_per_class = min(samples_per_class, min_count)
    else:
        samples_per_class = min_count
    
    print(f"  Samples per class: {samples_per_class}")
    print(f"  Estimated total size: {samples_per_class * len(target_emotions) * avg_size / 1024 / 1024 / 1024:.2f} GB")
    
    # Select samples (shuffle for variety)
    import random
    random.seed(42)  # Reproducibility
    
    balanced = {}
    for emotion, utts in filtered.items():
        shuffled = utts.copy()
        random.shuffle(shuffled)
        balanced[emotion] = shuffled[:samples_per_class]
    
    return balanced


def extract_subset(
    tar_path: str,
    output_dir: str,
    balanced_utterances: Dict[str, List[Utterance]],
    dry_run: bool = False
) -> None:
    """
    Extract the selected utterances to the output directory.
    """
    output_path = Path(output_dir)
    
    if dry_run:
        print("\n[DRY RUN] Would extract:")
        for emotion, utts in balanced_utterances.items():
            print(f"  {emotion}: {len(utts)} files")
        return
    
    # Create directory structure
    for emotion in balanced_utterances.keys():
        (output_path / emotion / "audio").mkdir(parents=True, exist_ok=True)
        (output_path / emotion / "text").mkdir(parents=True, exist_ok=True)
    
    # Single video folder (videos are per-dialog, not per-utterance)
    (output_path / "videos").mkdir(parents=True, exist_ok=True)
    
    # Collect all files to extract
    files_to_extract: Dict[str, Tuple[str, str]] = {}  # archive_path -> (output_path, type)
    video_dialogs: Set[str] = set()  # Track unique video dialogs to extract
    
    all_utterances = []
    for emotion, utts in balanced_utterances.items():
        for u in utts:
            all_utterances.append(u)
            
            # Audio
            audio_out = output_path / emotion / "audio" / f"{u.utterance_id}.wav"
            files_to_extract[u.audio_path] = (str(audio_out), 'audio')
            
            # Update utterance with new paths
            u.audio_path = str(audio_out.relative_to(output_path))
            
            # Video (collect unique dialogs)
            if u.video_path and u.video_path not in video_dialogs:
                video_dialogs.add(u.video_path)
                video_out = output_path / "videos" / Path(u.video_path).name
                files_to_extract[u.video_path] = (str(video_out), 'video')
                u.video_path = str(video_out.relative_to(output_path))
    
    print(f"\nExtracting {len(files_to_extract)} files...")
    
    # Extract files
    extracted_count = 0
    with tarfile.open(tar_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.name in files_to_extract:
                out_path, file_type = files_to_extract[member.name]
                
                # Extract to temp and move
                f = tar.extractfile(member)
                if f:
                    with open(out_path, 'wb') as out:
                        shutil.copyfileobj(f, out)
                    extracted_count += 1
                    
                    if extracted_count % 100 == 0:
                        print(f"  Extracted {extracted_count}/{len(files_to_extract)} files...")
    
    # Create text files with transcriptions
    for u in all_utterances:
        text_out = output_path / u.emotion / "text" / f"{u.utterance_id}.txt"
        text_out.write_text(u.transcription, encoding='utf-8')
        u.text_path = str(text_out.relative_to(output_path))
    
    print(f"  Created {len(all_utterances)} text files")
    
    # Generate metadata
    generate_metadata(output_path, all_utterances)
    
    # Calculate final size
    total_size = sum(
        f.stat().st_size 
        for f in output_path.rglob('*') 
        if f.is_file()
    )
    print(f"\nExtraction complete!")
    print(f"  Total files: {extracted_count + len(all_utterances)}")
    print(f"  Total size: {total_size / 1024 / 1024 / 1024:.2f} GB")


def generate_metadata(output_path: Path, utterances: List[Utterance]) -> None:
    """Generate metadata.csv and metadata.json files."""
    
    # CSV
    csv_path = output_path / "metadata.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'utterance_id', 'session', 'speaker', 'emotion',
            'audio_path', 'video_path', 'text_path', 
            'start_time', 'end_time', 'transcription'
        ])
        writer.writeheader()
        for u in utterances:
            writer.writerow({
                'utterance_id': u.utterance_id,
                'session': u.session,
                'speaker': u.speaker,
                'emotion': u.emotion,
                'audio_path': u.audio_path,
                'video_path': u.video_path or '',
                'text_path': u.text_path,
                'start_time': u.start_time,
                'end_time': u.end_time,
                'transcription': u.transcription
            })
    
    # JSON
    json_path = output_path / "metadata.json"
    metadata = {
        'total_samples': len(utterances),
        'emotions': list(set(u.emotion for u in utterances)),
        'samples_per_emotion': defaultdict(int),
        'utterances': []
    }
    
    for u in utterances:
        metadata['samples_per_emotion'][u.emotion] += 1
        metadata['utterances'].append({
            'utterance_id': u.utterance_id,
            'session': u.session,
            'speaker': u.speaker,
            'emotion': u.emotion,
            'audio_path': u.audio_path,
            'video_path': u.video_path,
            'text_path': u.text_path,
            'start_time': u.start_time,
            'end_time': u.end_time,
            'transcription': u.transcription
        })
    
    metadata['samples_per_emotion'] = dict(metadata['samples_per_emotion'])
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"  Generated {csv_path.name} and {json_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract balanced IEMOCAP subset for transfer learning'
    )
    parser.add_argument(
        '--input', '-i',
        default='/run/media/z23/DATA/IEMOCAP_full_release.tar.gz',
        help='Path to IEMOCAP tar.gz file'
    )
    parser.add_argument(
        '--output', '-o',
        default='/home/z23/Code/PM/iemocap_subset',
        help='Output directory for subset'
    )
    parser.add_argument(
        '--target-size', '-s',
        type=float,
        default=5.0,
        help='Target size in GB (default: 5.0)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Scan and analyze without extracting'
    )
    parser.add_argument(
        '--force-emotion',
        choices=['excited', 'fearful'],
        help='Force selection of 6th emotion instead of auto-selecting'
    )
    
    args = parser.parse_args()
    
    # Verify input exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Scan archive
    utterances_by_emotion, emotion_counts = scan_archive(args.input)
    
    if not utterances_by_emotion:
        print("Error: No valid utterances found in archive")
        return 1
    
    # Select 6th emotion
    if args.force_emotion:
        sixth_emotion = args.force_emotion
        print(f"\nUsing forced 6th emotion: {sixth_emotion}")
    else:
        sixth_emotion = select_sixth_emotion(emotion_counts)
    
    target_emotions = BASE_EMOTIONS | {sixth_emotion}
    print(f"\nTarget emotions: {sorted(target_emotions)}")
    
    # Balance classes
    target_size = int(args.target_size * 1024 * 1024 * 1024)
    balanced = balance_classes(utterances_by_emotion, target_emotions, target_size)
    
    if not balanced:
        print("Error: No samples to extract after balancing")
        return 1
    
    # Extract
    extract_subset(args.input, args.output, balanced, dry_run=args.dry_run)
    
    return 0


if __name__ == '__main__':
    exit(main())
