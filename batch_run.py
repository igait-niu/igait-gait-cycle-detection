"""
Batch testing script for gait_analysis_mediapipe.py
Processes all side-view landmark files and generates heel strike screenshots.
"""

import glob
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_SCRIPT = os.path.join(SCRIPT_DIR, 'gait_analysis_mediapipe.py')
MEDIAPIPE_DIR = os.path.join(SCRIPT_DIR, 'input', 'mediapipe')
RAW_VIDEO_DIR = os.path.join(SCRIPT_DIR, 'input', 'raw')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')


def find_video(landmark_filename):
    """Find the matching .MOV video file for a landmark JSON file."""
    base = landmark_filename.replace('_landmarks.json', '')

    # Try direct match first (e.g. 124_side_video -> 124_side_video.MOV)
    candidate = os.path.join(RAW_VIDEO_DIR, f'{base}.MOV')
    if os.path.isfile(candidate):
        return candidate

    # Try appending _video (e.g. 118_side -> 118_side_video.MOV)
    candidate = os.path.join(RAW_VIDEO_DIR, f'{base}_video.MOV')
    if os.path.isfile(candidate):
        return candidate

    return None

def main():
    # Find all side-view landmark JSON files
    pattern = os.path.join(MEDIAPIPE_DIR, '*', '*side*_landmarks.json')
    landmark_files = sorted(glob.glob(pattern))

    if not landmark_files:
        print('No side-view landmark files found.')
        sys.exit(1)

    print(f'Found {len(landmark_files)} side-view landmark files.\n')

    results = []

    for landmarks_path in landmark_files:
        filename = os.path.basename(landmarks_path)
        subject_id = filename.split('_')[0]
        subject_output_dir = os.path.join(OUTPUT_DIR, subject_id)

        print(f'{"="*60}')
        print(f'Subject {subject_id}: {filename}')
        print(f'{"="*60}')

        # Find matching video
        video_path = find_video(filename)

        # Build command
        cmd = [
            sys.executable, ANALYSIS_SCRIPT,
            '--landmarks-json', landmarks_path,
            '--output-dir', subject_output_dir,
            '--subject-id', subject_id,
        ]
        
        if video_path:
            cmd += ['--screenshot-heel-strikes', '--video-file', video_path]
            print(f'  Video: {os.path.basename(video_path)}')
        else:
            print(f'  WARNING: No matching video found — skipping screenshots')

        print(f'  Output: {subject_output_dir}')
        print()

        result = subprocess.run(cmd)
        success = result.returncode == 0
        results.append((subject_id, filename, success))

        print()

    # Summary
    print(f'\n{"="*60}')
    print('BATCH SUMMARY')
    print(f'{"="*60}')
    for subject_id, filename, success in results:
        status = 'OK' if success else 'FAILED'
        print(f'  [{status}] {subject_id} — {filename}')

    failed = sum(1 for _, _, s in results if not s)
    print(f'\n{len(results) - failed}/{len(results)} succeeded.')

    if failed:
        sys.exit(1)

if __name__ == '__main__':
    main()
