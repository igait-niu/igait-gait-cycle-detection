"""
Batch processing script for gait_analysis_mediapipe.py

Supports parallel execution via multiprocessing for HPC clusters.
Two input layouts:
  flat  (cluster):  ./input/{id}/{id}_Side_landmarks.json + {id}_Side_pose.mp4
  split (legacy):   ./input/mediapipe/{id}/*side*_landmarks.json + ./input/raw/*.MOV
"""

import argparse
import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_SCRIPT = os.path.join(SCRIPT_DIR, 'gait_analysis_mediapipe.py')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch gait cycle detection with multiprocessing support.')
    parser.add_argument('--input-dir', default=os.path.join(SCRIPT_DIR, 'input'),
                        help='Root input directory (default: ./input)')
    parser.add_argument('--output-dir', default=os.path.join(SCRIPT_DIR, 'output'),
                        help='Root output directory (default: ./output)')
    parser.add_argument('--workers', type=int, default=cpu_count(),
                        help=f'Number of parallel workers (default: {cpu_count()}). '
                             'Each worker uses ~200MB memory.')
    parser.add_argument('--input-layout', choices=['flat', 'split'], default='flat',
                        help='Input directory structure: '
                             'flat = {input}/{id}/{id}_Side_landmarks.json (cluster), '
                             'split = {input}/mediapipe/{id}/ + {input}/raw/ (legacy)')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Video FPS, passed to analysis script (default: 30.0)')
    parser.add_argument('--screenshot-heel-strikes', action='store_true',
                        help='Extract video frames at heel strikes')
    parser.add_argument('--plots', action='store_true',
                        help='Generate analysis plots')
    parser.add_argument('--dry-run', action='store_true',
                        help='List discovered subjects without processing')
    return parser.parse_args()


def discover_subjects_flat(input_dir):
    """Discover subjects in flat layout: input/{id}/{id}_Side_landmarks.json"""
    subjects = []
    if not os.path.isdir(input_dir):
        return subjects

    for entry in sorted(os.listdir(input_dir)):
        subject_dir = os.path.join(input_dir, entry)
        if not os.path.isdir(subject_dir):
            continue

        subject_id = entry
        landmarks_path = None
        video_path = None

        for f in os.listdir(subject_dir):
            fl = f.lower()
            if fl.endswith('_landmarks.json') and 'side' in fl:
                landmarks_path = os.path.join(subject_dir, f)
            elif 'side' in fl and (fl.endswith('.mp4') or fl.endswith('.mov')):
                video_path = os.path.join(subject_dir, f)

        if landmarks_path is None:
            print(f'  WARNING: No side landmarks found for {subject_id}, skipping')
            continue

        subjects.append({
            'subject_id': subject_id,
            'landmarks_path': landmarks_path,
            'video_path': video_path,
        })

    return subjects


def find_video_split(landmark_filename, raw_dir):
    """Find matching video in split layout's raw/ directory."""
    base = landmark_filename.replace('_landmarks.json', '')

    candidate = os.path.join(raw_dir, f'{base}.MOV')
    if os.path.isfile(candidate):
        return candidate

    candidate = os.path.join(raw_dir, f'{base}_video.MOV')
    if os.path.isfile(candidate):
        return candidate

    return None


def discover_subjects_split(input_dir):
    """Discover subjects in split layout: input/mediapipe/{id}/ + input/raw/"""
    mediapipe_dir = os.path.join(input_dir, 'mediapipe')
    raw_dir = os.path.join(input_dir, 'raw')
    pattern = os.path.join(mediapipe_dir, '*', '*side*_landmarks.json')
    landmark_files = sorted(glob.glob(pattern))

    subjects = []
    for lpath in landmark_files:
        filename = os.path.basename(lpath)
        subject_id = filename.split('_')[0]
        video_path = find_video_split(filename, raw_dir)
        subjects.append({
            'subject_id': subject_id,
            'landmarks_path': lpath,
            'video_path': video_path,
        })
    return subjects


def discover_subjects(input_dir, layout):
    """Route to the appropriate discovery function."""
    if layout == 'flat':
        return discover_subjects_flat(input_dir)
    else:
        return discover_subjects_split(input_dir)


def process_one_subject(subject, output_dir, fps, plots, screenshots):
    """Process a single subject. Runs in a worker process."""
    subject_id = subject['subject_id']
    subject_output_dir = os.path.join(output_dir, subject_id)
    os.makedirs(subject_output_dir, exist_ok=True)

    cmd = [
        sys.executable, ANALYSIS_SCRIPT,
        '--landmarks-json', subject['landmarks_path'],
        '--output-dir', subject_output_dir,
        '--subject-id', subject_id,
        '--fps', str(fps),
    ]

    if plots:
        cmd.append('--plots')

    if screenshots and subject['video_path']:
        cmd += ['--screenshot-heel-strikes', '--video-file', subject['video_path']]

    start = time.monotonic()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.monotonic() - start

        # Write per-subject log
        log_path = os.path.join(subject_output_dir, 'batch_log.txt')
        with open(log_path, 'w') as f:
            f.write(f'Command: {" ".join(cmd)}\n')
            f.write(f'Return code: {result.returncode}\n')
            f.write(f'Duration: {duration:.1f}s\n\n')
            f.write('=== STDOUT ===\n')
            f.write(result.stdout)
            f.write('\n=== STDERR ===\n')
            f.write(result.stderr)

        return {
            'subject_id': subject_id,
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'duration': duration,
            'error': None,
        }
    except Exception as e:
        duration = time.monotonic() - start
        return {
            'subject_id': subject_id,
            'success': False,
            'returncode': -1,
            'duration': duration,
            'error': str(e),
        }


def print_summary(results, total_time):
    """Print batch processing summary."""
    results_sorted = sorted(results, key=lambda r: r['subject_id'])

    print(f'\n{"="*60}')
    print('BATCH SUMMARY')
    print(f'{"="*60}')
    for r in results_sorted:
        status = ' OK ' if r['success'] else 'FAIL'
        extra = f' ({r["error"]})' if r['error'] else ''
        print(f'  [{status}] {r["subject_id"]:>6s}  {r["duration"]:6.1f}s{extra}')

    succeeded = sum(1 for r in results if r['success'])
    failed = len(results) - succeeded
    serial_time = sum(r['duration'] for r in results)

    print(f'\n  {succeeded}/{len(results)} succeeded, {failed} failed')
    print(f'  Wall time:   {total_time:.1f}s')
    print(f'  Serial time: {serial_time:.1f}s')
    if total_time > 0:
        print(f'  Speedup:     {serial_time / total_time:.1f}x')


def main():
    args = parse_args()

    subjects = discover_subjects(args.input_dir, args.input_layout)
    if not subjects:
        print(f'No subjects found in {args.input_dir} (layout: {args.input_layout})')
        sys.exit(1)

    print(f'Found {len(subjects)} subject(s) (layout: {args.input_layout})')

    if args.dry_run:
        for s in subjects:
            video = 'video found' if s['video_path'] else 'NO VIDEO'
            print(f'  {s["subject_id"]}: {os.path.basename(s["landmarks_path"])} ({video})')
        sys.exit(0)

    n_workers = min(args.workers, len(subjects))
    print(f'Processing with {n_workers} worker(s)...\n')

    results = []
    start_time = time.monotonic()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_id = {
            executor.submit(
                process_one_subject,
                subject, args.output_dir, args.fps,
                args.plots, args.screenshot_heel_strikes
            ): subject['subject_id']
            for subject in subjects
        }

        for future in as_completed(future_to_id):
            subject_id = future_to_id[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    'subject_id': subject_id,
                    'success': False,
                    'returncode': -1,
                    'duration': 0,
                    'error': str(e),
                }
            results.append(result)
            status = ' OK ' if result['success'] else 'FAIL'
            print(f'  [{status}] {result["subject_id"]} ({result["duration"]:.1f}s)')

    total_time = time.monotonic() - start_time
    print_summary(results, total_time)

    failed = sum(1 for r in results if not r['success'])
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
