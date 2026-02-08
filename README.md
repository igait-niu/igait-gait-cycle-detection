# iGait Cycle Detection

Gait cycle detection from MediaPipe 3D pose estimation landmarks. Processes a single video's landmark data to identify heel strikes and extract full gait cycles using rhythmic template matching.

## Features

- Gait cycle detection from MediaPipe landmark JSON files
- Rhythmic template matching with composite leg signals
- Automatic turn detection and inactive period trimming
- Optional left/right label correction
- Optional analysis plots for visual verification
- JSON output with gait cycles and landmark data passthrough

## Requirements

- Python 3.9+
- NumPy
- Pandas
- SciPy
- Matplotlib

### Install

```bash
pip install -r requirements.txt
```

## Usage

### Basic

```bash
python gait_analysis_mediapipe.py --landmarks-json input/mediapipe/118/118_side_landmarks.json
```

Output is saved to the `output/` directory by default.

### CLI Arguments

| Argument | Description | Default |
|---|---|---|
| `--landmarks-json` | Path to MediaPipe landmarks JSON file (required) | -- |
| `--output-dir` | Output directory path | `./output` |
| `--subject-id` | Subject identifier for report titles | Auto-detected from filename |
| `--fps` | Video FPS for time calculations | `30.0` |
| `--plots` | Generate analysis plots (gait detection + keypoint comparisons) | Disabled |
| `--perform-lr-correction` | Enable left/right label correction | Disabled |
| `--no-trim` | Skip inactive period trimming | Trimming enabled |
| `--confidence-threshold` | Landmark confidence threshold | `0.3` |
| `--velocity-threshold` | Walking velocity threshold | `0.0005` |

### Examples

Process with a specific subject ID and output directory:

```bash
python gait_analysis_mediapipe.py --landmarks-json input/mediapipe/124/124_side_video_landmarks.json --output-dir output/124 --subject-id 124
```

Generate analysis plots alongside JSON output:

```bash
python gait_analysis_mediapipe.py --landmarks-json input/mediapipe/118/118_side_landmarks.json --plots
```

Enable left/right correction and custom velocity threshold:

```bash
python gait_analysis_mediapipe.py --landmarks-json input/mediapipe/205/205_side_landmarks.json --perform-lr-correction --velocity-threshold 0.001
```

## Input

The script expects a MediaPipe landmarks JSON file produced by the [iGait 3D Pose Estimation](https://github.com/igait-niu/igait-mediapipe) pipeline. The JSON structure is:

```
<subject_id>_<view>_landmarks.json
```

Each file contains per-frame 3D pose landmarks (33 body keypoints) in MediaPipe world coordinates.

## Output

```
<output_dir>/
    <subject_id>_gait_analysis.json      # Gait cycles + landmark data
    gait_detection_<subject_id>.png       # Composite signal plot (with --plots)
    kp_<subject_id>_*.png                 # Individual keypoint plots (with --plots)
```

### Output JSON Format

```json
{
  "gait_cycles": [
    { "start": 62, "end": 95, "side": "R" },
    { "start": 79, "end": 108, "side": "L" }
  ],
  "landmark_data": [
    { "frame_number": 35, "pose_landmarks": [[x, y, z], ...] }
  ]
}
```

- `gait_cycles` -- full stride cycles (same-foot heel strike to next same-foot heel strike), filtered to exclude cycles that span turns or inactive periods
- `landmark_data` -- raw MediaPipe per-frame landmark array passed through from the input file
