# ==============================================================================
# Gait Analysis from MediaPipe 2D Pose Estimation
# Version: 5.0.0
#
# Converted from OpenPose-based gait_analysis_local_batch_v3.15.py
# - Input: MediaPipe landmark JSON files (pose-based, normalized to video resolution)
# - Coordinates: x,y in [0,1] normalized to frame width/height; y=0 is top, y=1 is bottom
# - Single video processing with CLI arguments
# - Output: JSON (gait cycles + landmark passthrough) and analysis plots
# ==============================================================================

import json
import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import cv2
except ImportError:
    cv2 = None

# ==============================================================================
# --- 1. MediaPipe Landmark Definitions & Keypoint Mapping ---
# ==============================================================================
MEDIAPIPE_POSE_LANDMARKS = {
    0: "Nose", 1: "Left_Eye_Inner", 2: "Left_Eye", 3: "Left_Eye_Outer",
    4: "Right_Eye_Inner", 5: "Right_Eye", 6: "Right_Eye_Outer",
    7: "Left_Ear", 8: "Right_Ear", 9: "Mouth_Left", 10: "Mouth_Right",
    11: "Left_Shoulder", 12: "Right_Shoulder", 13: "Left_Elbow", 14: "Right_Elbow",
    15: "Left_Wrist", 16: "Right_Wrist", 17: "Left_Pinky", 18: "Right_Pinky",
    19: "Left_Index", 20: "Right_Index", 21: "Left_Thumb", 22: "Right_Thumb",
    23: "Left_Hip", 24: "Right_Hip", 25: "Left_Knee", 26: "Right_Knee",
    27: "Left_Ankle", 28: "Right_Ankle", 29: "Left_Heel", 30: "Right_Heel",
    31: "Left_Foot_Index", 32: "Right_Foot_Index"
}

# Direct mapping: internal keypoint name -> MediaPipe landmark index
MP_TO_INTERNAL = {
    'Nose': 0,
    'RShoulder': 12, 'RElbow': 14, 'RWrist': 16,
    'LShoulder': 11, 'LElbow': 13, 'LWrist': 15,
    'RHip': 24, 'RKnee': 26, 'RAnkle': 28,
    'LHip': 23, 'LKnee': 25, 'LAnkle': 27,
    'LHeel': 29, 'RHeel': 30,
}

# Computed keypoints as midpoints of two MediaPipe landmarks
COMPUTED_KEYPOINTS = {
    'Neck': (11, 12),    # midpoint of LeftShoulder + RightShoulder
    'MidHip': (23, 24),  # midpoint of LeftHip + RightHip
}

MAJOR_KEYPOINTS_TO_TRACK = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
    "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "LHeel", "RHeel"
]

# ==============================================================================
# --- 2. Default Configuration ---
# ==============================================================================
CONFIDENCE_THRESHOLD = 0.3
# Walking velocity threshold for MidHip x-position displacement per frame.
# Derived from original pixel threshold (2.0px) normalized by typical frame width (~1920px).
WALKING_VELOCITY_THRESHOLD = 0.001
TURN_EXCLUSION_WINDOW_FRAMES = 20
# Acceleration threshold for detecting left/right label discontinuities.
# Derived from original pixel threshold (15.0px) normalized by typical frame width (~1920px).
DISCONTINUITY_ACCEL_THRESHOLD = 0.008
KEYPOINTS_TO_SWAP = ['Hip', 'Knee', 'Ankle', 'Heel']

# Rhythmic Template Matching parameters
MIN_STEPS_PER_FOOT = 2
MAX_STEPS_PER_FOOT = 7
MIN_STRIDE_TIME_FRAMES = 5
MAX_STRIDE_TIME_FRAMES = 70
GAIT_PEAK_PROMINENCE_LTR = 0.002
GAIT_PEAK_PROMINENCE_RTL = 0.002

# ==============================================================================
# --- 3. CLI Argument Parsing ---
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Gait Analysis from MediaPipe 2D Pose Estimation'
    )
    parser.add_argument(
        '--landmarks-json', type=str, required=True,
        help='Path to MediaPipe landmarks JSON file'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./output',
        help='Directory for output files (default: ./output)'
    )
    parser.add_argument(
        '--subject-id', type=str, default=None,
        help='Subject identifier for report titles (auto-detected from filename if omitted)'
    )
    parser.add_argument(
        '--fps', type=float, default=30.0,
        help='Video FPS for time calculations (default: 30.0)'
    )
    parser.add_argument(
        '--plots', action='store_true', default=False,
        help='Generate analysis plots (main gait detection + individual keypoint comparisons)'
    )
    parser.add_argument(
        '--perform-lr-correction', action='store_true', default=False,
        help='Enable L/R label correction via discontinuity detection'
    )
    parser.add_argument(
        '--no-trim', action='store_true', default=False,
        help='Skip inactive period trimming'
    )
    parser.add_argument(
        '--confidence-threshold', type=float, default=CONFIDENCE_THRESHOLD,
        help=f'Landmark confidence threshold (default: {CONFIDENCE_THRESHOLD})'
    )
    parser.add_argument(
        '--velocity-threshold', type=float, default=WALKING_VELOCITY_THRESHOLD,
        help=f'Walking velocity threshold (default: {WALKING_VELOCITY_THRESHOLD})'
    )
    parser.add_argument(
        '--screenshot-heel-strikes', action='store_true', default=False,
        help='Save video frames at detected heel strike moments as PNG images'
    )
    parser.add_argument(
        '--video-file', type=str, default=None,
        help='Path to source video file (required when --screenshot-heel-strikes is enabled)'
    )
    return parser.parse_args()

# ==============================================================================
# --- 4. Data Loading ---
# ==============================================================================
def load_mediapipe_json(json_path):
    """
    Load a MediaPipe landmarks JSON file and return:
      - DataFrame with columns matching the v3.15 analysis pipeline schema
      - Raw landmarks_data list for passthrough into output

    Expects pose-based landmarks normalized to video resolution:
      x in [0,1] (horizontal, left-to-right)
      y in [0,1] (vertical, 0=top, 1=bottom)
      Null landmarks are handled as NaN with confidence 0.0.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    total_frames = data.get('total_frames', 0)
    landmarks_list = data.get('landmarks_data', [])

    # Build frame_number -> pose_landmarks lookup
    frame_lookup = {}
    for entry in landmarks_list:
        fn = entry.get('frame_number')
        pl = entry.get('pose_landmarks')
        if fn is not None and pl is not None:
            frame_lookup[fn] = np.array(pl)  # shape (33, 3)

    # Determine frame range
    if total_frames > 0:
        frame_range = range(total_frames)
    elif frame_lookup:
        frame_range = range(max(frame_lookup.keys()) + 1)
    else:
        return pd.DataFrame(), landmarks_list

    kp_cols = [f'{name}_{ax}' for name in MAJOR_KEYPOINTS_TO_TRACK for ax in ['x', 'y', 'conf']]
    rows = []

    for frame_idx in frame_range:
        row = {'frame': frame_idx}
        pose = frame_lookup.get(frame_idx)

        if pose is None:
            for col in kp_cols:
                row[col] = np.nan
            row['x'] = np.nan
            row['y'] = np.nan
            row['confidence'] = 0.0
            rows.append(row)
            continue

        # Extract direct-mapped keypoints
        for internal_name, mp_idx in MP_TO_INTERNAL.items():
            landmark = pose[mp_idx]
            if landmark is None or (hasattr(landmark, '__iter__') and any(v is None for v in landmark)):
                row[f'{internal_name}_x'] = np.nan
                row[f'{internal_name}_y'] = np.nan
                row[f'{internal_name}_conf'] = 0.0
            else:
                row[f'{internal_name}_x'] = landmark[0]
                row[f'{internal_name}_y'] = landmark[1]
                row[f'{internal_name}_conf'] = 1.0

        # Compute synthetic keypoints (Neck, MidHip)
        for internal_name, (idx_a, idx_b) in COMPUTED_KEYPOINTS.items():
            a, b = pose[idx_a], pose[idx_b]
            if a is None or b is None:
                row[f'{internal_name}_x'] = np.nan
                row[f'{internal_name}_y'] = np.nan
                row[f'{internal_name}_conf'] = 0.0
            else:
                row[f'{internal_name}_x'] = (a[0] + b[0]) / 2.0
                row[f'{internal_name}_y'] = (a[1] + b[1]) / 2.0
                row[f'{internal_name}_conf'] = 1.0

        # Center point = average of Neck and MidHip
        neck_x, midhip_x = row.get('Neck_x'), row.get('MidHip_x')
        neck_y, midhip_y = row.get('Neck_y'), row.get('MidHip_y')
        if (neck_x is not None and midhip_x is not None
                and not (np.isnan(neck_x) or np.isnan(midhip_x))):
            row['x'] = (neck_x + midhip_x) / 2.0
            row['y'] = (neck_y + midhip_y) / 2.0
            row['confidence'] = 1.0
        else:
            row['x'] = np.nan
            row['y'] = np.nan
            row['confidence'] = 0.0

        rows.append(row)

    return pd.DataFrame(rows), landmarks_list

# ==============================================================================
# --- 5. Analysis Pipeline Functions (ported from Dr. Wang's code V3.15) ---
# ==============================================================================
def clean_tracking_data(df, confidence_threshold):
    if df.empty or df['x'].isnull().all():
        return pd.DataFrame()
    cleaned_df = df.copy()
    cols_to_process = ['x', 'y'] + [
        f'{kp}_{ax}' for kp in MAJOR_KEYPOINTS_TO_TRACK for ax in ['x', 'y', 'conf']
    ]
    for col in cols_to_process:
        if col in cleaned_df.columns:
            if '_conf' not in col:
                conf_col = col.replace('_x', '_conf').replace('_y', '_conf')
                if conf_col in cleaned_df.columns:
                    cleaned_df.loc[cleaned_df[conf_col] < confidence_threshold, col] = np.nan
            cleaned_df[col] = cleaned_df[col].interpolate(method='linear').ffill().bfill()
    window_length, polyorder = 15, 3
    coord_cols_to_smooth = [
        c for c in cols_to_process if '_conf' not in c and c in cleaned_df.columns
    ]
    if len(cleaned_df) > window_length:
        for col in coord_cols_to_smooth:
            cleaned_df[f'{col}_smoothed'] = savgol_filter(
                cleaned_df[col].fillna(0), window_length, polyorder
            )
    else:
        for col in coord_cols_to_smooth:
            cleaned_df[f'{col}_smoothed'] = cleaned_df[col]
    return cleaned_df


def correct_lr_with_discontinuity(df, threshold):
    print("\n--- Correcting L/R using Discontinuity-Driven Segmentation ---")
    corrected_df = df.copy()
    l_vel = corrected_df['LAnkle_x_smoothed'].diff().fillna(0)
    r_vel = corrected_df['RAnkle_x_smoothed'].diff().fillna(0)
    l_accel = l_vel.diff().fillna(0).abs()
    r_accel = r_vel.diff().fillna(0).abs()
    spike_frames = corrected_df[(l_accel > threshold) & (r_accel > threshold)].index.tolist()
    print(f"  Found {len(spike_frames)} frames with high simultaneous ankle acceleration.")
    if not spike_frames:
        print("  No significant discontinuities found. No swaps performed.")
        return corrected_df
    corrected_intervals = 0
    for i in range(len(spike_frames) - 1):
        start_frame = spike_frames[i]
        end_frame = spike_frames[i + 1]
        if not (1 < (end_frame - start_frame) < MAX_STRIDE_TIME_FRAMES):
            continue
        interval_df = corrected_df.loc[start_frame:end_frame]
        votes_for_swap, votes_against_swap = 0, 0
        valid_x = interval_df['MidHip_x_smoothed'].dropna()
        if len(valid_x) < 2:
            continue
        walks_left_to_right = valid_x.iloc[-1] > valid_x.iloc[0]
        for _, frame_data in interval_df.iterrows():
            r_knee_x = frame_data['RKnee_x_smoothed']
            l_shoulder_x = frame_data['LShoulder_x_smoothed']
            if pd.notna(r_knee_x) and pd.notna(l_shoulder_x):
                r_knee_is_ahead = (
                    r_knee_x > l_shoulder_x if walks_left_to_right
                    else r_knee_x < l_shoulder_x
                )
                if not r_knee_is_ahead:
                    votes_for_swap += 1
                else:
                    votes_against_swap += 1
        if votes_for_swap > votes_against_swap:
            corrected_intervals += 1
            segment_to_swap = corrected_df.loc[start_frame:end_frame].copy()
            for name in KEYPOINTS_TO_SWAP:
                for suffix in ['_x', '_y', '_conf', '_x_smoothed', '_y_smoothed']:
                    l_col, r_col = f'L{name}{suffix}', f'R{name}{suffix}'
                    if l_col in segment_to_swap.columns and r_col in segment_to_swap.columns:
                        temp_data = segment_to_swap[l_col].copy()
                        segment_to_swap[l_col] = segment_to_swap[r_col]
                        segment_to_swap[r_col] = temp_data
            corrected_df.loc[start_frame:end_frame] = segment_to_swap
    print(f"  Corrected {corrected_intervals} intermittent swap interval(s).")
    return corrected_df


def detect_walking_segments(df):
    print("--- Detecting Turns ---")
    if 'MidHip_x_smoothed' not in df.columns:
        print("  MidHip data not found. Treating as single segment.")
        return np.array([])
    raw_velocity = df['MidHip_x_smoothed'].diff().fillna(0)
    window = min(51, len(df) - 1 if len(df) % 2 == 0 else len(df))
    if window < 5:
        return np.array([])
    if window % 2 == 0:
        window -= 1
    smooth_directional_velocity = savgol_filter(raw_velocity, window, 3)
    turn_frames = np.where(np.diff(np.sign(smooth_directional_velocity)))[0]
    print(f"  Found {len(turn_frames)} turns at frames: {turn_frames}")
    return turn_frames


def get_analysis_exclusion_zones(df, turn_frames, velocity_threshold, turn_window):
    print("\n--- Identifying Analysis Zones (Trimming Inactivity & Turns) ---")
    speed = df['MidHip_x_smoothed'].diff().abs()
    window = min(51, len(df) - 1 if len(df) % 2 == 0 else len(df))
    if window < 5:
        return set()
    if window % 2 == 0:
        window -= 1
    smooth_speed = savgol_filter(speed.fillna(0), window, 3)
    active_frames = np.where(smooth_speed > velocity_threshold)[0]
    if len(active_frames) == 0:
        print("  Warning: No walking activity detected.")
        return set(range(len(df)))
    first_active, last_active = active_frames[0], active_frames[-1]
    print(f"  Active walking from frame {first_active} to {last_active}.")
    excluded_frames = set()
    excluded_frames.update(range(0, first_active))
    excluded_frames.update(range(last_active + 1, len(df)))
    for turn_frame in turn_frames:
        start_exclude = max(0, turn_frame - turn_window)
        end_exclude = turn_frame + turn_window
        excluded_frames.update(range(start_exclude, end_exclude + 1))
    print(f"  Total excluded frames: {len(excluded_frames)}")
    return excluded_frames


def create_composite_signal(df, side):
    """
    Create a normalized composite signal from hip, knee, ankle, and heel y-coordinates.

    In pose-based coordinates (y=0 top, y=1 bottom), peaks in this signal correspond
    to the foot being at its lowest point in the image (i.e., on the ground = heel strike).
    """
    hip_col = f'{side}Hip_y_smoothed'
    knee_col = f'{side}Knee_y_smoothed'
    ankle_col = f'{side}Ankle_y_smoothed'
    heel_col = f'{side}Heel_y_smoothed'
    required_cols = [hip_col, knee_col, ankle_col, heel_col]
    if not all(col in df.columns for col in required_cols):
        print(f"  Warning: Missing leg columns for {side} side.")
        return pd.Series(dtype='float64')
    leg_df = df[required_cols].copy()
    for col in leg_df.columns:
        min_val, max_val = leg_df[col].min(), leg_df[col].max()
        leg_df[col] = (
            (leg_df[col] - min_val) / (max_val - min_val)
            if (max_val - min_val) > 0 else 0.5
        )
    return leg_df.mean(axis=1)


def find_best_gait_pattern_iterative(segment_df, l_comp_sig, r_comp_sig, direction, prominence):
    best_overall_score = float('inf')
    best_overall_pattern = []
    best_overall_nm = (0, 0)
    l_peaks, _ = find_peaks(l_comp_sig, prominence=prominence)
    r_peaks, _ = find_peaks(r_comp_sig, prominence=prominence)
    l_peak_frames = segment_df.iloc[l_peaks]['frame'].values
    r_peak_frames = segment_df.iloc[r_peaks]['frame'].values
    print(f"  Found {len(l_peak_frames)} potential Left peaks and {len(r_peak_frames)} potential Right peaks.")

    for n_l in range(MIN_STEPS_PER_FOOT, MAX_STEPS_PER_FOOT + 1):
        for n_r in [n_l - 1, n_l, n_l + 1]:
            if n_r < MIN_STEPS_PER_FOOT:
                continue
            best_score_for_nm = float('inf')
            best_pattern_for_nm = []
            total_steps = n_l + n_r
            if total_steps <= 1:
                continue
            avg_step_time = len(segment_df) / total_steps
            min_st = max(10, int(avg_step_time * 0.8))
            max_st = min(MAX_STRIDE_TIME_FRAMES, int(avg_step_time * 1.2))
            for step_time in range(min_st, max_st + 1):
                if step_time == 0:
                    continue
                for phase_shift in range(step_time):
                    for start_side in ['L', 'R']:
                        ideal_strikes, l_count, r_count = [], 0, 0
                        for i in range(total_steps):
                            frame = segment_df['frame'].iloc[0] + phase_shift + (i * step_time)
                            if frame > segment_df['frame'].iloc[-1]:
                                break
                            side = (
                                start_side if i % 2 == 0
                                else ('R' if start_side == 'L' else 'L')
                            )
                            if (side == 'L' and l_count < n_l) or (side == 'R' and r_count < n_r):
                                ideal_strikes.append({'frame': frame, 'side': side})
                                if side == 'L':
                                    l_count += 1
                                else:
                                    r_count += 1
                        if not ideal_strikes or l_count != n_l or r_count != n_r:
                            continue
                        snapped_strikes, score = [], 0
                        is_valid_pattern = True
                        for strike in ideal_strikes:
                            real_peaks = (
                                l_peak_frames if strike['side'] == 'L' else r_peak_frames
                            )
                            if len(real_peaks) == 0:
                                score = float('inf')
                                break
                            distances = np.abs(real_peaks - strike['frame'])
                            closest_peak_idx = np.argmin(distances)
                            snapped_frame = real_peaks[closest_peak_idx]
                            l_heel_x_vals = segment_df.loc[
                                segment_df['frame'] == snapped_frame, 'LHeel_x_smoothed'
                            ]
                            r_heel_x_vals = segment_df.loc[
                                segment_df['frame'] == snapped_frame, 'RHeel_x_smoothed'
                            ]
                            if l_heel_x_vals.empty or r_heel_x_vals.empty:
                                is_valid_pattern = False
                                break
                            l_heel_x = l_heel_x_vals.iloc[0]
                            r_heel_x = r_heel_x_vals.iloc[0]
                            if pd.isna(l_heel_x) or pd.isna(r_heel_x):
                                is_valid_pattern = False
                                break
                            # Soft direction check: penalize but don't reject.
                            # In pose-based coords, x increases left-to-right (same as world coords),
                            # so the heel x-position validation logic is unchanged.
                            is_correct_pos = False
                            if strike['side'] == 'L':
                                if ((direction == 'L_to_R' and l_heel_x > r_heel_x) or
                                        (direction == 'R_to_L' and l_heel_x < r_heel_x)):
                                    is_correct_pos = True
                            else:
                                if ((direction == 'L_to_R' and r_heel_x > l_heel_x) or
                                        (direction == 'R_to_L' and r_heel_x < l_heel_x)):
                                    is_correct_pos = True
                            snap_cost = distances[closest_peak_idx] ** 2
                            if is_correct_pos:
                                score += snap_cost
                            else:
                                score += snap_cost + (step_time * 0.5) ** 2
                            snapped_strikes.append((snapped_frame, strike['side']))
                        if is_valid_pattern and score < best_score_for_nm:
                            best_score_for_nm = score
                            best_pattern_for_nm = sorted(
                                list(set(snapped_strikes)), key=lambda x: x[0]
                            )
            if best_score_for_nm < best_overall_score:
                best_overall_score = best_score_for_nm
                best_overall_pattern = best_pattern_for_nm
                best_overall_nm = (n_l, n_r)

    print(f"  --> Best pattern: (L={best_overall_nm[0]}, R={best_overall_nm[1]}) "
          f"with score {best_overall_score:.2f}")
    return best_overall_pattern, l_peak_frames, r_peak_frames

# ==============================================================================
# --- 6. Gait Cycle Construction ---
# ==============================================================================
def build_gait_cycles(left_strikes, right_strikes, exclusion_frames, strike_direction):
    """
    Convert heel strike lists into full gait cycles (same foot to same foot).
    Only pairs consecutive strikes that don't span an excluded zone (turn/inactive).
    Each cycle is {start, end, side, direction}.
    """
    cycles = []
    sorted_l = sorted(left_strikes)
    sorted_r = sorted(right_strikes)
    for side, strikes in [('L', sorted_l), ('R', sorted_r)]:
        for i in range(len(strikes) - 1):
            start, end = strikes[i], strikes[i + 1]
            # Skip cycles that cross excluded zones
            if any(f in exclusion_frames for f in range(start, end + 1)):
                continue
            cycles.append({
                'start': int(start),
                'end': int(end),
                'side': side,
                'direction': strike_direction.get(start, 'unknown')
            })
    cycles.sort(key=lambda c: c['start'])
    return cycles

# ==============================================================================
# --- 6b. Heel Strike Screenshot Extraction ---
# ==============================================================================
def screenshot_heel_strikes(video_path, left_strikes, right_strikes, output_dir):
    """
    Extract video frames at detected heel strike moments and save as PNG images.

    Args:
        video_path: Path to the source video file
        left_strikes: List of frame numbers for left heel strikes
        right_strikes: List of frame numbers for right heel strikes
        output_dir: Base output directory (screenshots go into heel_strike_screenshots/ subfolder)

    Returns:
        List of saved screenshot file paths
    """
    print("\n--- Extracting Heel Strike Screenshots ---")

    screenshot_dir = os.path.join(output_dir, 'heel_strike_screenshots')
    os.makedirs(screenshot_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Could not open video file: {video_path}")
        return []

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {video_path} ({total_video_frames} frames)")

    # Combine all strikes with side labels, sorted by frame number
    all_strikes = [(f, 'L') for f in left_strikes] + [(f, 'R') for f in right_strikes]
    all_strikes.sort(key=lambda x: x[0])

    saved_paths = []
    for frame_num, side in all_strikes:
        if frame_num >= total_video_frames:
            print(f"  Warning: Frame {frame_num} exceeds video length ({total_video_frames}), skipping.")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"  Warning: Could not read frame {frame_num}, skipping.")
            continue

        filename = f'{frame_num}_{side}_heel_strike.png'
        filepath = os.path.join(screenshot_dir, filename)
        cv2.imwrite(filepath, frame)
        saved_paths.append(filepath)

    cap.release()
    print(f"  Saved {len(saved_paths)} heel strike screenshots to: {screenshot_dir}")
    return saved_paths

# ==============================================================================
# --- 7. Plot Generation ---
# ==============================================================================
def generate_plots(df_for_analysis, cleaned_df, exclusion_frames, final_l_strikes,
                   final_r_strikes, all_l_peaks, all_r_peaks, subject_id, output_dir,
                   generate_individual_plots):
    print("\n--- Generating Plots ---")
    saved_paths = []

    # Main gait cycle detection plot
    fig_gait, ax_gait = plt.subplots(figsize=(15, 7))
    full_l_signal = create_composite_signal(df_for_analysis, 'L')
    full_r_signal = create_composite_signal(df_for_analysis, 'R')
    ax_gait.plot(df_for_analysis['frame'], full_l_signal, label='Left Leg Composite', color='blue')
    ax_gait.plot(
        df_for_analysis['frame'], full_r_signal,
        label='Right Leg Composite', color='red', linestyle='--'
    )

    df_for_plotting = df_for_analysis.set_index('frame')
    if len(all_l_peaks) > 0:
        valid_l = [p for p in all_l_peaks if p in df_for_plotting.index]
        if valid_l:
            ax_gait.plot(
                valid_l, full_l_signal[df_for_plotting.index.isin(valid_l)],
                "o", color='lightblue', markersize=5, label='Potential Left Peaks', linestyle='None'
            )
    if len(all_r_peaks) > 0:
        valid_r = [p for p in all_r_peaks if p in df_for_plotting.index]
        if valid_r:
            ax_gait.plot(
                valid_r, full_r_signal[df_for_plotting.index.isin(valid_r)],
                "o", color='lightpink', markersize=5, label='Potential Right Peaks', linestyle='None'
            )
    if final_l_strikes:
        valid_ls = [s for s in final_l_strikes if s in df_for_plotting.index]
        if valid_ls:
            ax_gait.plot(
                valid_ls, full_l_signal[df_for_plotting.index.isin(valid_ls)],
                "x", color='cyan', markersize=10, mew=2, label='Left Heel-Strike', linestyle='None'
            )
    if final_r_strikes:
        valid_rs = [s for s in final_r_strikes if s in df_for_plotting.index]
        if valid_rs:
            ax_gait.plot(
                valid_rs, full_r_signal[df_for_plotting.index.isin(valid_rs)],
                "x", color='magenta', markersize=10, mew=2, label='Right Heel-Strike', linestyle='None'
            )

    filtered_df = df_for_analysis[df_for_analysis['frame'].isin(exclusion_frames)]
    for _, group in filtered_df.groupby((filtered_df['frame'].diff() > 1).cumsum()):
        if not group.empty:
            label = (
                'Excluded Zone'
                if 'Excluded Zone' not in [p.get_label() for p in ax_gait.patches]
                else ""
            )
            ax_gait.axvspan(
                group['frame'].iloc[0], group['frame'].iloc[-1],
                color='gray', alpha=0.3, label=label
            )

    ax_gait.set_title(f'Gait Cycle Detection - Subject {subject_id}')
    ax_gait.set_xlabel('Frame Number')
    ax_gait.set_ylabel('Normalized Composite Signal')
    ax_gait.legend()
    ax_gait.grid(True, alpha=0.5)
    plot_path = os.path.join(output_dir, f'gait_detection_{subject_id}.png')
    fig_gait.savefig(plot_path, dpi=150, bbox_inches='tight')
    saved_paths.append(plot_path)
    plt.close(fig_gait)
    print(f"  Saved: {plot_path}")

    if generate_individual_plots:
        print("  Generating individual keypoint plots...")
        comparison_kps = ['Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle', 'Heel']
        for kp_base in comparison_kps:
            for coord in ['x', 'y']:
                l_col = f'L{kp_base}_{coord}_smoothed'
                r_col = f'R{kp_base}_{coord}_smoothed'
                if l_col not in df_for_analysis.columns or r_col not in df_for_analysis.columns:
                    continue

                # Left side plot
                fig_l, ax_l = plt.subplots(figsize=(15, 5))
                ax_l.plot(
                    cleaned_df['frame'], cleaned_df[l_col],
                    '--', color='gray', alpha=0.5, label='Pre-Correction'
                )
                ax_l.plot(
                    df_for_analysis['frame'], df_for_analysis[l_col],
                    '-', color='blue', label='Post-Correction'
                )
                ax_l.set_title(f'{l_col} - Subject {subject_id}')
                ax_l.set_xlabel('Frame Number')
                ax_l.set_ylabel(f'{coord}-coordinate')
                ax_l.legend()
                ax_l.grid(True, alpha=0.5)
                path_l = os.path.join(output_dir, f'kp_{subject_id}_{l_col}.png')
                fig_l.savefig(path_l, dpi=100, bbox_inches='tight')
                saved_paths.append(path_l)
                plt.close(fig_l)

                # Right side plot
                fig_r, ax_r = plt.subplots(figsize=(15, 5))
                ax_r.plot(
                    cleaned_df['frame'], cleaned_df[r_col],
                    '--', color='gray', alpha=0.5, label='Pre-Correction'
                )
                ax_r.plot(
                    df_for_analysis['frame'], df_for_analysis[r_col],
                    '-', color='red', label='Post-Correction'
                )
                ax_r.set_title(f'{r_col} - Subject {subject_id}')
                ax_r.set_xlabel('Frame Number')
                ax_r.set_ylabel(f'{coord}-coordinate')
                ax_r.legend()
                ax_r.grid(True, alpha=0.5)
                path_r = os.path.join(output_dir, f'kp_{subject_id}_{r_col}.png')
                fig_r.savefig(path_r, dpi=100, bbox_inches='tight')
                saved_paths.append(path_r)
                plt.close(fig_r)

        print(f"  Generated {len(saved_paths) - 1} keypoint plots.")

    return saved_paths

# ==============================================================================
# --- 8. Main Analysis Function ---
# ==============================================================================
def process_video(raw_df, landmarks_data, subject_id, output_dir, fps, args):
    print(f"--- MediaPipe Gait Analysis (v5.0.0) ---")
    print(f"--- Subject: {subject_id} ---")
    print(f"--- Frames: {len(raw_df)}, FPS: {fps} ---\n")

    # Step 1: Clean and smooth
    print("--- Step 1: Cleaning and Smoothing Data ---")
    cleaned_df = clean_tracking_data(raw_df.copy(), args.confidence_threshold)
    if cleaned_df.empty:
        print("ERROR: No valid data after cleaning.")
        return None
    print(f"  Data cleaned. {len(cleaned_df)} frames.")

    # Step 2: L/R Correction (optional)
    print("\n--- Step 2: Data Correction ---")
    df_for_analysis = cleaned_df.copy()
    if args.perform_lr_correction:
        df_for_analysis = correct_lr_with_discontinuity(
            df_for_analysis.set_index('frame'), DISCONTINUITY_ACCEL_THRESHOLD
        ).reset_index()
    else:
        print("  L/R correction disabled.")

    # Step 3: Turn detection and trimming
    print("\n--- Step 3: Turn Detection & Trimming ---")
    turn_frames = detect_walking_segments(df_for_analysis)
    exclusion_frames = set()
    if not args.no_trim:
        exclusion_frames = get_analysis_exclusion_zones(
            df_for_analysis, turn_frames,
            args.velocity_threshold, TURN_EXCLUSION_WINDOW_FRAMES
        )
    else:
        print("  Trimming disabled.")

    # Step 4: Gait cycle detection
    print("\n--- Step 4: Detecting Gait Cycles (Rhythmic Template Matching) ---")
    gait_analysis_df = df_for_analysis[~df_for_analysis['frame'].isin(exclusion_frames)]
    final_l_strikes, final_r_strikes = [], []
    all_l_peaks, all_r_peaks = [], []
    strike_direction = {}

    clean_segments = gait_analysis_df.groupby(
        (gait_analysis_df['frame'].diff() > 1).cumsum()
    )

    for i, (_, segment) in enumerate(clean_segments):
        print(f"\n  Segment {i + 1} (Frames {segment['frame'].iloc[0]} - "
              f"{segment['frame'].iloc[-1]})")
        if len(segment) < (MIN_STRIDE_TIME_FRAMES * 2):
            print("    Too short, skipping.")
            continue

        l_comp_sig = create_composite_signal(segment, 'L')
        r_comp_sig = create_composite_signal(segment, 'R')

        valid_x = segment['MidHip_x_smoothed'].dropna()
        if len(valid_x) < 2:
            print("    Insufficient MidHip data, skipping.")
            continue
        direction = 'L_to_R' if valid_x.iloc[-1] > valid_x.iloc[0] else 'R_to_L'
        prominence = (
            GAIT_PEAK_PROMINENCE_LTR if direction == 'L_to_R'
            else GAIT_PEAK_PROMINENCE_RTL
        )

        best_pattern, l_p, r_p = find_best_gait_pattern_iterative(
            segment, l_comp_sig, r_comp_sig, direction, prominence
        )

        for frame_num, side in best_pattern:
            strike_direction[frame_num] = direction
        final_l_strikes.extend([s[0] for s in best_pattern if s[1] == 'L'])
        final_r_strikes.extend([s[0] for s in best_pattern if s[1] == 'R'])
        all_l_peaks.extend(l_p)
        all_r_peaks.extend(r_p)

    final_l_strikes = sorted(final_l_strikes)
    final_r_strikes = sorted(final_r_strikes)
    print(f"\n--- Results ---")
    print(f"  Left Heel-Strikes:  {len(final_l_strikes)} at frames {final_l_strikes}")
    print(f"  Right Heel-Strikes: {len(final_r_strikes)} at frames {final_r_strikes}")

    # Step 4b: Screenshot heel strikes (if requested)
    if args.screenshot_heel_strikes:
        screenshot_heel_strikes(
            args.video_file, final_l_strikes, final_r_strikes, output_dir
        )

    # Step 5: Build gait cycles
    gait_cycles = build_gait_cycles(
        final_l_strikes, final_r_strikes, exclusion_frames, strike_direction
    )
    print(f"  Gait Cycles: {len(gait_cycles)}")
    for c in gait_cycles:
        duration = (c['end'] - c['start']) / fps
        print(f"    {c['side']} ({c['direction']}): frame {c['start']} -> {c['end']} ({duration:.2f}s)")

    # Step 6: Generate plots (if requested)
    if args.plots:
        generate_plots(
            df_for_analysis, cleaned_df, exclusion_frames,
            final_l_strikes, final_r_strikes, all_l_peaks, all_r_peaks,
            subject_id, output_dir, True
        )

    # Step 7: Write output JSON
    output_data = {
        'gait_cycles': gait_cycles,
        'landmark_data': landmarks_data,
    }
    output_json_path = os.path.join(output_dir, f'{subject_id}_gait_analysis.json')
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Output JSON saved: {output_json_path}")

    return output_data

# ==============================================================================
# --- 9. Entry Point ---
# ==============================================================================
if __name__ == '__main__':
    args = parse_args()

    # Validate screenshot arguments
    if args.screenshot_heel_strikes:
        if args.video_file is None:
            print("ERROR: --video-file is required when --screenshot-heel-strikes is enabled.")
            sys.exit(1)
        if cv2 is None:
            print("ERROR: opencv-python is required for --screenshot-heel-strikes. "
                  "Install with: pip install opencv-python")
            sys.exit(1)
        if not os.path.isfile(args.video_file):
            print(f"ERROR: Video file not found: {args.video_file}")
            sys.exit(1)

    # Derive subject ID from filename if not provided
    subject_id = args.subject_id
    if subject_id is None:
        basename = os.path.basename(args.landmarks_json)
        subject_id = basename.split('_')[0]
        print(f"Auto-detected subject ID: {subject_id}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading landmarks from: {args.landmarks_json}")
    raw_df, landmarks_data = load_mediapipe_json(args.landmarks_json)
    if raw_df.empty:
        print("ERROR: No data loaded from landmarks file.")
        sys.exit(1)
    print(f"Loaded {len(raw_df)} frames with {len(landmarks_data)} landmark entries.\n")

    # Run analysis
    result = process_video(raw_df, landmarks_data, subject_id, args.output_dir, args.fps, args)
    if result is None:
        print("\nAnalysis failed.")
        sys.exit(1)

    print("\n--- Analysis complete. ---")
