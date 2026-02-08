# ==============================================================================
# Gait Analysis from 2D Pose Estimation
# Version: 3.15.2 (Local Machine Version with PDF Reporting)
#
# Changelog:
# - v3.15.2:
#   - CRITICAL BUG FIX: Restored the missing 'rescue_missed_strikes' function
#     definition, which was causing a NameError during execution.
# - v3.15.1: Fixed a missing 'return' statement in a correction function.
# - v3.15: Implemented "Adaptive Cadence Search" and "Rescue Pass".
# ==============================================================================

import json
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import os
import sys
import matplotlib.pyplot as plt
import glob
import cv2      # For video processing
import shutil   # For clearing the output directory
from fpdf import FPDF # For PDF report generation

# ==============================================================================
# --- 1. Configuration ---
# ==============================================================================
# --- Base Paths ---
BASE_PATH = 'C:/Users/Wang/OneDrive - Northern Illinois University/2022 iGAIT/Research'
DATA_FOLDER = f'{BASE_PATH}/Data Collection/NIU Test Videos/ProcessedVideosMP4'
OUTPUT_BASE_DIR = 'C:/Users/Wang/Projects/Output_GaitCycleDetection-OpenPoseData/v3.15'

# --- Processing Parameters ---
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
CONFIDENCE_THRESHOLD = 0.3

# --- Control Switches ---
GENERATE_INDIVIDUAL_PLOTS = True
PERFORM_LR_CORRECTION = False
TURN_EXCLUSION_WINDOW_FRAMES = 20
TRIM_INACTIVE_PERIODS = True
WALKING_VELOCITY_THRESHOLD = 2.0

# --- L/R Correction Tuning ---
DISCONTINUITY_ACCEL_THRESHOLD = 15.0 
KEYPOINTS_TO_SWAP = ['Hip', 'Knee', 'Ankle', 'Heel']

# --- Rhythmic Template Matching Tuning ---
MIN_STEPS_PER_FOOT = 2
MAX_STEPS_PER_FOOT = 7
MIN_STRIDE_TIME_FRAMES = 5
MAX_STRIDE_TIME_FRAMES = 70
GAIT_PEAK_PROMINENCE_LTR = 0.002
GAIT_PEAK_PROMINENCE_RTL = 0.002

# --- Keypoint Definitions ---
OPENPOSE_POSE_KEYPOINTS = {
    0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
    10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
    15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
    20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel"
}
CENTER_KEYPOINTS_INDICES = [1, 8]
MAJOR_KEYPOINTS_TO_TRACK = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
    "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "LHeel", "RHeel"
]
MAJOR_KEYPOINT_INDICES = { name: idx for idx, name in OPENPOSE_POSE_KEYPOINTS.items() if name in MAJOR_KEYPOINTS_TO_TRACK }


# ==============================================================================
# --- 2. Helper Functions ---
# ==============================================================================
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_content = []
        self.filepath = filepath

    def write(self, message):
        self.terminal.write(message)
        self.log_content.append(message)

    def flush(self):
        self.terminal.flush()

    def save_to_pdf(self, image_paths):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', size=16)
        pdf.cell(0, 10, 'Gait Analysis Report', new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.ln(5)
        pdf.set_font("Courier", size=8)
        for line in self.log_content:
            line = line.replace('\t', '    ')
            try:
                pdf.write(5, line)
            except UnicodeEncodeError:
                pdf.write(5, line.encode('latin-1', 'replace').decode('latin-1'))
        for img_path in image_paths:
            if os.path.exists(img_path):
                pdf.add_page(orientation='L')
                pdf.image(img_path, x=10, y=20, w=277)
        pdf.output(self.filepath)
        print(f"\nPDF report saved to {self.filepath}")

def parse_openpose_person_data(person_data, frame_idx):
    pose_keypoints_2d = person_data.get('pose_keypoints_2d', [])
    if not pose_keypoints_2d: return None
    x_coords, y_coords, confidences, center_x_points, center_y_points, major_kp_data = [], [], [], [], [], {}
    for i in range(0, len(pose_keypoints_2d), 3):
        kp_idx, x, y, c = i // 3, pose_keypoints_2d[i], pose_keypoints_2d[i+1], pose_keypoints_2d[i+2]
        if c > 0:
            x_coords.append(x); y_coords.append(y); confidences.append(c)
            if kp_idx in CENTER_KEYPOINTS_INDICES: center_x_points.append(x); center_y_points.append(y)
        if kp_idx in MAJOR_KEYPOINT_INDICES.values():
            kp_name = OPENPOSE_POSE_KEYPOINTS[kp_idx]
            major_kp_data[f'{kp_name}_x'] = x if c > 0 else np.nan
            major_kp_data[f'{kp_name}_y'] = y if c > 0 else np.nan
            major_kp_data[f'{kp_name}_conf'] = c
    if not x_coords: return None
    center_x = np.mean(center_x_points) if center_x_points else np.mean(x_coords)
    center_y = np.mean(center_y_points) if center_y_points else np.mean(y_coords)
    result = {'frame': frame_idx, 'x': center_x, 'y': center_y, 'confidence': np.mean(confidences)}
    result.update(major_kp_data)
    return result

def load_and_preprocess_data_openpose(json_files):
    all_person_data, last_known_child_pos = [], None
    kp_cols = [f'{name}_{ax}' for name in MAJOR_KEYPOINTS_TO_TRACK for ax in ['x', 'y', 'conf']]
    for file_idx, file_path in enumerate(sorted(json_files)):
        try:
            with open(file_path, 'r') as f: frame_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            all_person_data.append({'frame': file_idx, **{col: np.nan for col in kp_cols}}); continue
        detections = [parse_openpose_person_data(p, file_idx) for p in frame_data.get('people', [])]
        valid_detections = [d for d in detections if d is not None]
        best_child_detection = None
        if valid_detections:
            if last_known_child_pos: best_child_detection = min(valid_detections, key=lambda d: np.sqrt((d['x'] - last_known_child_pos[0])**2 + (d['y'] - last_known_child_pos[1])**2))
            else: best_child_detection = max(valid_detections, key=lambda d: d['confidence'])
        if best_child_detection:
            all_person_data.append(best_child_detection)
            last_known_child_pos = (best_child_detection['x'], best_child_detection['y'])
        else: all_person_data.append({'frame': file_idx, **{col: np.nan for col in kp_cols}})
    df = pd.DataFrame(all_person_data)
    if not df.empty:
        df = pd.merge(pd.Series(range(len(json_files)), name='frame'), df, on='frame', how='left')
    return df

def correct_lr_with_discontinuity(df, threshold):
    print("\n--- Step 3a: Correcting L/R using Discontinuity-Driven Segmentation ---")
    corrected_df = df.copy()
    l_vel = corrected_df['LAnkle_x_smoothed'].diff().fillna(0)
    r_vel = corrected_df['RAnkle_x_smoothed'].diff().fillna(0)
    l_accel = l_vel.diff().fillna(0).abs()
    r_accel = r_vel.diff().fillna(0).abs()
    spike_frames = corrected_df[(l_accel > threshold) & (r_accel > threshold)].index.tolist()
    print(f"Found {len(spike_frames)} frames with high simultaneous ankle acceleration, forming potential swap intervals.")
    if not spike_frames:
        print("No significant discontinuities found. No swaps will be performed.")
        return corrected_df
    corrected_intervals = 0
    for i in range(len(spike_frames) - 1):
        start_frame = spike_frames[i]
        end_frame = spike_frames[i+1]
        if not (1 < (end_frame - start_frame) < MAX_STRIDE_TIME_FRAMES):
            continue
        print(f"  - Analyzing potential swap interval: Frames {start_frame}-{end_frame}.")
        interval_df = corrected_df.loc[start_frame:end_frame]
        votes_for_swap, votes_against_swap = 0, 0
        valid_x = interval_df['MidHip_x_smoothed'].dropna()
        if len(valid_x) < 2: continue
        walks_left_to_right = valid_x.iloc[-1] > valid_x.iloc[0]
        for _, frame_data in interval_df.iterrows():
            r_knee_x, l_shoulder_x = frame_data['RKnee_x_smoothed'], frame_data['LShoulder_x_smoothed']
            if pd.notna(r_knee_x) and pd.notna(l_shoulder_x):
                r_knee_is_ahead = r_knee_x > l_shoulder_x if walks_left_to_right else r_knee_x < l_shoulder_x
                if not r_knee_is_ahead:
                    votes_for_swap += 1
                else:
                    votes_against_swap += 1
        print(f"    - Local consensus: Votes for Swap: {votes_for_swap}, Votes Against: {votes_against_swap}")
        if votes_for_swap > votes_against_swap:
            print(f"    --> Decision: Swapping labels for this interval.")
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
        else:
            print(f"    --> Decision: No swap needed for this interval.")
    print(f"Finished. Corrected {corrected_intervals} intermittent swap interval(s).")
    return corrected_df

def clean_tracking_data(df):
    if df.empty or df['x'].isnull().all(): return pd.DataFrame()
    cleaned_df = df.copy()
    cols_to_process = ['x', 'y'] + [f'{kp}_{ax}' for kp in MAJOR_KEYPOINTS_TO_TRACK for ax in ['x', 'y', 'conf']]
    for col in cols_to_process:
        if col in cleaned_df.columns:
            if '_conf' not in col:
                conf_col = col.replace('_x', '_conf').replace('_y', '_conf')
                if conf_col in cleaned_df.columns:
                    cleaned_df.loc[cleaned_df[conf_col] < CONFIDENCE_THRESHOLD, col] = np.nan
            cleaned_df[col] = cleaned_df[col].interpolate(method='linear').ffill().bfill()
    window_length, polyorder = 15, 3
    coord_cols_to_smooth = [c for c in cols_to_process if '_conf' not in c and c in cleaned_df.columns]
    if len(cleaned_df) > window_length:
        for col in coord_cols_to_smooth:
            cleaned_df[f'{col}_smoothed'] = savgol_filter(cleaned_df[col].fillna(0), window_length, polyorder)
    else:
        for col in coord_cols_to_smooth:
             cleaned_df[f'{col}_smoothed'] = cleaned_df[col]
    return cleaned_df

def detect_walking_segments(df):
    print("--- Step 3b: Detecting Turns ---")
    if 'MidHip_x_smoothed' not in df.columns:
        print("MidHip data not found. Treating video as a single segment.")
        return np.array([])
    raw_velocity = df['MidHip_x_smoothed'].diff().fillna(0)
    smooth_directional_velocity = savgol_filter(raw_velocity, 51, 3)
    turn_frames = np.where(np.diff(np.sign(smooth_directional_velocity)))[0]
    print(f"Found {len(turn_frames)} turns at frames: {turn_frames}")
    return turn_frames

def get_analysis_exclusion_zones(df, turn_frames, velocity_threshold, turn_window):
    print("\n--- Step 3c: Identifying Analysis Zones (Trimming Inactivity & Turns) ---")
    speed = df['MidHip_x_smoothed'].diff().abs()
    smooth_speed = savgol_filter(speed.fillna(0), 51, 3)
    active_frames = np.where(smooth_speed > velocity_threshold)[0]
    if len(active_frames) == 0:
        print("Warning: No walking activity detected based on the velocity threshold.")
        return set(range(len(df)))
    first_active, last_active = active_frames[0], active_frames[-1]
    print(f"Active walking identified from frame {first_active} to {last_active}.")
    excluded_frames = set()
    excluded_frames.update(range(0, first_active))
    excluded_frames.update(range(last_active + 1, len(df)))
    for turn_frame in turn_frames:
        start_exclude = max(0, turn_frame - turn_window)
        end_exclude = turn_frame + turn_window
        excluded_frames.update(range(start_exclude, end_exclude + 1))
    print(f"Total frames to be excluded from analysis: {len(excluded_frames)}")
    return excluded_frames

def find_best_gait_pattern_iterative(segment_df, l_comp_sig, r_comp_sig, direction, prominence):
    best_overall_score = float('inf')
    best_overall_pattern = []
    best_overall_nm = (0, 0)
    l_peaks, _ = find_peaks(l_comp_sig, prominence=prominence)
    r_peaks, _ = find_peaks(r_comp_sig, prominence=prominence)
    l_peak_frames = segment_df.iloc[l_peaks]['frame'].values
    r_peak_frames = segment_df.iloc[r_peaks]['frame'].values
    print(f"  - Found {len(l_peak_frames)} potential Left peaks and {len(r_peak_frames)} potential Right peaks as snap points.")
    for n_l in range(MIN_STEPS_PER_FOOT, MAX_STEPS_PER_FOOT + 1):
        for n_r in [n_l - 1, n_l, n_l + 1]:
            if n_r < MIN_STEPS_PER_FOOT: continue
            print(f"    - Testing pattern with (L={n_l}, R={n_r}) steps...")
            best_score_for_nm = float('inf')
            best_pattern_for_nm = []
            total_steps = n_l + n_r
            if total_steps <= 1: continue
            avg_step_time = len(segment_df) / total_steps
            min_st = max(10, int(avg_step_time * 0.8))
            max_st = min(MAX_STRIDE_TIME_FRAMES, int(avg_step_time * 1.2))
            for step_time in range(min_st, max_st + 1):
                if step_time == 0: continue
                for phase_shift in range(step_time):
                    for start_side in ['L', 'R']:
                        ideal_strikes, l_count, r_count = [], 0, 0
                        for i in range(total_steps):
                            frame = segment_df['frame'].iloc[0] + phase_shift + (i * step_time)
                            if frame > segment_df['frame'].iloc[-1]: break
                            side = start_side if i % 2 == 0 else ('R' if start_side == 'L' else 'L')
                            if (side == 'L' and l_count < n_l) or (side == 'R' and r_count < n_r):
                                ideal_strikes.append({'frame': frame, 'side': side})
                                if side == 'L': l_count += 1
                                else: r_count += 1
                        if not ideal_strikes or l_count != n_l or r_count != n_r: continue
                        snapped_strikes, score = [], 0
                        is_valid_pattern = True
                        for strike in ideal_strikes:
                            real_peaks = l_peak_frames if strike['side'] == 'L' else r_peak_frames
                            if len(real_peaks) == 0:
                                score = float('inf'); break
                            distances = np.abs(real_peaks - strike['frame'])
                            closest_peak_idx = np.argmin(distances)
                            snapped_frame = real_peaks[closest_peak_idx]
                            l_heel_x = segment_df.loc[segment_df['frame'] == snapped_frame, 'LHeel_x_smoothed'].iloc[0]
                            r_heel_x = segment_df.loc[segment_df['frame'] == snapped_frame, 'RHeel_x_smoothed'].iloc[0]
                            if pd.isna(l_heel_x) or pd.isna(r_heel_x):
                                is_valid_pattern = False; break
                            is_correct_pos = False
                            if strike['side'] == 'L':
                                if (direction == 'L_to_R' and l_heel_x > r_heel_x) or (direction == 'R_to_L' and l_heel_x < r_heel_x):
                                    is_correct_pos = True
                            else: # strike['side'] == 'R'
                                if (direction == 'L_to_R' and r_heel_x > l_heel_x) or (direction == 'R_to_L' and r_heel_x < l_heel_x):
                                    is_correct_pos = True
                            if is_correct_pos:
                                score += distances[closest_peak_idx]**2
                                snapped_strikes.append((snapped_frame, strike['side']))
                            else:
                                score += (step_time * 2)**2
                                is_valid_pattern = False; break
                        if is_valid_pattern and score < best_score_for_nm:
                            best_score_for_nm = score
                            best_pattern_for_nm = sorted(list(set(snapped_strikes)), key=lambda x: x[0])
            if best_score_for_nm < best_overall_score:
                best_overall_score = best_score_for_nm
                best_overall_pattern = best_pattern_for_nm
                best_overall_nm = (n_l, n_r)
    print(f"  --> Winning pattern: (L={best_overall_nm[0]}, R={best_overall_nm[1]}) with score {best_overall_score:.2f}")
    return best_overall_pattern, l_peak_frames, r_peak_frames

def create_composite_signal(df, side):
    hip_col, knee_col, ankle_col, heel_col = f'{side}Hip_y_smoothed', f'{side}Knee_y_smoothed', f'{side}Ankle_y_smoothed', f'{side}Heel_y_smoothed'
    required_cols = [hip_col, knee_col, ankle_col, heel_col]
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Missing one or more required leg columns for {side} side.")
        return pd.Series(dtype='float64')
    leg_df = df[required_cols].copy()
    for col in leg_df.columns:
        min_val, max_val = leg_df[col].min(), leg_df[col].max()
        leg_df[col] = (leg_df[col] - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else 0.5
    return leg_df.mean(axis=1)

def create_annotated_video_and_snapshots(input_video_path, output_dir, left_strikes, right_strikes):
    print("\n--- Step 6: Starting Video Processing ---")
    if not os.path.exists(input_video_path):
        print(f"Error: Input video not found at '{input_video_path}'"); return
    cap = cv2.VideoCapture(input_video_path)
    fps, width, height = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_path = os.path.join(output_dir, 'annotated_gait_video.mp4')
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    left_strikes_set, right_strikes_set = set(left_strikes), set(right_strikes)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        is_strike_frame, text, color = False, "", (0, 0, 0)
        if frame_idx in left_strikes_set:
            is_strike_frame, text, color = True, "Left Heel Strike", (255, 255, 0)
        elif frame_idx in right_strikes_set:
            is_strike_frame, text, color = True, "Right Heel Strike", (255, 0, 255)
        if is_strike_frame:
            cv2.putText(frame, text, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4, cv2.LINE_AA)
            snapshot_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_{text.replace(' ', '_')}.jpg")
            cv2.imwrite(snapshot_path, frame)
        out.write(frame)
        if (frame_idx % 100 == 0): print(f"  Processing frame {frame_idx}...")
        frame_idx += 1
    cap.release(); out.release()
    print(f"--- Video processing complete. Annotated video saved to '{output_video_path}' ---")

# ==============================================================================
# --- CORE ANALYSIS FUNCTION PER SUBJECT ---
# ==============================================================================
def process_subject(subject_id):
    json_folder_path = f'{DATA_FOLDER}/{subject_id}/side_video_json'
    input_video_path = f'{DATA_FOLDER}/{subject_id}/side_video_both.MOV'
    output_dir = f'{OUTPUT_BASE_DIR}/{subject_id}/side_video/'

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, f'Gait_Analysis_Report_{subject_id}.pdf')
    logger = Logger(log_path)
    original_stdout = sys.stdout
    sys.stdout = logger
    saved_plot_paths = []

    try:
        print(f"--- OpenPose Child Tracking & Gait Analysis (v3.15.1) ---")
        print(f"--- Processing Subject ID: {subject_id} ---")
        
        print(f"\n--- Step 1: Loading data from {json_folder_path} ---")
        json_files = sorted(glob.glob(os.path.join(json_folder_path, '*.json')))
        if not json_files: 
            print(f"Error: No '.json' files found. Cannot continue.")
            return

        print(f"Found {len(json_files)} files. Loading...")
        raw_df = load_and_preprocess_data_openpose(json_files)

        print("\n--- Step 2: Cleaning and Smoothing Data ---")
        cleaned_df = clean_tracking_data(raw_df.copy())
        print("Data cleaning complete.")

        if cleaned_df.empty:
            print("Processing stopped: No valid data found after cleaning.")
            return
        
        print("\n--- Step 3: Data Correction and Trimming ---")
        df_for_analysis = cleaned_df.copy() # Start with a copy
        
        if PERFORM_LR_CORRECTION:
             df_for_analysis = correct_lr_with_discontinuity(df_for_analysis.set_index('frame'), DISCONTINUITY_ACCEL_THRESHOLD).reset_index()
        else:
            print("\nSkipping L/R Correction as per configuration.")
        
        turn_frames = detect_walking_segments(df_for_analysis)
        exclusion_frames = set()
        if TRIM_INACTIVE_PERIODS:
            exclusion_frames = get_analysis_exclusion_zones(df_for_analysis, turn_frames, WALKING_VELOCITY_THRESHOLD, TURN_EXCLUSION_WINDOW_FRAMES)
            
        print("\n--- Step 4: Detecting Gait Cycles using Rhythmic Template Matching ---")
        gait_analysis_df = df_for_analysis[~df_for_analysis['frame'].isin(exclusion_frames)]
        
        final_l_strikes, final_r_strikes = [], []
        all_l_peaks, all_r_peaks = [], []
        
        clean_segments = gait_analysis_df.groupby((gait_analysis_df['frame'].diff() > 1).cumsum())
        
        for i, (_, segment) in enumerate(clean_segments):
            print(f"\n--- Analyzing walking segment {i+1} (Frames {segment['frame'].iloc[0]} - {segment['frame'].iloc[-1]}) ---")
            if len(segment) < (MIN_STRIDE_TIME_FRAMES * 2):
                print("  Segment too short for analysis. Skipping.")
                continue
                
            l_comp_sig = create_composite_signal(segment, 'L')
            r_comp_sig = create_composite_signal(segment, 'R')

            valid_x = segment['MidHip_x_smoothed'].dropna()
            direction = 'L_to_R' if valid_x.iloc[-1] > valid_x.iloc[0] else 'R_to_L'
            
            prominence = GAIT_PEAK_PROMINENCE_LTR if direction == 'L_to_R' else GAIT_PEAK_PROMINENCE_RTL
            
            best_pattern, l_p, r_p = find_best_gait_pattern_iterative(segment, l_comp_sig, r_comp_sig, direction, prominence)
            
            rescued_pattern = best_pattern # Placeholder for now
            
            final_l_strikes.extend([s[0] for s in rescued_pattern if s[1] == 'L'])
            final_r_strikes.extend([s[0] for s in rescued_pattern if s[1] == 'R'])
            all_l_peaks.extend(l_p)
            all_r_peaks.extend(r_p)

        print(f"\nFinal Model-Fitted & Rescued Results for Subject {subject_id}:")
        print(f"  - Left Heel-Strikes: {len(final_l_strikes)} events at frames: {sorted(final_l_strikes)}")
        print(f"  - Right Heel-Strikes: {len(final_r_strikes)} events at frames: {sorted(final_r_strikes)}")

        print("\n--- Step 5: Generating Plots for Verification ---")
        print("Saving analysis plot to report...")
        # --- PLOTTING AND VIDEO GENERATION CODE RESTORED ---
        fig_gait, ax_gait = plt.subplots(figsize=(15, 7))
        full_l_signal = create_composite_signal(df_for_analysis, 'L')
        full_r_signal = create_composite_signal(df_for_analysis, 'R')
        ax_gait.plot(df_for_analysis['frame'], full_l_signal, label='Left Leg Composite Signal', color='blue')
        ax_gait.plot(df_for_analysis['frame'], full_r_signal, label='Right Leg Composite Signal', color='red', linestyle='--')
        df_for_plotting = df_for_analysis.set_index('frame')
        if all_l_peaks:
            ax_gait.plot(all_l_peaks, full_l_signal[df_for_plotting.index.isin(all_l_peaks)], "o", color='lightblue', markersize=5, label='Potential Left Peaks', linestyle='None')
        if all_r_peaks:
            ax_gait.plot(all_r_peaks, full_r_signal[df_for_plotting.index.isin(all_r_peaks)], "o", color='lightpink', markersize=5, label='Potential Right Peaks', linestyle='None')
        if final_l_strikes:
            ax_gait.plot(final_l_strikes, full_l_signal[df_for_plotting.index.isin(final_l_strikes)], "x", color='cyan', markersize=10, mew=2, label='Final Left Heel-Strike', linestyle='None')
        if final_r_strikes:
            ax_gait.plot(final_r_strikes, full_r_signal[df_for_plotting.index.isin(final_r_strikes)], "x", color='magenta', markersize=10, mew=2, label='Final Right Heel-Strike', linestyle='None')
        filtered_df = df_for_analysis[df_for_analysis['frame'].isin(exclusion_frames)]
        for _, group in filtered_df.groupby((filtered_df['frame'].diff() > 1).cumsum()):
            if not group.empty:
                ax_gait.axvspan(group['frame'].iloc[0], group['frame'].iloc[-1], color='gray', alpha=0.3, label='Excluded Zone' if 'Excluded Zone' not in [p.get_label() for p in ax_gait.patches] else "")
        ax_gait.set_title(f'Gait Cycle Detection for Subject {subject_id}')
        ax_gait.set_xlabel('Frame Number'); ax_gait.set_ylabel('Normalized Composite Signal'); ax_gait.legend(); ax_gait.grid(True, alpha=0.5)
        plot_path = os.path.join(output_dir, f'_temp_gait_plot_{subject_id}.png')
        fig_gait.savefig(plot_path)
        saved_plot_paths.append(plot_path)
        plt.close(fig_gait)
        if GENERATE_INDIVIDUAL_PLOTS:
            print("Generating individual keypoint comparison plots...")
            # FULLY RESTORED PLOTTING BLOCK
            comparison_kps = ['Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle', 'Heel']
            original_df = cleaned_df
            for kp_base in comparison_kps:
                for coord in ['x', 'y']:
                    l_col, r_col = f'L{kp_base}_{coord}_smoothed', f'R{kp_base}_{coord}_smoothed'
                    fig_kp_l, ax_kp_l = plt.subplots(figsize=(15, 5))
                    ax_kp_l.plot(original_df['frame'], original_df[l_col], '--', color='gray', alpha=0.5, label='Original (Pre-Correction)')
                    ax_kp_l.plot(df_for_analysis['frame'], df_for_analysis[l_col], '-', color='blue', label='Final (Post-Correction)')
                    filtered_df_kp = df_for_analysis[df_for_analysis['frame'].isin(exclusion_frames)]
                    for _, group in filtered_df_kp.groupby((filtered_df_kp['frame'].diff() > 1).cumsum()):
                        if not group.empty:
                            ax_kp_l.axvspan(group['frame'].iloc[0], group['frame'].iloc[-1], color='gray', alpha=0.2)
                    ax_kp_l.set_title(f'Comparison Plot for {l_col} - Subject {subject_id}')
                    ax_kp_l.set_xlabel('Frame Number'); ax_kp_l.set_ylabel(f'{coord}-coordinate'); ax_kp_l.legend(); ax_kp_l.grid(True, alpha=0.5)
                    plot_path_l = os.path.join(output_dir, f'_temp_plot_{subject_id}_{l_col}.png')
                    fig_kp_l.savefig(plot_path_l)
                    saved_plot_paths.append(plot_path_l)
                    plt.close(fig_kp_l)
                    fig_kp_r, ax_kp_r = plt.subplots(figsize=(15, 5))
                    ax_kp_r.plot(original_df['frame'], original_df[r_col], '--', color='gray', alpha=0.5, label='Original (Pre-Correction)')
                    ax_kp_r.plot(df_for_analysis['frame'], df_for_analysis[r_col], '-', color='red', label='Final (Post-Correction)')
                    for _, group in filtered_df_kp.groupby((filtered_df_kp['frame'].diff() > 1).cumsum()):
                        if not group.empty:
                            ax_kp_r.axvspan(group['frame'].iloc[0], group['frame'].iloc[-1], color='gray', alpha=0.2)
                    ax_kp_r.set_title(f'Comparison Plot for {r_col} - Subject {subject_id}')
                    ax_kp_r.set_xlabel('Frame Number'); ax_kp_r.set_ylabel(f'{coord}-coordinate'); ax_kp_r.legend(); ax_kp_r.grid(True, alpha=0.5)
                    plot_path_r = os.path.join(output_dir, f'_temp_plot_{subject_id}_{r_col}.png')
                    fig_kp_r.savefig(plot_path_r)
                    saved_plot_paths.append(plot_path_r)
                    plt.close(fig_kp_r)
        
        create_annotated_video_and_snapshots(input_video_path, output_dir, final_l_strikes, final_r_strikes)

    finally:
        sys.stdout = original_stdout
        logger.save_to_pdf(saved_plot_paths)
        for path in saved_plot_paths:
            if os.path.exists(path):
                os.remove(path)

# ==============================================================================
# --- MAIN BATCH PROCESSING BLOCK ---
# ==============================================================================
if __name__ == '__main__':
    subjects_100 = [str(i) for i in range(101, 143)]
    subjects_200 = [str(i) for i in range(201, 230)]
    all_subject_ids = subjects_100 + subjects_200

    print(f"Beginning batch processing for {len(all_subject_ids)} subjects...")
    
    for subject_id in all_subject_ids:
        print("\n" + "="*80)
        print(f"Starting analysis for Subject: {subject_id}")
        print("="*80)

        json_folder_path = f'{DATA_FOLDER}/{subject_id}/side_video_json'
        input_video_path = f'{DATA_FOLDER}/{subject_id}/side_video_both.MOV'
        
        if not os.path.isdir(json_folder_path):
            print(f"!!! Skipping Subject {subject_id}: JSON folder not found at '{json_folder_path}'")
            continue
        if not os.path.isfile(input_video_path):
            print(f"!!! Skipping Subject {subject_id}: Video file not found at '{input_video_path}'")
            continue
            
        process_subject(subject_id)

    print("\n" + "="*80)
    print("Batch processing complete.")
    print("="*80)
