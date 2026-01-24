import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

# --- Helper Functions ---
def read_csv_files(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    data = {}
    for file in files:
        file_path = os.path.join(folder, file)
        try:
            # Flexible reading: handle headers or no headers
            # Try reading first few bytes to check if empty
            if os.stat(file_path).st_size == 0:
                print(f"Skipped empty file (0 bytes): {file}")
                continue
                
            df = pd.read_csv(file_path, header=None)
            
            # Check if dataframe is effectively empty
            if df.empty or len(df) < 10: # arbitrary small threshold
                print(f"Skipped file with insufficient data: {file}")
                continue

            if isinstance(df.iloc[0,0], str): # Check if first row is a header
                df = pd.read_csv(file_path, header=0)
            
            data[file] = df
        except pd.errors.EmptyDataError:
            print(f"Skipped EmptyDataError: {file}")
        except Exception as e:
            print(f"Skipped file {file} due to error: {e}")
    return data

def handle_nan_and_scale(data, scaler):
    """
    1. Fill NaNs
    2. Apply Standard Scaling (Z-Score)
    """
    # Safety check for empty data
    if data.shape[0] == 0:
        return data # Should be caught by window loop, but safety first
        
    # Handle NaNs
    if np.isnan(data).any():
        col_mean = np.nanmean(data, axis=0)
        inds = np.where(np.isnan(data))
        data[inds] = np.take(col_mean, inds[1])
    
    # Scale
    return scaler.fit_transform(data)

def adjust_keypoints(skeleton_window, key_joint_indexes, joint_order):
    adjusted_skeleton = []
    # skeleton_window is shape (Time, Joints*3)
    
    for joint_index in joint_order:
        if joint_index in key_joint_indexes:
            try:
                src_idx = key_joint_indexes.index(joint_index)
                start_col = src_idx * 3
                adjusted_skeleton.append(skeleton_window[:, start_col:start_col + 3])
            except ValueError:
                continue
            
    return np.hstack(adjusted_skeleton)

# --- Main Dataset Class ---
class SlidingWindowDataset(Dataset):
    def __init__(self, skeleton_data, sensor1_data, sensor2_data, common_files, window_size, overlap, label_encoder):
        self.skeleton_data = skeleton_data
        self.sensor1_data = sensor1_data
        self.sensor2_data = sensor2_data
        self.common_files = list(common_files)
        self.window_size = window_size
        self.overlap = overlap
        self.label_encoder = label_encoder
        
        # SSDL Target Joints (UTD-MHAD Topology)
        self.key_joint_indexes = [0, 2, 3, 5, 6, 7, 12, 13, 14, 18, 19, 20, 22, 23, 24, 26]
        self.joint_order = [26, 3, 2, 0, 5, 6, 7, 12, 13, 14, 18, 19, 20, 22, 23, 24]
        
        self.skeleton_windows, self.sensor1_windows, self.sensor2_windows, self.labels = self._create_windows()

    def _create_windows(self):
        skeleton_windows_list = []
        sensor1_windows_list = []
        sensor2_windows_list = []
        labels_list = []
        
        step = self.window_size - self.overlap

        print("Processing files and creating windows...")
        for file in self.common_files:
            # 1. Load Data
            skeleton_df = self.skeleton_data[file]
            sensor1_df = self.sensor1_data[file]
            sensor2_df = self.sensor2_data[file]
            
            # 2. Extract Label
            try:
                activity_code = file.split('A')[1][:2].lstrip('0')
                label = self.label_encoder.transform([[activity_code]])[0]
            except:
                continue 
            
            # 3. Create Windows
            # Ensure we don't go out of bounds for ANY modality
            min_len = min(len(skeleton_df), len(sensor1_df), len(sensor2_df))
            num_windows = (min_len - self.window_size) // step + 1

            for i in range(num_windows):
                start = i * step
                end = start + self.window_size
                
                # Double check bounds
                if end > min_len: break
                
                # 4. Raw Slices
                skeleton_window = skeleton_df.iloc[start:end, :].values
                sensor1_window = sensor1_df.iloc[start:end, -3:].values
                sensor2_window = sensor2_df.iloc[start:end, -3:].values

                # --- CRITICAL FIX: Check Shapes Before Processing ---
                if skeleton_window.shape[0] != self.window_size: continue
                if sensor1_window.shape[0] != self.window_size: continue
                if sensor2_window.shape[0] != self.window_size: continue
                # ----------------------------------------------------

                # Handle Timestamp Column (Index 0)
                if skeleton_window.shape[1] == 97:
                    skeleton_window = skeleton_window[:, 1:]

                # 5. Extract Key Joints Only
                joint_indices = np.array(self.key_joint_indexes)
                final_indices = np.concatenate([[j * 3, j * 3 + 1, j * 3 + 2] for j in joint_indices])
                
                # Check if we have enough columns
                if skeleton_window.shape[1] < np.max(final_indices):
                    continue
                    
                skeleton_window = skeleton_window[:, final_indices]

                # 6. UNIT CONVERSION (MM -> M)
                skeleton_window = skeleton_window / 1000.0

                # 7. ADJUST KEYPOINTS
                skeleton_window = adjust_keypoints(skeleton_window, self.key_joint_indexes, self.joint_order)

                # 8. CENTER OF MASS NORMALIZATION
                hip_index_in_new_order = 3 
                # Safety check for reshaping
                if skeleton_window.shape[1] % 3 != 0: continue
                
                sk_reshaped = skeleton_window.reshape(self.window_size, -1, 3)
                hip_coords = sk_reshaped[:, hip_index_in_new_order:hip_index_in_new_order+1, :] 
                sk_reshaped = sk_reshaped - hip_coords
                skeleton_window = sk_reshaped.reshape(self.window_size, -1)

                # 9. Scale Sensors
                try:
                    scaler = StandardScaler()
                    sensor1_window = handle_nan_and_scale(sensor1_window, scaler)
                    sensor2_window = handle_nan_and_scale(sensor2_window, scaler)
                except ValueError as e:
                    # Catch empty array errors if they somehow sneak through
                    print(f"Skipping window due to scaler error: {e}")
                    continue

                # Store
                skeleton_windows_list.append(torch.tensor(skeleton_window, dtype=torch.float32))
                sensor1_windows_list.append(sensor1_window)
                sensor2_windows_list.append(sensor2_window)
                labels_list.append(label)

        # --- Oversampling ---
        class_indices = defaultdict(list)
        for idx, lbl in enumerate(labels_list):
            lbl_index = lbl.argmax()
            class_indices[lbl_index].append(idx)

        skeleton_final, sensor1_final, sensor2_final, labels_final = [], [], [], []
        
        # Check if we actually found data
        if not class_indices:
            print("WARNING: No valid windows created! Check your data paths and window_size.")
            return [], [], [], []

        num_classes = len(self.label_encoder.categories_[0]) 

        for lbl, indices in class_indices.items():
            needed = 2000 if len(indices) < 2000 else len(indices)
            selected_indices = random.choices(indices, k=needed)

            for idx in selected_indices:
                skeleton_final.append(skeleton_windows_list[idx])
                sensor1_final.append(sensor1_windows_list[idx])
                sensor2_final.append(sensor2_windows_list[idx])
                labels_final.append(to_one_hot(lbl, num_classes))

        return skeleton_final, sensor1_final, sensor2_final, labels_final

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.skeleton_windows[idx],
            torch.tensor(self.sensor1_windows[idx], dtype=torch.float32),
            torch.tensor(self.sensor2_windows[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

def to_one_hot(label, num_classes):
    one_hot = np.zeros(num_classes)
    one_hot[label] = 1
    return one_hot