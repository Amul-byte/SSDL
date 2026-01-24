import os
import math
import torch
import imageio
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from torch.nn import functional as F
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from .dataset import SlidingWindowDataset, read_csv_files

# ==========================================
# 1. CORE LOSS & TRAINING FUNCTIONS (FIXED)
# ==========================================

def compute_loss(
    args, model, x0, label, context, t, mask=None, noise=None, device="cpu",
    diffusion_process=None, angular_loss=False, lip_reg=False, epoch=None, rank=0, batch_idx=0
):
    """
    Computes diffusion loss (MSE) + Angular Loss + Lipschitz Regularization.
    """
    if noise is None:
        noise = torch.randn_like(x0)

    # Ensure all tensors are on the correct device
    x0 = x0.to(device)
    label = label.to(device)
    context = context.to(device)
    t = t.to(device)
    noise = noise.to(device)

    # Add noise to x0 to get xt
    xt, _ = diffusion_process.add_noise(x0, t)
    
    # Predict
    if diffusion_process.ddim_scale == 1.0:
        # Predict noise (DDPM)
        predicted_noise = model(xt, context, t, sensor_pred=label).to(device)
        mse_loss = F.mse_loss(predicted_noise, noise)
    else:
        # Predict clean data (DDIM)
        x0_pred = model(xt, context, t, sensor_pred=label).to(device)
        mse_loss = F.mse_loss(x0_pred, x0)

        # Visualize sample (Rank 0 only)
        if epoch is not None and batch_idx == 0 and rank == 0:
            if epoch % 50 == 0 or epoch in [1, 99, 599, 999]:
                sample_idx = 0 
                os.makedirs('./gif_tl', exist_ok=True)
                
                # Detach and convert to numpy
                x_gen_np = x0_pred[sample_idx].cpu().detach().numpy()
                x_orig_np = x0[sample_idx].cpu().detach().numpy()

                visualize_skeleton(
                    x_orig_np,
                    save_path=f'./gif_tl/epoch_{epoch}_original.gif'
                )
                visualize_skeleton(
                    x_gen_np,
                    save_path=f'./gif_tl/epoch_{epoch}_generated.gif'
                )

    total_loss = mse_loss

    # --- Angular Loss (Fixed Indices) ---
    if angular_loss:
        # If predicting noise, we skip angular loss or need to reconstruct x0.
        # For this fix, we assume x0_pred is available (DDIM mode) or skip.
        if diffusion_process.ddim_scale != 1.0:
            predicted_angles = compute_joint_angles(x0_pred)
            target_angles = compute_joint_angles(x0)
            diff = target_angles - predicted_angles
            angular_val = torch.norm(diff, p='fro')
            total_loss += 0.05 * angular_val

    # --- Lipschitz Regularization ---
    if lip_reg:
        noisy_context = add_random_noise(context.clone(), noise_std=0.01, noise_fraction=0.2)
        
        if diffusion_process.ddim_scale == 1.0:
            pred_noise_lr = model(xt, noisy_context, t, sensor_pred=label).to(device)
            lip_loss = F.mse_loss(pred_noise_lr, predicted_noise)
        else:
            x0_pred_lr = model(xt, noisy_context, t, sensor_pred=label).to(device)
            lip_loss = F.mse_loss(x0_pred_lr, x0_pred)

        total_loss += 0.05 * lip_loss

    # Return prediction for downstream use
    return total_loss, (x0_pred if diffusion_process.ddim_scale != 1.0 else predicted_noise)


def add_random_noise(context, noise_std=0.01, noise_fraction=0.2):
    num_samples = context.size(0) 
    num_noisy = int(noise_fraction * num_samples)
    if num_noisy > 0:
        indices = torch.randperm(num_samples)[:num_noisy]
        noise = torch.randn_like(context[indices]) * noise_std
        context[indices] += noise
    return context

# ==========================================
# 2. SKELETON & GEOMETRY HELPERS (FIXED)
# ==========================================

def compute_joint_angles(positions):
    """
    Computes angles for SmartFall (16 joints)
    """
    # Tuples: [Joint1, Center, Joint2]
    # We calculate the angle at 'Center'
    triplets = torch.tensor([
        [4, 5, 6],    # L-Shoulder -> L-Elbow -> L-Wrist
        [7, 8, 9],    # R-Shoulder -> R-Elbow -> R-Wrist
        [10, 11, 12], # L-Hip -> L-Knee -> L-Ankle
        [13, 14, 15]  # R-Hip -> R-Knee -> R-Ankle
    ], device=positions.device)
    
    batch, frames, _ = positions.shape
    
    # Reshape to (Batch, Frames, 16, 3)
    if positions.shape[-1] != 48:
        # Safety fallback if shape is wrong
        return torch.tensor(0.0, device=positions.device)

    pos_3d = positions.view(batch, frames, 16, 3)
    
    v1 = pos_3d[:, :, triplets[:, 0]] - pos_3d[:, :, triplets[:, 1]]
    v2 = pos_3d[:, :, triplets[:, 2]] - pos_3d[:, :, triplets[:, 1]]
    
    v1 = F.normalize(v1, p=2, dim=-1)
    v2 = F.normalize(v2, p=2, dim=-1)
    
    dot = torch.sum(v1 * v2, dim=-1)
    dot = torch.clamp(dot, -0.999, 0.999)
    return torch.acos(dot)

def visualize_skeleton(positions, save_path='skeleton.gif'):
    """
    Visualizes SmartFall skeleton (16 joints) without bad connections.
    """
    # 0:Head, 1:Neck, 2:Spine, 3:Hip
    # 4:LSh, 5:LEl, 6:LWr
    # 7:RSh, 8:REl, 9:RWr
    # 10:LHip, 11:LKnee, 12:LAnk
    # 13:RHip, 14:RKnee, 15:RAnk
    connections = [
        (0, 1), (1, 2), (2, 3),      # Spine
        (1, 4), (4, 5), (5, 6),      # Left Arm
        (1, 7), (7, 8), (8, 9),      # Right Arm
        (3, 10), (10, 11), (11, 12), # Left Leg
        (3, 13), (13, 14), (14, 15)  # Right Leg
    ]

    if len(positions.shape) == 2:
        frames = positions.shape[0]
        positions = positions.reshape(frames, 16, 3)
    else:
        frames = positions.shape[0]

    gif_frames = []
    
    for t in range(frames):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()
        
        for i, j in connections:
            x = [positions[t, i, 0], positions[t, j, 0]]
            y = [positions[t, i, 1], positions[t, j, 1]]
            z = [positions[t, i, 2], positions[t, j, 2]]
            ax.plot(x, y, z, color='blue')
            
        ax.scatter(positions[t, :, 0], positions[t, :, 1], positions[t, :, 2], c='red', s=10)
        
        # Lock camera
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        gif_frames.append(image)
        plt.close(fig)
        
    imageio.mimsave(save_path, gif_frames, fps=10)
    print(f"Saved GIF: {save_path}")

# ==========================================
# 3. RESTORED HELPER FUNCTIONS (REQUIRED)
# ==========================================

def calculate_fid(real_activations, generated_activations):
    """
    Calculates Fréchet Inception Distance (FID).
    Required by __init__.py imports.
    """
    real_activations = np.concatenate(real_activations, axis=0)
    generated_activations = np.concatenate(generated_activations, axis=0)

    real_activations = real_activations.reshape(real_activations.shape[0], -1)
    generated_activations = generated_activations.reshape(generated_activations.shape[0], -1)

    mu_real = np.mean(real_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)
    mu_generated = np.mean(generated_activations, axis=0)
    sigma_generated = np.cov(generated_activations, rowvar=False)

    diff = mu_real - mu_generated
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_generated, disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm(sigma_real @ sigma_generated + np.eye(sigma_real.shape[0]) * 1e-6)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_generated) - 2 * np.trace(covmean)
    return fid

def frobenius_norm_loss(predicted, target):
    return torch.norm(predicted - target, p='fro')

def min_max_scale(data, data_min, data_max, feature_range=(0, 1)):
    data_min = np.array(data_min)
    data_max = np.array(data_max)
    scale = (feature_range[1] - feature_range[0]) / (data_max - data_min + 1e-8)
    min_range = feature_range[0]
    return scale * (data - data_min) + min_range

def prepare_dataset(args):
    """
    Wraps the dataset loading using the configuration args.
    """
    skeleton_folder = args.skeleton_folder
    sensor_folder1 = args.sensor_folder1
    sensor_folder2 = args.sensor_folder2

    skeleton_data = read_csv_files(skeleton_folder)
    sensor_data1 = read_csv_files(sensor_folder1)
    sensor_data2 = read_csv_files(sensor_folder2)

    common_files = list(set(skeleton_data.keys()).intersection(set(sensor_data1.keys()), set(sensor_data2.keys())))

    if not common_files:
        raise ValueError("No common files found.")

    # Sort classes
    activity_codes = sorted(set(f.split('A')[1][:2].lstrip('0') for f in common_files))
    label_encoder = OneHotEncoder(sparse_output=False)
    label_encoder.fit([[code] for code in activity_codes])

    dataset = SlidingWindowDataset(
        skeleton_data=skeleton_data,
        sensor1_data=sensor_data1,
        sensor2_data=sensor_data2,
        common_files=common_files,
        window_size=args.window_size,
        overlap=args.overlap,
        label_encoder=label_encoder
    )
    return dataset

def sample_by_t(tensor_to_sample, timesteps, x_shape):
    batch_size = timesteps.shape[0]
    timesteps = timesteps.to(tensor_to_sample.device)
    sampled_tensor = tensor_to_sample.gather(0, timesteps)
    sampled_tensor = torch.reshape(sampled_tensor, (batch_size,) + (1,) * (len(x_shape) - 1))
    return sampled_tensor

def extract_joint_subset(positions):
    # This was for the OLD 32-joint dataset. 
    # Can leave as placeholder or remove if not called.
    return positions 

# --- Noise Schedules ---
def linear_noise_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return np.linspace(beta_start, beta_end, timesteps)

def cosine_noise_schedule(timesteps, s=0.008):
    steps = np.arange(timesteps + 1) / timesteps
    alphas_cumprod = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas

def quadratic_noise_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return np.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_noise_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = np.linspace(-6, 6, timesteps)
    return beta_start + (beta_end - beta_start) / (1 + np.exp(-betas))

def get_noise_schedule(schedule_type, timesteps, beta_start=0.0001, beta_end=0.02):
    if schedule_type == 'linear':
        return linear_noise_schedule(timesteps, beta_start, beta_end)
    elif schedule_type == 'cosine':
        return cosine_noise_schedule(timesteps)
    elif schedule_type == 'quadratic':
        return quadratic_noise_schedule(timesteps, beta_start, beta_end)
    elif schedule_type == 'sigmoid':
        return sigmoid_noise_schedule(timesteps, beta_start, beta_end)
    else:
        raise ValueError(f"Unknown noise schedule type: {schedule_type}")

def create_stratified_split(dataset, test_size=0.3, val_size=0.5, random_state=42):
    labels = [label.argmax().item() for _, _, _, label in dataset] 
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_test_idx = next(stratified_split.split(np.zeros(len(labels)), labels))
    
    stratified_val_test_split = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    val_idx, test_idx = next(stratified_val_test_split.split(np.zeros(len(val_test_idx)), [labels[i] for i in val_test_idx]))

    val_idx = [val_test_idx[i] for i in val_idx]
    test_idx = [val_test_idx[i] for i in test_idx]
    return train_idx, val_idx, test_idx

def get_time_embedding(timestep, dtype):
    half_dim = 320 // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim) * -emb)
    emb = timestep * emb
    emb = np.concatenate((np.sin(emb), np.cos(emb)))
    return torch.tensor(emb, dtype=dtype)

def get_file_path(file_name):
    return os.path.join(os.path.dirname(__file__), file_name)

def rescale(tensor, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    tensor = ((tensor - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    if clamp:
        tensor = torch.clamp(tensor, new_min, new_max)
    return tensor