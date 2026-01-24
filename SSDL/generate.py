import os
import torch
import argparse
import imageio
import random
import numpy as np
from diffusion_model import Diffusion1D
from diffusion_model.model_loader import load_sensor_model, load_diffusion_model_for_testing
from diffusion_model.skeleton_model import SkeletonLSTMModel
from diffusion_model.diffusion import DiffusionProcess, Scheduler
from torch.utils.data import DataLoader
from diffusion_model.util import prepare_dataset
import matplotlib.pyplot as plt

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def visualize_skeleton(positions, save_path='skeleton_animation.gif'):
    """
    Visualizes the 16-joint SmartFall skeleton correctly.
    Indices:
    0:Head, 1:Neck, 2:Spine, 3:Hip
    4:LSh, 5:LEl, 6:LWr, 7:RSh, 8:REl, 9:RWr
    10:LHip, 11:LKnee, 12:LAnk, 13:RHip, 14:RKnee, 15:RAnk
    """
    
    # CORRECTED TOPOLOGY for SSDL/SmartFall
    connections = [
        (0, 1), (1, 2), (2, 3),        # Spine Chain (Head->Neck->Spine->Hip)
        (1, 4), (4, 5), (5, 6),        # Left Arm (Neck->LSh->LEl->LWr)
        (1, 7), (7, 8), (8, 9),        # Right Arm (Neck->RSh->REl->RWr)
        (3, 10), (10, 11), (11, 12),   # Left Leg (Hip->LHip->LKnee->LAnk)
        (3, 13), (13, 14), (14, 15)    # Right Leg (Hip->RHip->RKnee->RAnk)
    ]
    
    frames = []
    # Dynamic loop range based on actual data shape
    num_frames = positions.shape[1] 
    sample_idx = 0

    for frame_idx in range(num_frames):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Remove background and axes
        ax.set_facecolor('white')
        ax.grid(False)
        ax.set_axis_off()
        
        for joint1, joint2 in connections:
            # Handle potential index errors if data shape is wrong
            try:
                joint1_coords = positions[sample_idx, frame_idx, joint1*3:(joint1*3)+3]
                joint2_coords = positions[sample_idx, frame_idx, joint2*3:(joint2*3)+3]
            except IndexError:
                continue

            if len(joint1_coords) < 3 or len(joint2_coords) < 3:
                continue

            xs = [joint1_coords[0], joint2_coords[0]]
            ys = [joint1_coords[1], joint2_coords[1]]
            zs = [joint1_coords[2], joint2_coords[2]]

            # Plot the bones as dark blue lines
            ax.plot(xs, ys, zs, marker='o', color='darkblue')

            # Plot the joints as red dots
            ax.scatter(joint1_coords[0], joint1_coords[1], joint1_coords[2], color='red', s=50) 
            ax.scatter(joint2_coords[0], joint2_coords[1], joint2_coords[2], color='red', s=50)

        # Set consistent limits to prevent camera jumping
        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([-1.0, 1.0])
        ax.set_zlim([-1.0, 1.0])

        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=10, azim=45) 

        plt.tight_layout()
        
        # Matplotlib 3.8+ compatible buffer extraction
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3] # Get RGB
        frames.append(image)

        plt.close(fig)

    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    imageio.mimsave(save_path, frames, fps=10) # fps=10 is smoother for 48 frames
    print(f'GIF saved as {save_path}')

def check_for_nans(tensor, name):
    if torch.isnan(tensor).any():
        print(f"WARNING: NaNs detected in {name}")
    # else:
    #     print(f"No NaNs in {name}")

def generate_samples(args, sensor_model, diffusion_model, device):
    dataset = prepare_dataset(args)
    # Important: Drop last to ensure full batches, though batch_size=1 is typical for inference
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    diffusion_process = DiffusionProcess(
        scheduler=Scheduler(sched_type='cosine', T=args.timesteps, step=1, device=device),
        device=device,
        ddim_scale=args.ddim_scale
    )

    generated_samples = []
    sensor_model.eval()
    diffusion_model.eval()

    print("Starting generation...")
    
    with torch.no_grad():
        try:
            _, sensor1, sensor2, label = next(iter(dataloader))
        except StopIteration:
            print("Dataset is empty or batch size too large.")
            return None, None

        label_index = torch.argmax(label, dim=1)
        sensor1, sensor2 = sensor1.to(device), sensor2.to(device)
        
        # Get Context from Sensor Model
        _, context = sensor_model(sensor1, sensor2, return_attn_output=True)
        check_for_nans(context, "context") 

        # Generate
        # FIX: Use args.window_size instead of hardcoded 90
        generated_sample = diffusion_process.generate(
            model=diffusion_model, 
            context=context, 
            label=label_index, 
            shape=(args.batch_size, args.window_size, 48), # Dynamic shape 
            steps=args.timesteps, 
            predict_noise=False
        )
        
        check_for_nans(generated_sample, "generated_sample")
        generated_samples.append(generated_sample.cpu())

        # Save numpy file for analysis
        current_class = label_index.item()
        npy_path = os.path.join(args.output_dir, f"generated_motion_class_{current_class}.npy")
        np.save(npy_path, generated_sample.cpu().numpy())
        print(f"Saved motion data to {npy_path}")

    generated_samples = torch.cat(generated_samples, dim=0)
    
    return generated_samples, label_index

def load_skeleton_model(skeleton_model_path, skeleton_model):
    if not os.path.exists(skeleton_model_path):
        print(f"Warning: Skeleton model not found at {skeleton_model_path}")
        return skeleton_model
        
    checkpoint = torch.load(skeleton_model_path)
    new_state_dict = {}
    for key in checkpoint:
        new_key = key.replace("module.", "") 
        new_state_dict[new_key] = checkpoint[key]

    skeleton_model.load_state_dict(new_state_dict, strict=False)
    return skeleton_model

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Sensor Model
    sensor_model = load_sensor_model(args, device)
    
    # 2. Load Diffusion Model
    # Explicitly creating instance to ensure window_size compatibility
    diffusion_model = Diffusion1D(
        num_joints=16, 
        num_classes=14, 
        window_size=args.window_size
    ).to(device)
    
    # Load weights
    model_path = os.path.join(args.output_dir, "diffusion_model", "best_diffusion_model.pth")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        # Handle DDP prefix
        new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        diffusion_model.load_state_dict(new_state)
        print("Loaded Diffusion Model weights.")
    else:
        print(f"WARNING: Diffusion model not found at {model_path}")

    # 3. Load Skeleton Model (Evaluator)
    skeleton_model = SkeletonLSTMModel(input_size=48, num_classes=14).to(device)
    skeleton_model = load_skeleton_model(args.skeleton_model_path, skeleton_model)

    # 4. Generate
    generated_samples, label_index = generate_samples(args, sensor_model, diffusion_model, device)
    
    if generated_samples is not None:
        actual_class = label_index.item()
        print(f"Generated Shape: {generated_samples.shape}")
        print(f"Generated Activity Class: {actual_class}")
        
        visualize_skeleton(
            generated_samples.cpu().detach().numpy(),
            save_path=f'./results/generated_class_{actual_class}.gif'
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skeleton_model_path", type=str, default="./results/skeleton_model/best_skeleton_model.pth")
    
    # Paths
    parser.add_argument("--sensor_folder1", type=str, default="/home/qsw26/smartfall/SSDL_drive/labelled_data/metahip")
    parser.add_argument("--sensor_folder2", type=str, default="/home/qsw26/smartfall/SSDL_drive/labelled_data/metahip")
    parser.add_argument("--skeleton_folder", type=str, default="/home/qsw26/smartfall/SSDL_drive/datasets/skeleton")
    parser.add_argument("--output_dir", type=str, default="./results")

    # Config (MUST MATCH TRAIN.PY)
    parser.add_argument("--window_size", type=int, default=48, help="MUST match training window size (48)")
    parser.add_argument("--overlap", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument('--ddim_scale', type=float, default=0.0) # 0.0 for cleaner generation (DDIM)
    
    # Dummy args for compatibility with dataset loader if needed
    parser.add_argument("--train_sensor_model", type=eval, default=False)
    parser.add_argument("--train_skeleton_model", type=eval, default=False)
    parser.add_argument("--test_diffusion_model", type=eval, default=True)
    
    args = parser.parse_args()

    main(args)