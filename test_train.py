import os
import torch
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from collections import Counter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.tensorboard import SummaryWriter
from diffusion_model.diffusion import DiffusionProcess, Scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusion_model.model_loader import load_sensor_model, load_diffusion
from diffusion_model.skeleton_model import SkeletonLSTMModel
from diffusion_model.util import (
    prepare_dataset,
    compute_loss,
)

print("""------------
      ----------
      SSDL Training Script
      ----------
      ------------""")


def ensure_dir(path, rank):
    # In single-process, behave like rank 0
    if rank == 0:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    # Only synchronize if distributed is initialized
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def setup(rank, world_size, seed):
    # If single-process, just set seeds and return
    if world_size == 1:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        return

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Use NCCL only when CUDA is available; otherwise use GLOO (CPU-safe)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # Set the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup():
    # Safe cleanup: only destroy if initialized
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _labels_to_class_indices(labels):
    """
    Robustly convert labels to integer class indices.
    - If labels are one-hot/probabilities: argmax
    - If labels are already integer: cast to long
    """
    if isinstance(labels, (list, tuple)):
        labels = torch.tensor(labels)

    if torch.is_tensor(labels):
        if labels.ndim > 1:
            return labels.argmax(dim=1).long()
        return labels.long()

    # Fallback
    labels = torch.tensor(labels)
    if labels.ndim > 1:
        return labels.argmax(dim=1).long()
    return labels.long()


def _fix_out_of_bounds_targets(y, num_classes):
    """
    Prevent CrossEntropyLoss crash when targets contain values >= num_classes.
    Tries a light-touch fix:
      - If labels look 1-indexed (1..C), shift to 0..C-1
      - Otherwise clamp into valid range (keeps training running)
    """
    if num_classes <= 0:
        return y

    if y.numel() == 0:
        return y

    y_max = int(y.max().item())
    if y_max < num_classes:
        return y

    y_min = int(y.min().item())
    # Heuristic: 1-indexed labels case (common): 1..C
    if y_min == 1 and y_max == num_classes:
        return (y - 1).clamp(0, num_classes - 1)

    # Last resort: clamp to valid range to avoid runtime error
    return y.clamp(0, num_classes - 1)


def train_sensor_model(rank, args, device, train_loader, val_loader):
    print("Training Sensor model")
    # Set seed for reproducibility within this process
    torch.manual_seed(args.seed + rank)

    sensor_model = load_sensor_model(args, device)

    # Wrap with DDP only if distributed is initialized
    if args.world_size > 1 and dist.is_available() and dist.is_initialized():
        sensor_model = DDP(sensor_model, find_unused_parameters=True)

    sensor_optimizer = torch.optim.Adam(
        sensor_model.parameters(),
        lr=args.sensor_lr,
        betas=(0.9, 0.98)
    )

    sensor_model_save_dir = os.path.join(args.output_dir, "sensor_model")
    ensure_dir(sensor_model_save_dir, rank)

    sensor_log_dir = os.path.join(sensor_model_save_dir, "sensor_logs")
    ensure_dir(sensor_log_dir, rank)

    writer = SummaryWriter(log_dir=sensor_log_dir) if rank == 0 else None
    best_loss = float('inf')

    for epoch in range(args.sensor_epoch):
        sensor_model.train()
        epoch_train_loss = 0.0

        for _, sensor1, sensor2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.sensor_epoch} (Training)"):
            sensor1, sensor2, labels = sensor1.to(device), sensor2.to(device), labels.to(device)
            sensor_optimizer.zero_grad()

            out = sensor_model(sensor1, sensor2)
            output = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(output, (tuple, list)):
                output = output[0]

            y = _labels_to_class_indices(labels)
            # Fix out-of-bounds targets relative to model output classes
            num_classes = int(output.shape[1]) if output.ndim >= 2 else 0
            y = _fix_out_of_bounds_targets(y, num_classes)

            loss = torch.nn.CrossEntropyLoss()(output, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(sensor_model.parameters(), max_norm=1.0)
            sensor_optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / max(1, len(train_loader))

        # Validation phase
        sensor_model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for _, sensor1, sensor2, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.sensor_epoch} (Validation)"):
                sensor1, sensor2, labels = sensor1.to(device), sensor2.to(device), labels.to(device)

                out = sensor_model(sensor1, sensor2)
                output = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(output, (tuple, list)):
                    output = output[0]

                y = _labels_to_class_indices(labels)
                num_classes = int(output.shape[1]) if output.ndim >= 2 else 0
                y = _fix_out_of_bounds_targets(y, num_classes)

                loss = torch.nn.CrossEntropyLoss()(output, y)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / max(1, len(val_loader))

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.sensor_epoch}, Average Training Loss: {avg_train_loss}, Average Validation Loss: {avg_val_loss}")
            if writer is not None:
                writer.add_scalar('Loss/Train', avg_train_loss, epoch)
                writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                # If DDP wrapped, save underlying module state_dict
                state_dict = sensor_model.module.state_dict() if hasattr(sensor_model, "module") else sensor_model.state_dict()
                torch.save(state_dict, os.path.join(sensor_model_save_dir, "best_sensor_model.pth"))
                print(f"Saved best sensor model with Validation Loss: {best_loss}")


def train_skeleton_model(rank, args, device, train_loader, val_loader):
    print("Training Skeleton model")
    # Set seed for reproducibility within this process
    torch.manual_seed(args.seed + rank)

    skeleton_model = SkeletonLSTMModel(input_size=48, num_classes=args.num_classes).to(device)

    # Wrap with DDP only if distributed is initialized (CPU-safe: no device_ids)
    if args.world_size > 1 and dist.is_available() and dist.is_initialized():
        skeleton_model = DDP(skeleton_model, find_unused_parameters=True)

    skeleton_optimizer = torch.optim.Adam(
        skeleton_model.parameters(),
        lr=args.skeleton_lr,
        betas=(0.9, 0.98)
    )

    scheduler = torch.optim.lr_scheduler.StepLR(skeleton_optimizer, step_size=args.step_size, gamma=0.1)

    skeleton_model_save_dir = os.path.join(args.output_dir, "skeleton_model")
    ensure_dir(skeleton_model_save_dir, rank)

    writer = SummaryWriter(log_dir=skeleton_model_save_dir) if rank == 0 else None

    best_loss = float('inf')
    best_accuracy = 0.0

    for epoch in range(args.skeleton_epochs):
        skeleton_model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for skeleton_data, _, _, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.skeleton_epochs} (Training)"):
            skeleton_data, labels = skeleton_data.to(device), labels.to(device)
            skeleton_optimizer.zero_grad()

            out = skeleton_model(skeleton_data)
            output = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(output, (tuple, list)):
                output = output[0]

            y = _labels_to_class_indices(labels)
            num_classes = int(output.shape[1]) if output.ndim >= 2 else 0
            y = _fix_out_of_bounds_targets(y, num_classes)

            loss = torch.nn.CrossEntropyLoss()(output, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(skeleton_model.parameters(), max_norm=1.0)
            skeleton_optimizer.step()

            epoch_train_loss += loss.item()

            # Accuracy
            _, predicted = torch.max(output, 1)
            total_train += y.size(0)
            correct_train += (predicted == y).sum().item()

        avg_train_loss = epoch_train_loss / max(1, len(train_loader))
        train_accuracy = (correct_train / max(1, total_train)) * 100

        # Validation phase
        skeleton_model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for skeleton_data, _, _, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.skeleton_epochs} (Validation)"):
                skeleton_data, labels = skeleton_data.to(device), labels.to(device)

                out = skeleton_model(skeleton_data)
                output = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(output, (tuple, list)):
                    output = output[0]

                y = _labels_to_class_indices(labels)
                num_classes = int(output.shape[1]) if output.ndim >= 2 else 0
                y = _fix_out_of_bounds_targets(y, num_classes)

                loss = torch.nn.CrossEntropyLoss()(output, y)
                epoch_val_loss += loss.item()

                _, predicted = torch.max(output, 1)
                total_val += y.size(0)
                correct_val += (predicted == y).sum().item()

        avg_val_loss = epoch_val_loss / max(1, len(val_loader))
        val_accuracy = (correct_val / max(1, total_val)) * 100

        scheduler.step()

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.skeleton_epochs}, "
                  f"Avg Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% "
                  f"Avg Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            if writer is not None:
                writer.add_scalar('Loss/Train', avg_train_loss, epoch)
                writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
                writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
                writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_accuracy = val_accuracy
                state_dict = skeleton_model.module.state_dict() if hasattr(skeleton_model, "module") else skeleton_model.state_dict()
                torch.save(state_dict, os.path.join(skeleton_model_save_dir, "best_skeleton_model.pth"))
                print(f"Saved best skeleton model with Validation Loss: {best_loss:.4f} and Accuracy: {best_accuracy:.2f}%")


def train_diffusion_model(rank, args, device, train_loader, val_loader):
    print("Training Diffusion model")
    torch.manual_seed(args.seed + rank)

    # Load models
    sensor_model = load_sensor_model(args, device)
    diffusion_model = load_diffusion(device)
    skeleton_model = SkeletonLSTMModel(input_size=48, num_classes=args.num_classes).to(device)

    # Enable DDP only if distributed is initialized (CPU-safe: no device_ids)
    if args.world_size > 1 and dist.is_available() and dist.is_initialized():
        sensor_model = DDP(sensor_model, find_unused_parameters=True)
        diffusion_model = DDP(diffusion_model, find_unused_parameters=True)
        skeleton_model = DDP(skeleton_model, find_unused_parameters=True)

    diffusion_optimizer = optim.Adam(
        diffusion_model.parameters(),
        lr=args.diffusion_lr,
        eps=1e-8,
        betas=(0.9, 0.98)
    )
    skeleton_optimizer = optim.Adam(
        skeleton_model.parameters(),
        lr=args.skeleton_lr,
        eps=1e-8,
        betas=(0.9, 0.98)
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        diffusion_optimizer,
        mode='min',
        factor=0.5,
        patience=8,
        verbose=False
    )
    skeleton_scheduler = torch.optim.lr_scheduler.StepLR(
        skeleton_optimizer,
        step_size=args.step_size,
        gamma=0.1
    )

    diffusion_model_save_dir = os.path.join(args.output_dir, "diffusion_model")
    skeleton_model_save_dir = os.path.join(args.output_dir, "skeleton_model")
    ensure_dir(diffusion_model_save_dir, rank)
    ensure_dir(skeleton_model_save_dir, rank)

    writer = SummaryWriter(log_dir=diffusion_model_save_dir) if rank == 0 else None

    best_diffusion_loss = float('inf')
    best_skeleton_loss = float('inf')
    scaling_factor = 0.01

    diffusion_process = DiffusionProcess(
        scheduler=Scheduler(sched_type='cosine', T=args.timesteps, step=1, device=device),
        device=device,
        ddim_scale=args.ddim_scale
    )

    for epoch in range(args.epochs):
        diffusion_model.train()
        sensor_model.train()
        epoch_train_loss = 0.0
        epoch_skeleton_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (skeleton, sensor1, sensor2, mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Training)")):
            skeleton, sensor1, sensor2, mask = skeleton.to(device), sensor1.to(device), sensor2.to(device), mask.to(device)

            t = torch.randint(1, args.timesteps, (skeleton.shape[0],), device=device).long()
            output, context = sensor_model(sensor1, sensor2, return_attn_output=True)

            # Robust mask conversion: one-hot -> class index; else already class index
            mask = _labels_to_class_indices(mask)
            # Avoid out-of-bounds for skeleton classifier
            mask = _fix_out_of_bounds_targets(mask, args.num_classes)

            diffusion_optimizer.zero_grad()
            skeleton_optimizer.zero_grad()

            loss, x0_pred = compute_loss(
                args=args,
                model=diffusion_model,
                x0=skeleton,
                context=context,
                label=mask,
                t=t,
                mask=mask,
                device=device,
                diffusion_process=diffusion_process,
                angular_loss=args.angular_loss,
                epoch=epoch,
                rank=rank
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
            diffusion_optimizer.step()
            epoch_train_loss += loss.item()

            skeleton_model.train()
            skeleton_out = skeleton_model(x0_pred.detach())
            skeleton_output = skeleton_out[0] if isinstance(skeleton_out, (tuple, list)) else skeleton_out
            if isinstance(skeleton_output, (tuple, list)):
                skeleton_output = skeleton_output[0]

            # Ensure mask in-range w.r.t. skeleton_output
            num_classes = int(skeleton_output.shape[1]) if skeleton_output.ndim >= 2 else 0
            mask_fixed = _fix_out_of_bounds_targets(mask, num_classes)

            skeleton_loss = torch.nn.CrossEntropyLoss()(skeleton_output, mask_fixed)
            adjusted_skeleton_loss = scaling_factor * skeleton_loss

            adjusted_skeleton_loss.backward()
            torch.nn.utils.clip_grad_norm_(skeleton_model.parameters(), max_norm=1.0)
            skeleton_optimizer.step()
            epoch_skeleton_loss += adjusted_skeleton_loss.item()

            _, predicted = torch.max(skeleton_output, 1)
            total_train += mask_fixed.size(0)
            correct_train += (predicted == mask_fixed).sum().item()

        avg_train_loss = epoch_train_loss / max(1, len(train_loader))
        avg_skeleton_loss = epoch_skeleton_loss / max(1, len(train_loader))
        train_accuracy = 100 * correct_train / max(1, total_train)

        diffusion_model.eval()
        skeleton_model.eval()
        epoch_val_loss = 0.0
        epoch_skeleton_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_idx, (skeleton, sensor1, sensor2, mask) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Validation)")):
                skeleton, sensor1, sensor2, mask = skeleton.to(device), sensor1.to(device), sensor2.to(device), mask.to(device)

                t = torch.randint(1, args.timesteps, (skeleton.shape[0],), device=device).long()
                output, context = sensor_model(sensor1, sensor2, return_attn_output=True)

                mask = _labels_to_class_indices(mask)
                mask = _fix_out_of_bounds_targets(mask, args.num_classes)

                val_loss, x0_pred_val = compute_loss(
                    args=args,
                    model=diffusion_model,
                    x0=skeleton,
                    context=context,
                    label=mask,
                    t=t,
                    mask=mask,
                    device=device,
                    diffusion_process=diffusion_process,
                    angular_loss=args.angular_loss,
                    epoch=epoch,
                    rank=rank
                )
                epoch_val_loss += val_loss.item()

                sk_out_val = skeleton_model(x0_pred_val.detach())
                skeleton_output_val = sk_out_val[0] if isinstance(sk_out_val, (tuple, list)) else sk_out_val
                if isinstance(skeleton_output_val, (tuple, list)):
                    skeleton_output_val = skeleton_output_val[0]

                num_classes_val = int(skeleton_output_val.shape[1]) if skeleton_output_val.ndim >= 2 else 0
                mask_fixed_val = _fix_out_of_bounds_targets(mask, num_classes_val)

                skeleton_val_loss = torch.nn.CrossEntropyLoss()(skeleton_output_val, mask_fixed_val)
                epoch_skeleton_val_loss += skeleton_val_loss.item()

                _, predicted_val = torch.max(skeleton_output_val, 1)
                total_val += mask_fixed_val.size(0)
                correct_val += (predicted_val == mask_fixed_val).sum().item()

        avg_val_loss = epoch_val_loss / max(1, len(val_loader))
        avg_skeleton_val_loss = epoch_skeleton_val_loss / max(1, len(val_loader))
        val_accuracy = 100 * correct_val / max(1, total_val)

        if avg_val_loss < best_diffusion_loss:
            best_diffusion_loss = avg_val_loss
            if rank == 0:
                state_dict = diffusion_model.module.state_dict() if hasattr(diffusion_model, "module") else diffusion_model.state_dict()
                torch.save(state_dict, os.path.join(diffusion_model_save_dir, "best_diffusion_model.pth"))
                print(f"Saved best diffusion model at {diffusion_model_save_dir}")
            scaling_factor = min(scaling_factor + 0.1, 1.0)

        scheduler.step(avg_val_loss)
        skeleton_scheduler.step(avg_skeleton_val_loss)

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, "
                  f"Avg Diffusion Train Loss: {avg_train_loss:.4f}, Avg Diffusion Val Loss: {avg_val_loss:.4f}, "
                  f"Avg Skeleton Train Loss: {avg_skeleton_loss:.4f}, Avg Skeleton Val Loss: {avg_skeleton_val_loss:.4f}, "
                  f"Skeleton Train Accuracy: {train_accuracy:.2f}, Skeleton Val Accuracy: {val_accuracy:.2f}")

            if writer is not None:
                writer.add_scalar('Loss/Diffusion Train', avg_train_loss, epoch)
                writer.add_scalar('Loss/Diffusion Validation', avg_val_loss, epoch)
                writer.add_scalar('Loss/Skeleton Train', avg_skeleton_loss, epoch)
                writer.add_scalar('Loss/Skeleton Validation', avg_skeleton_val_loss, epoch)
                writer.add_scalar('Accuracy/Skeleton Train', train_accuracy, epoch)
                writer.add_scalar('Accuracy/Skeleton Validation', val_accuracy, epoch)
                writer.add_scalar('Scaling Factor', scaling_factor, epoch)

            # Save periodic checkpoints
            if (rank == 0 and (epoch + 1) % 300 == 0) or ((epoch + 1) == args.epochs):
                diffusion_model_path = os.path.join(diffusion_model_save_dir, f"diffusion_model_epoch_{epoch+1}.pth")
                skeleton_model_path = os.path.join(skeleton_model_save_dir, f"skeleton_model_epoch_{epoch+1}.pth")

                d_state = diffusion_model.module.state_dict() if hasattr(diffusion_model, "module") else diffusion_model.state_dict()
                s_state = skeleton_model.module.state_dict() if hasattr(skeleton_model, "module") else skeleton_model.state_dict()

                torch.save(d_state, diffusion_model_path)
                torch.save(s_state, skeleton_model_path)
                print(f"Saved diffusion model checkpoint at epoch {epoch+1} to {diffusion_model_path}")
                print(f"Saved skeleton model checkpoint at epoch {epoch+1} to {skeleton_model_path}")

            if avg_skeleton_val_loss < best_skeleton_loss:
                best_skeleton_loss = avg_skeleton_val_loss
                s_state = skeleton_model.module.state_dict() if hasattr(skeleton_model, "module") else skeleton_model.state_dict()
                torch.save(s_state, os.path.join(skeleton_model_save_dir, "best_skeleton_model.pth"))
                print(f"Saved best Skeleton model at {skeleton_model_save_dir}")


def main(rank, args):
    setup(rank, args.world_size, seed=42)

    # CPU device (Colab CPU)
    device = torch.device("cpu")

    # Prepare the full dataset
    dataset = prepare_dataset(args)

    # Now you can compute num_classes
    args.num_classes = int(dataset[0][3].shape[-1])
    print("Detected num_classes:", args.num_classes)

    labels = [dataset[i][3] for i in range(len(dataset))]

    if isinstance(labels[0], (list, torch.Tensor, np.ndarray)):
        labels = [int(torch.argmax(torch.tensor(label)).item()) for label in labels]

    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, val_test_idx = next(stratified_split.split(range(len(dataset)), labels))

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_test_idx)

    if args.world_size > 1 and dist.is_available() and dist.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=args.world_size, rank=rank, shuffle=False
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if args.train_skeleton_model:
        train_skeleton_model(rank, args, device, train_loader, val_loader)
    elif args.train_sensor_model:
        train_sensor_model(rank, args, device, train_loader, val_loader)
    else:
        train_diffusion_model(rank, args, device, train_loader, val_loader)

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Training for Diffusion and Sensor Models")
    parser.add_argument('--seed', type=int, default=42, help="seed")

    # Setting up learning rates for the models
    parser.add_argument("--sensor_lr", type=float, default=1e-3, help="Weight decay for sensor regularization")
    parser.add_argument("--skeleton_lr", type=float, default=1e-3, help="Learning rate for training skeleton data")
    parser.add_argument("--diffusion_lr", type=float, default=1e-5, help="Learning rate for training diffusion model")

    # Whether to train the sensor or skeleton model separately
    parser.add_argument("--train_sensor_model", type=eval, choices=[True, False], default=False,
                        help="Set to True to train the sensor model; set to False to train the diffusion model")
    parser.add_argument("--train_skeleton_model", type=eval, choices=[True, False], default=False,
                        help="Set to True to train the skeleton model; Set to False if skeleton model is already trained")

    # Data folders and setting up the parameters for the dataset.py
    parser.add_argument("--overlap", type=int, default=45, help="Overlap for the sliding window dataset")
    parser.add_argument("--window_size", type=int, default=90, help="Window size for the sliding window dataset")
    parser.add_argument("--skeleton_folder", type=str, default="datasets/skeleton", help="Path to the skeleton data folder")
    parser.add_argument("--sensor_folder1", type=str, default="datasets/meta_wrist", help="Path to the first sensor data folder")
    parser.add_argument("--sensor_folder2", type=str, default="datasets/meta_hip", help="Path to the second sensor data folder")

    # Epochs to train the models
    parser.add_argument("--epochs", type=int, default=3000, help="Number of epochs to train the diffusion model")
    parser.add_argument("--sensor_epoch", type=int, default=500, help="Number of epochs to train the sensor model")
    parser.add_argument("--skeleton_epochs", type=int, default=100, help="Number of epochs to train the skeleton model")

    parser.add_argument("--sensor_model_path", type=str, default="./models/sensor_model.pth", help="Path to the pre-trained sensor model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")

    parser.add_argument("--step_size", type=int, default=20, help="Step size for weight decay")
    parser.add_argument("--world_size", type=int, default=8, help="Number of GPUs to use for training")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save the trained model")

    # Timesteps to use for diffusion forward or reverse process
    parser.add_argument("--timesteps", type=int, default=10000, help="Number of timesteps for the diffusion process")
    parser.add_argument('--ddim_scale', type=float, default=0.0, help='Scale factor for DDIM (0 for pure DDIM, 1 for pure DDPM)')

    # Whether to use the Angular loss and Lip Reg. modules as proposed.
    parser.add_argument("--angular_loss", type=eval, choices=[True, False], default=False,
                        help="Whether to use angular loss during training")
    parser.add_argument("--lip_reg", type=eval, choices=[True, False], default=True,
                        help="Flag to determine whether to inlcude LR or not")

    parser.add_argument("--predict_noise", type=eval, choices=[True, False], default=False,
                        help="Flag to determine whether to inlcude LR or not")

    args = parser.parse_args()

    # Colab CPU safety: if CUDA is not available, force world_size=1 via an if/else (no variable renames)
    if not torch.cuda.is_available():
        args.world_size = 1

    # Spawn only if truly using distributed
    if args.world_size == 1:
        main(0, args)
    else:
        mp.spawn(main, args=(args,), nprocs=args.world_size, join=True)
