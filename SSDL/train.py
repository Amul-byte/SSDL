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

# Assumes these exist in your diffusion_model package
from diffusion_model.diffusion import DiffusionProcess, Scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusion_model.model_loader import load_sensor_model, load_diffusion
from diffusion_model.skeleton_model import SkeletonLSTMModel
from diffusion_model.util import (
    prepare_dataset,
    compute_loss,
)

def ensure_dir(path, rank):
    if rank == 0:
        if os.path.exists(path):
            try:
                # safe remove
                shutil.rmtree(path)
            except:
                pass
        os.makedirs(path, exist_ok=True)
    try:
        dist.barrier()
    except:
        pass

def setup(rank, world_size, seed):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize Process Group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Set the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cleanup():
    dist.destroy_process_group()

def train_sensor_model(rank, args, device, train_loader, val_loader):
    if rank == 0: print("Training Sensor model")
    
    torch.manual_seed(args.seed + rank)
    
    # LOAD SENSOR MODEL
    sensor_model = load_sensor_model(args, device)
    sensor_model = DDP(sensor_model, device_ids=[rank], find_unused_parameters=True)

    sensor_optimizer = torch.optim.Adam(
        sensor_model.parameters(),
        lr=args.sensor_lr,
        betas=(0.9, 0.98)
    )

    sensor_model_save_dir = os.path.join(args.output_dir, "sensor_model")
    ensure_dir(sensor_model_save_dir, rank)

    sensor_log_dir = os.path.join(sensor_model_save_dir, "sensor_logs")
    if rank == 0:
        writer = SummaryWriter(log_dir=sensor_log_dir)

    best_loss = float('inf')

    for epoch in range(args.sensor_epoch):
        sensor_model.train()
        epoch_train_loss = 0.0

        for _, sensor1, sensor2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.sensor_epoch} (Training)", disable=(rank!=0)):
            sensor1, sensor2, labels = sensor1.to(device), sensor2.to(device), labels.to(device)
            sensor_optimizer.zero_grad()
            output = sensor_model(sensor1, sensor2)
            loss = torch.nn.CrossEntropyLoss()(output, labels.argmax(dim=1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(sensor_model.parameters(), max_norm=1.0)
            sensor_optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)

        # Validation phase
        sensor_model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for _, sensor1, sensor2, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.sensor_epoch} (Validation)", disable=(rank!=0)):
                sensor1, sensor2, labels = sensor1.to(device), sensor2.to(device), labels.to(device)
                output = sensor_model(sensor1, sensor2)
                loss = torch.nn.CrossEntropyLoss()(output, labels.argmax(dim=1))
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.sensor_epoch}, Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(sensor_model.state_dict(), os.path.join(sensor_model_save_dir, "best_sensor_model.pth"))
                print(f"Saved best sensor model with Validation Loss: {best_loss}")

def train_skeleton_model(rank, args, device, train_loader, val_loader):
    if rank == 0: print("Training Skeleton model")
    
    torch.manual_seed(args.seed + rank)
    
    # FIX: Use args.num_classes and args.num_joints
    input_dim = args.num_joints * 3 # 16*3 = 48
    skeleton_model = SkeletonLSTMModel(input_size=input_dim, num_classes=args.num_classes).to(device)
    skeleton_model = DDP(skeleton_model, device_ids=[rank], find_unused_parameters=True)

    skeleton_optimizer = torch.optim.Adam(
        skeleton_model.parameters(),
        lr=args.skeleton_lr,
        betas=(0.9, 0.98)
    )

    scheduler = torch.optim.lr_scheduler.StepLR(skeleton_optimizer, step_size=args.step_size, gamma=0.1)
    skeleton_model_save_dir = os.path.join(args.output_dir, "skeleton_model")
    ensure_dir(skeleton_model_save_dir, rank)

    if rank == 0:
        writer = SummaryWriter(log_dir=skeleton_model_save_dir)

    best_loss = float('inf')

    for epoch in range(args.skeleton_epochs):
        skeleton_model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for skeleton_data, _, _, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.skeleton_epochs} (Training)", disable=(rank!=0)):
            skeleton_data, labels = skeleton_data.to(device), labels.to(device)
            skeleton_optimizer.zero_grad()
            output = skeleton_model(skeleton_data)
            loss = torch.nn.CrossEntropyLoss()(output, labels.argmax(dim=1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(skeleton_model.parameters(), max_norm=1.0)
            skeleton_optimizer.step()

            epoch_train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels.argmax(dim=1)).sum().item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train * 100

        # Validation phase
        skeleton_model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for skeleton_data, _, _, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.skeleton_epochs} (Validation)", disable=(rank!=0)):
                skeleton_data, labels = skeleton_data.to(device), labels.to(device)
                output = skeleton_model(skeleton_data)
                loss = torch.nn.CrossEntropyLoss()(output, labels.argmax(dim=1))
                epoch_val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels.argmax(dim=1)).sum().item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val * 100
        scheduler.step()

        if rank == 0:
            print(f"Epoch {epoch+1}: Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
            writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(skeleton_model.state_dict(), os.path.join(skeleton_model_save_dir, "best_skeleton_model.pth"))

def train_diffusion_model(rank, args, device, train_loader, val_loader):
    if rank == 0: print("Training Diffusion model")
    torch.manual_seed(args.seed + rank)

    # 1. Load Sensor Model
    sensor_model = load_sensor_model(args, device)
    
    # 2. Load Diffusion Model
    # Note: load_diffusion in model_loader.py MUST support num_joints/num_classes args
    # If it doesn't, we need to fix model_loader.py next.
    diffusion_model = load_diffusion(device) 
    
    # 3. Load Skeleton Model (Evaluator)
    input_dim = args.num_joints * 3
    skeleton_model = SkeletonLSTMModel(input_size=input_dim, num_classes=args.num_classes).to(device)

    # DDP Wrappers
    sensor_model = DDP(sensor_model, device_ids=[rank], find_unused_parameters=True)
    diffusion_model = DDP(diffusion_model, device_ids=[rank], find_unused_parameters=True)
    skeleton_model = DDP(skeleton_model, device_ids=[rank], find_unused_parameters=True)

    diffusion_optimizer = optim.Adam(diffusion_model.parameters(), lr=args.diffusion_lr, eps=1e-8, betas=(0.9, 0.98))
    skeleton_optimizer = optim.Adam(skeleton_model.parameters(), lr=args.skeleton_lr, eps=1e-8, betas=(0.9, 0.98))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(diffusion_optimizer, mode='min', factor=0.5, patience=8)
    
    # Directories
    diffusion_model_save_dir = os.path.join(args.output_dir, "diffusion_model")
    ensure_dir(diffusion_model_save_dir, rank)
    
    if rank == 0:
        writer = SummaryWriter(log_dir=diffusion_model_save_dir)

    best_diffusion_loss = float('inf')
    
    # Initialize Diffusion Process
    diffusion_process = DiffusionProcess(
        scheduler=Scheduler(sched_type='cosine', T=args.timesteps, step=1, device=device),
        device=device,
        ddim_scale=args.ddim_scale
    )

    for epoch in range(args.epochs):
        diffusion_model.train()
        sensor_model.train()
        epoch_train_loss = 0.0

        for batch_idx, (skeleton, sensor1, sensor2, mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} (Train)", disable=(rank!=0))):
            skeleton, sensor1, sensor2, mask = skeleton.to(device), sensor1.to(device), sensor2.to(device), mask.to(device)
            t = torch.randint(1, args.timesteps, (skeleton.shape[0],), device=device).long()
            
            # Get Sensor Condition
            output, context = sensor_model(sensor1, sensor2, return_attn_output=True)
            
            # Label
            mask_label = mask.argmax(dim=1)
            
            diffusion_optimizer.zero_grad()
            
            # Compute Loss (Using util.py)
            loss, x0_pred = compute_loss(
                args=args,
                model=diffusion_model,
                x0=skeleton,
                context=context,
                label=mask_label,
                t=t,
                mask=mask_label,
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

        avg_train_loss = epoch_train_loss / len(train_loader)

        # Validation (Simplified for brevity)
        diffusion_model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (skeleton, sensor1, sensor2, mask) in enumerate(val_loader):
                skeleton, sensor1, sensor2, mask = skeleton.to(device), sensor1.to(device), sensor2.to(device), mask.to(device)
                t = torch.randint(1, args.timesteps, (skeleton.shape[0],), device=device).long()
                output, context = sensor_model(sensor1, sensor2, return_attn_output=True)
                mask_label = mask.argmax(dim=1)

                val_loss, _ = compute_loss(
                    args=args, model=diffusion_model, x0=skeleton, context=context,
                    label=mask_label, t=t, mask=mask_label, device=device,
                    diffusion_process=diffusion_process, angular_loss=args.angular_loss,
                    epoch=epoch, rank=rank
                )
                epoch_val_loss += val_loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)

        if rank == 0:
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
            
            if avg_val_loss < best_diffusion_loss:
                best_diffusion_loss = avg_val_loss
                torch.save(diffusion_model.state_dict(), os.path.join(diffusion_model_save_dir, "best_diffusion_model.pth"))

def main(rank, args):
    setup(rank, args.world_size, seed=42)
    device = torch.device(f'cuda:{rank}')

    # Prepare dataset
    dataset = prepare_dataset(args)
    
    # Stratified Split
    labels = [dataset[i][3] for i in range(len(dataset))]
    if isinstance(labels[0], (list, torch.Tensor, np.ndarray)):
        labels = [torch.argmax(torch.tensor(label)).item() for label in labels]

    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, val_test_idx = next(stratified_split.split(range(len(dataset)), labels))

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_test_idx)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=args.world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

    if args.train_skeleton_model:
        train_skeleton_model(rank, args, device, train_loader, val_loader)
    elif args.train_sensor_model:
        train_sensor_model(rank, args, device, train_loader, val_loader)
    else:
        train_diffusion_model(rank, args, device, train_loader, val_loader)

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--sensor_lr", type=float, default=1e-3)
    parser.add_argument("--skeleton_lr", type=float, default=1e-3)
    parser.add_argument("--diffusion_lr", type=float, default=1e-5)
    
    # Modes
    parser.add_argument("--train_sensor_model", type=eval, choices=[True, False], default=False)
    parser.add_argument("--train_skeleton_model", type=eval, choices=[True, False], default=False)
    
    # Dataset Config (UPDATED DEFAULTS)
    parser.add_argument("--overlap", type=int, default=24, help="Overlap")
    parser.add_argument("--window_size", type=int, default=48, help="MUST BE 48 for UNet compatibility")
    parser.add_argument("--num_joints", type=int, default=16, help="16 for SmartFall")
    parser.add_argument("--num_classes", type=int, default=14, help="14 for SmartFall")
    
    # Paths
    parser.add_argument("--sensor_folder1", type=str, default="/home/qsw26/smartfall/SSDL_drive/labelled_data/metahip")
    parser.add_argument("--sensor_folder2", type=str, default="/home/qsw26/smartfall/SSDL_drive/labelled_data/metahip")
    parser.add_argument("--skeleton_folder", type=str, default="/home/qsw26/smartfall/SSDL_drive/datasets/skeleton")

    # Training Config
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--sensor_epoch", type=int, default=500)
    parser.add_argument("--skeleton_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--step_size", type=int, default=20)
    
    # Distributed Config (UPDATED SAFE DEFAULT)
    parser.add_argument("--world_size", type=int, default=1, help="Set to 1 for single GPU")
    parser.add_argument("--output_dir", type=str, default="./results")

    # Diffusion Config
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument('--ddim_scale', type=float, default=0.0)
    parser.add_argument("--angular_loss", type=eval, choices=[True, False], default=True)
    parser.add_argument("--lip_reg", type=eval, choices=[True, False], default=True)
    parser.add_argument("--predict_noise", type=eval, choices=[True, False], default=False)
    
    args = parser.parse_args()
    
    # Safe Spawn
    mp.spawn(main, args=(args,), nprocs=args.world_size, join=True)