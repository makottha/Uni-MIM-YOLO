import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from ultralytics.nn.tasks import DetectionModel
import sys
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as T
import time

import models  # Triggers registration

# Import custom modules
from models.mim_modules import MIMAdapter, MIMHead
from utils.mim_utils import MIMDataset, generate_mask

# --- HYPERPARAMETERS ---
EPOCHS = 100
BATCH_SIZE = 32
IMG_SIZE = 640
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data Path
DATA_LIST = "data/coco_splits/unlabeled.txt"
YAML_CFG = "models/yolo_mim.yaml"


# --- LOGGING SETUP ---
def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "train_log.txt")

    # Simple logger that writes to file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def main():
    # 1. Setup Directories
    SAVE_DIR = "runs/mim_experiment_1"
    WEIGHTS_DIR = os.path.join(SAVE_DIR, "weights")
    VIS_DIR = os.path.join(SAVE_DIR, "visuals")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    setup_logger(SAVE_DIR)
    logging.info(f"🚀 Starting MIM Pre-training on {DEVICE} with Mixed Precision...")
    logging.info(f"   Configs: Epochs={EPOCHS}, Batch={BATCH_SIZE}, LR={LR}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 2. Data & Model
    dataset = MIMDataset(DATA_LIST, img_size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True)

    import models.mim_modules
    sys.modules['ultralytics.nn.modules.mim_modules'] = models.mim_modules

    try:
        model = DetectionModel(YAML_CFG).to(DEVICE)
    except Exception as e:
        logging.error(f"Model load failed: {e}")
        raise e

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    # 3. Training Loop
    best_loss = float('inf')
    start_time = time.time()

    # Create CSV Header for easy plotting later
    with open(os.path.join(SAVE_DIR, "metrics.csv"), "w") as f:
        f.write("epoch,loss,lr\n")

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        epoch_loss = 0

        for imgs in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)

            mask = generate_mask(imgs.size(0), imgs.size(1), IMG_SIZE, IMG_SIZE, patch_size=32, mask_ratio=0.6)
            masked_imgs = imgs * mask

            with torch.cuda.amp.autocast():
                reconstruction = model(masked_imgs)
                if isinstance(reconstruction, list): reconstruction = reconstruction[0]

                reconstruction = torch.nn.functional.interpolate(
                    reconstruction, size=imgs.shape[-2:], mode='bilinear', align_corners=False
                )

                loss = (reconstruction - imgs) ** 2
                loss = loss.mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # End of Epoch
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']

        # LOGGING
        logging.info(f"Epoch {epoch + 1} Done. Avg Loss: {avg_loss:.6f}")

        # Write to CSV
        with open(os.path.join(SAVE_DIR, "metrics.csv"), "a") as f:
            f.write(f"{epoch + 1},{avg_loss:.6f},{current_lr:.8f}\n")

        # SAVE CHECKPOINTS
        # 1. Save Last (Always)
        torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "last.pt"))

        # 2. Save Best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "best.pt"))
            logging.info("   ---> New Best Model Saved.")

        # 3. Save Every Epoch (Your request)
        # Note: YOLOv8n is small (~6MB). 100 epochs = 600MB. This is safe.
        torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, f"epoch_{epoch + 1}.pt"))

        # VISUALIZATION (Every 5 epochs)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            save_visuals(imgs, masked_imgs, reconstruction, epoch + 1, VIS_DIR)

    total_time = (time.time() - start_time) / 3600
    logging.info(f"✅ Training Finished in {total_time:.2f} hours.")


def save_visuals(orig, masked, rec, epoch, save_dir):
    inv_normalize = T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    # Detach and move to CPU to avoid VRAM leaks
    orig = orig.detach().float().cpu()
    masked = masked.detach().float().cpu()
    rec = rec.detach().float().cpu()

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        if i >= orig.size(0): break

        axes[0, i].imshow(inv_normalize(orig[i]).permute(1, 2, 0).clamp(0, 1))
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")

        axes[1, i].imshow(inv_normalize(masked[i]).permute(1, 2, 0).clamp(0, 1))
        axes[1, i].axis('off')
        axes[1, i].set_title("Masked")

        axes[2, i].imshow(inv_normalize(rec[i]).permute(1, 2, 0).clamp(0, 1))
        axes[2, i].axis('off')
        axes[2, i].set_title("Reconstructed")

    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}.png"))
    plt.close()


if __name__ == "__main__":
    main()