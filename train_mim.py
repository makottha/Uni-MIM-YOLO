import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import intersect_dicts
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

import models  # This triggers the registration

# Import our custom modules
from models.mim_modules import MIMAdapter, MIMHead
from utils.mim_utils import MIMDataset, generate_mask

# --- HYPERPARAMETERS ---
EPOCHS = 100
BATCH_SIZE = 16  # Adjust based on your GPU VRAM
IMG_SIZE = 640
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data Path (Use the unlabeled/train list we created)
DATA_LIST = "data/coco_splits/unlabeled.txt"
YAML_CFG = "models/yolo_mim.yaml"


def main():
    print(f"🚀 Starting MIM Pre-training on {DEVICE}...")

    # 1. Initialize Dataset
    dataset = MIMDataset(DATA_LIST, img_size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 2. Initialize Model
    # We use Ultralytics DetectionModel to parse the YAML
    # IMPORTANT: We pass our custom modules to the safe_globals or manual injection if needed,
    # but since we are running a raw PyTorch loop, we just need to ensure the class is visible.

    # Hack: Inject modules into the global namespace so Ultralytics YAML parser finds them
    import models.mim_modules
    sys.modules['ultralytics.nn.modules.mim_modules'] = models.mim_modules

    # Load Model from YAML
    # Note: Ultralytics might throw a warning about 'MIMAdapter', but it will instantiate if imported.
    # If this fails, we construct the backbone manually. Let's try the standard way first.
    try:
        model = DetectionModel(YAML_CFG).to(DEVICE)
    except Exception as e:
        print("Standard loading failed. Attempting manual registration...")
        # (If this block fails, we will need to register the module in ultralytics source,
        # but usually direct import works if the class name matches).
        raise e

    # 3. Optimizer (AdamW is standard for MIM/MAE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 4. Training Loop
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        epoch_loss = 0

        for imgs in pbar:
            imgs = imgs.to(DEVICE)

            # A. Generate Mask
            mask = generate_mask(imgs.size(0), imgs.size(1), IMG_SIZE, IMG_SIZE, patch_size=32, mask_ratio=0.6)

            # B. Apply Mask (Input = Image * Mask)
            # We replace masked pixels with mean (0) or noise. Here we zero them.
            masked_imgs = imgs * mask

            # C. Forward Pass
            reconstruction = model(masked_imgs)

            if isinstance(reconstruction, list):
                reconstruction = reconstruction[0]

            # --- FIX: UPSAMPLE OUTPUT TO MATCH INPUT SIZE (160 -> 640) ---
            # We use bilinear interpolation to stretch the 160x160 output to 640x640
            reconstruction = torch.nn.functional.interpolate(
                reconstruction,
                size=imgs.shape[-2:],  # Target size (640, 640)
                mode='bilinear',
                align_corners=False
            )
            # -------------------------------------------------------------

            # D. Loss Calculation (MSE)
            loss = (reconstruction - imgs) ** 2
            loss = loss.mean()
            loss = loss.mean()  # Simple global MSE for stability first
            # Advanced: loss = (loss * (1 - mask)).sum() / (1 - mask).sum()

            # E. Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # Scheduler Step
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_loss:.6f}")

        # 5. Save Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_mim_weights.pt")
            print("💾 Saved Best Model.")

            # Save a visualization every 5 epochs
            if epoch % 5 == 0:
                save_visuals(imgs, masked_imgs, reconstruction, epoch)


def save_visuals(orig, masked, rec, epoch):
    """
    Saves a comparison image to disk for the paper.
    """
    os.makedirs("runs/mim_vis", exist_ok=True)

    # Denormalize for visualization
    inv_normalize = T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        if i >= orig.size(0): break

        # Original
        o = inv_normalize(orig[i]).cpu().permute(1, 2, 0).clamp(0, 1)
        axes[0, i].imshow(o)
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

        # Masked
        m = inv_normalize(masked[i]).cpu().permute(1, 2, 0).clamp(0, 1)
        axes[1, i].imshow(m)
        axes[1, i].set_title("Masked Input")
        axes[1, i].axis('off')

        # Reconstruction
        r = inv_normalize(rec[i]).cpu().permute(1, 2, 0).clamp(0, 1)
        axes[2, i].imshow(r)
        axes[2, i].set_title("Reconstruction")
        axes[2, i].axis('off')

    plt.savefig(f"runs/mim_vis/epoch_{epoch}.png")
    plt.close()


if __name__ == "__main__":
    main()