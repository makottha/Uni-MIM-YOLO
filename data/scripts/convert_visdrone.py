import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# MAPPING: VisDrone has 10 classes + ignored regions
# We map them to 0-9. (Ignored regions usually class 0 or 11, we skip them)
# VisDrone Classes: 1:pedestrian, 2:people, 3:bicycle, 4:car, 5:van,
# 6:truck, 7:tricycle, 8:awning-tricycle, 9:bus, 10:motor
VISDRONE_TO_YOLO_MAP = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
    6: 5, 7: 6, 8: 7, 9: 8, 10: 9
}


def convert_visdrone(root_dir, split='train'):
    img_dir = Path(root_dir) / f'VisDrone2019-DET-{split}/images'
    ann_dir = Path(root_dir) / f'VisDrone2019-DET-{split}/annotations'
    out_dir = Path(root_dir) / f'labels_yolo/{split}'

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting VisDrone {split}...")

    for ann_file in tqdm(list(ann_dir.glob('*.txt'))):
        img_file = img_dir / ann_file.with_suffix('.jpg').name

        # We need image size to normalize
        try:
            with Image.open(img_file) as img:
                w_img, h_img = img.size
        except FileNotFoundError:
            continue  # Skip if image missing

        lines = []
        with open(ann_file, 'r') as f:
            for line in f:
                data = line.strip().split(',')
                if len(data) < 6: continue

                # Parse VisDrone format
                x_min, y_min, w_box, h_box = map(int, data[:4])
                score = int(data[4])
                cls_id = int(data[5])

                # Filter: Keep only valid classes
                if cls_id in VISDRONE_TO_YOLO_MAP:
                    yolo_cls = VISDRONE_TO_YOLO_MAP[cls_id]

                    # Normalize xywh
                    x_c = (x_min + w_box / 2) / w_img
                    y_c = (y_min + h_box / 2) / h_img
                    w_norm = w_box / w_img
                    h_norm = h_box / h_img

                    # Clamp to [0,1]
                    x_c = max(0, min(1, x_c))
                    y_c = max(0, min(1, y_c))
                    w_norm = max(0, min(1, w_norm))
                    h_norm = max(0, min(1, h_norm))

                    lines.append(f"{yolo_cls} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        # Write YOLO label file
        if lines:
            with open(out_dir / ann_file.name, 'w') as f:
                f.writelines(lines)


if __name__ == "__main__":
    # Robust Pathing:
    # 1. Get the folder where this script lives (data/scripts)
    # 2. Go up one level to 'data'
    # 3. Go into 'visdrone'

    current_script_path = Path(__file__).resolve()
    data_dir = current_script_path.parent.parent  # Points to Uni-MIM-YOLO/data
    VIS_ROOT = data_dir / "visdrone"

    print(f"Targeting VisDrone Data at: {VIS_ROOT}")

    if not VIS_ROOT.exists():
        print(f"❌ Error: Could not find {VIS_ROOT}")
        print("Please check your folder structure.")
    else:
        convert_visdrone(VIS_ROOT, 'train')
        convert_visdrone(VIS_ROOT, 'val')
        print("✅ Conversion Complete.")