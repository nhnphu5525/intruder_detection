from ultralytics import YOLO
from pathlib import Path
import yaml
import random
import shutil
from IPython.display import display, Markdown

DATASET_PATH = Path(r"D:\NX Hackathon\Dataset\Human Detection")  # Change this to your dataset path
IMAGE_DIR = DATASET_PATH / "images"
LABEL_DIR = DATASET_PATH / "labels"
CONFIG_FILE = DATASET_PATH / "human_dataset.yaml"
WEIGHTS_PATH = "yolo11m.pt"  # Pretrained weights
EPOCHS = 10
IMG_SIZE = 640
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def split_dataset():
    train_img_dir = IMAGE_DIR / "train"
    val_img_dir = IMAGE_DIR / "val"
    test_img_dir = IMAGE_DIR / "test"
    train_lbl_dir = LABEL_DIR / "train"
    val_lbl_dir = LABEL_DIR / "val"
    test_lbl_dir = LABEL_DIR / "test"
    
    for folder in [train_img_dir, val_img_dir, test_img_dir, train_lbl_dir, val_lbl_dir, test_lbl_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    
    image_files = list(IMAGE_DIR.glob("*.jpg"))
    random.shuffle(image_files)
    
    train_split = int(len(image_files) * TRAIN_RATIO)
    val_split = max(1, int(len(image_files) * VAL_RATIO))  # Ensure at least 1 image in val
    test_split = max(1, len(image_files) - train_split - val_split)  # Ensure at least 1 image in test
    
    train_files = image_files[:train_split]
    val_files = image_files[train_split:train_split + val_split]
    test_files = image_files[train_split + val_split:]
    
    for img_path in train_files:
        shutil.move(img_path, train_img_dir / img_path.name)
        shutil.move(LABEL_DIR / f"{img_path.stem}.txt", train_lbl_dir / f"{img_path.stem}.txt")
    
    for img_path in val_files:
        shutil.move(img_path, val_img_dir / img_path.name)
        shutil.move(LABEL_DIR / f"{img_path.stem}.txt", val_lbl_dir / f"{img_path.stem}.txt")
    
    for img_path in test_files:
        shutil.move(img_path, test_img_dir / img_path.name)
        shutil.move(LABEL_DIR / f"{img_path.stem}.txt", test_lbl_dir / f"{img_path.stem}.txt")
    
    display(Markdown("✅ **Dataset split into train/val/test folders. Validation and test sets ensured to have at least one image.**"))

def create_dataset_config():
    dataset_config = {
        "path": str(DATASET_PATH),
        "train": str(IMAGE_DIR / "train"),
        "val": str(IMAGE_DIR / "val"),
        "test": str(IMAGE_DIR / "test"),
        "nc": 1,  # Number of classes (human detection)
        "names": ["human"]
    }
    
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    display(Markdown(f"✅ **Dataset config file created at:** `{CONFIG_FILE}`"))

def download_model():
    if not Path(WEIGHTS_PATH).exists():
        display(Markdown("🔄 **Downloading YOLOv11 pretrained model...**"))
        model = YOLO("yolo11m.pt")  # Automatically downloads if not present
        display(Markdown("✅ **Model downloaded!**"))
    else:
        display(Markdown("✅ **Pretrained model already exists. Skipping download.**"))

def train_yolov11():
    display(Markdown("🚀 **Starting YOLOv11 training...**"))
    model = YOLO(WEIGHTS_PATH)
    results = model.train(data=CONFIG_FILE, epochs=EPOCHS, imgsz=IMG_SIZE)
    display(Markdown("✅ **Training completed!**"))

split_dataset()
download_model()
create_dataset_config()
train_yolov11()


