from pathlib import Path

# ============================================================
# MODIFICA DOAR AICI, DACA LA TINE CAILE SUNT DIFERITE
# ============================================================

# Datasetul tau vechi, asa cum il ai acum, cu images/train, images/val, labels/train, labels/val
SOURCE_DATASET = Path(
    r"C:\Users\Cristiana\OneDrive - Technical University of Cluj-Napoca\LICENTA\coco_selectat\coco_blind_subset"
)

# Datasetul nou, care va fi creat automat prin COPIERE, nu prin mutare.
# Aici vor aparea train/val/test.
SPLIT_DATASET = Path(
    r"C:\Users\Cristiana\OneDrive - Technical University of Cluj-Napoca\LICENTA\coco_selectat\coco_blind_subset_split_70_20_10"
)

# YAML-ul nou, creat automat de scriptul 00_imparte_dataset_70_20_10.py
DATA_YAML = SPLIT_DATASET / "coco_blind_split.yaml"

# Folderul unde se vor salva antrenarile noi.
# Nu il pune peste folderul vechi, ca sa nu incurci rezultatele.
PROJECT_DIR = Path(
    r"C:\Users\Cristiana\Desktop\LICENTA\runs\detect_split_70_20_10"
)

CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
    "bench",
    "fire hydrant",
    "parking meter",
    "dog",
]

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

TRAIN_PERCENT = 0.70
VAL_PERCENT = 0.20
# TEST_PERCENT este ce ramane: 0.10

# Setari comune pentru antrenare
EPOCHS = 30
IMGSZ = 640
BATCH = 9
DEVICE = 0
WORKERS = 0
SEED = 42

# Aceste setari dezactiveaza augmentarile default YOLO.
# Le folosim la variantele "fara_aug", ca sa fie chiar fara augmentari.
NO_AUGMENTATION_ARGS = dict(
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.0,
    close_mosaic=0,
)
