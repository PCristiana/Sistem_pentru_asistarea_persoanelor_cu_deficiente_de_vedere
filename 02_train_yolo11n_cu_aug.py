from functools import partial
from multiprocessing import freeze_support
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.data.augment import Albumentations as UltralyticsAlbumentations
from ultralytics.models.yolo.detect.train import DetectionTrainer

from config_experimente import (
    DATA_YAML,
    PROJECT_DIR,
    EPOCHS,
    IMGSZ,
    BATCH,
    DEVICE,
    WORKERS,
    SEED,
    NO_AUGMENTATION_ARGS,
)

RUN_NAME = "yolo11n_cu_aug"
MODEL_START = "yolo11n.pt"


class CustomAlbumentations(UltralyticsAlbumentations):
    def __init__(self, transform, contains_spatial=True, p=1.0):
        super().__init__(p)
        self.transform = transform

        if hasattr(self.transform, "set_random_seed"):
            self.transform.set_random_seed(torch.initial_seed() % (2**32 - 1))

        self.contains_spatial = contains_spatial

    def __call__(self, labels):
        labels = super().__call__(labels)

        if "cls" in labels:
            labels["cls"] = labels["cls"].reshape(-1, 1)

        return labels

    def __repr__(self):
        return str(self.transform)


class CustomTrainer(DetectionTrainer):
    def __init__(self, custom_albumentations_transforms, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_albumentations_transforms = custom_albumentations_transforms
        self.replacement_logged = False

    def _close_dataloader_mosaic(self):
        super()._close_dataloader_mosaic()
        self.__replace_albumentations(self.train_loader.dataset)

    def __replace_albumentations(self, dataset):
        transforms = dataset.transforms.tolist()

        for i, transform in enumerate(transforms):
            if isinstance(transform, UltralyticsAlbumentations):
                if not self.replacement_logged:
                    print("\n" + "=" * 80)
                    print("INLOCUIESC AUGMENTARILE DEFAULT ULTRALYTICS CU ALBUMENTATIONS CUSTOM")
                    print("Augmentari custom folosite:")
                    print(self.custom_albumentations_transforms)
                    print("=" * 80 + "\n")
                    self.replacement_logged = True

                transforms[i] = self.custom_albumentations_transforms

    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode=mode, batch=batch)

        # Important: augmentarile se aplica doar pe train.
        # Validarea ramane pe imagini diferite si neaugmentate.
        if mode == "train":
            self.__replace_albumentations(dataset)

        return dataset


def build_albumentations_transform():
    return A.Compose(
        [
            # Intoarce imaginea pe orizontala
            A.HorizontalFlip(p=0.5),

            # Modifica luminozitatea si contrastul
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.6,
            ),

            # Modifica usor culorile imaginii
            A.HueSaturationValue(
                hue_shift_limit=8,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.4,
            ),

            # Modifica luminozitatea intr-un mod mai natural
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.3,
            ),

            # Imbunatateste contrastul local
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=0.2,
            ),

            # Alege aleator un tip de blur
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.GaussianBlur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ],
                p=0.25,
            ),

            # Adauga zgomot in imagine
            A.GaussNoise(
                std_range=(0.02, 0.06),
                p=0.2,
            ),

            # Simuleaza imagine comprimata / calitate mai slaba
            A.ImageCompression(
                quality_range=(60, 95),
                p=0.2,
            ),

            # Adauga umbre artificiale
            A.RandomShadow(
                shadow_roi=(0, 0.3, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                p=0.2,
            ),

            # Simuleaza ploaie usoara
            A.RandomRain(
                slant_range=(-5, 5),
                drop_length=12,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=3,
                brightness_coefficient=0.9,
                rain_type="drizzle",
                p=0.15,
            ),

            # Acopera mici zone din imagine
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.03, 0.10),
                hole_width_range=(0.03, 0.10),
                fill=0,
                p=0.2,
            ),

            # Modificari geometrice usoare
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent=(-0.05, 0.05),
                rotate=(-5, 5),
                p=0.5,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.25,
            clip=True,
            filter_invalid_bboxes=True,
        ),
    )


def save_losses_after_each_epoch(trainer):
    results_csv = Path(trainer.csv)

    if not results_csv.exists():
        return

    df = pd.read_csv(results_csv)
    df.columns = [col.strip() for col in df.columns]

    loss_columns = [col for col in df.columns if "loss" in col.lower()]
    if len(loss_columns) == 0:
        return

    columns_to_save = []
    if "epoch" in df.columns:
        columns_to_save.append("epoch")
    if "time" in df.columns:
        columns_to_save.append("time")

    columns_to_save.extend(loss_columns)

    loss_df = df[columns_to_save]
    output_csv = Path(trainer.save_dir) / "loss_dupa_epoch.csv"
    loss_df.to_csv(output_csv, index=False)


def plot_losses_after_training(train_dir):
    loss_csv = train_dir / "loss_dupa_epoch.csv"

    if not loss_csv.exists():
        print("Nu exista loss_dupa_epoch.csv.")
        return

    df = pd.read_csv(loss_csv)
    df.columns = [col.strip() for col in df.columns]

    if "epoch" not in df.columns:
        print("Nu exista coloana epoch in loss_dupa_epoch.csv.")
        return

    loss_columns = [col for col in df.columns if "loss" in col.lower()]
    if len(loss_columns) == 0:
        print("Nu exista coloane de loss in loss_dupa_epoch.csv.")
        return

    for col in loss_columns:
        plt.plot(df["epoch"], df[col], label=col)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss dupa fiecare epoch")
    plt.legend()
    plt.grid(True)

    output_png = train_dir / "loss_dupa_epoch.png"
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Grafic loss salvat in: {output_png}")


def main():
    print("Antrenez YOLO11n cu Albumentations on-the-fly")
    print(f"Dataset YAML: {DATA_YAML}")

    transform = build_albumentations_transform()
    custom_albumentations = CustomAlbumentations(
        transform=transform,
        contains_spatial=True,
    )

    model = YOLO(MODEL_START)
    model.add_callback("on_fit_epoch_end", save_losses_after_each_epoch)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        plots=True,
        seed=SEED,
        deterministic=True,
        trainer=partial(
            CustomTrainer,
            custom_albumentations_transforms=custom_albumentations,
        ),
        # Oprim augmentarile default YOLO, ca sa comparam clar:
        # fara_aug = fara augmentari, cu_aug = doar Albumentations custom.
        **NO_AUGMENTATION_ARGS,
    )

    train_dir = Path(model.trainer.save_dir)
    save_losses_after_each_epoch(model.trainer)
    plot_losses_after_training(train_dir)

    print("\nAntrenarea s-a terminat.")
    print(f"Folder rezultate: {train_dir}")
    print(f"Best model: {train_dir / 'weights' / 'best.pt'}")
    print(f"Last model: {train_dir / 'weights' / 'last.pt'}")
    print(f"CSV complet Ultralytics: {train_dir / 'results.csv'}")
    print(f"Loss separat: {train_dir / 'loss_dupa_epoch.csv'}")
    print(f"Grafic loss: {train_dir / 'loss_dupa_epoch.png'}")


if __name__ == "__main__":
    freeze_support()
    main()
