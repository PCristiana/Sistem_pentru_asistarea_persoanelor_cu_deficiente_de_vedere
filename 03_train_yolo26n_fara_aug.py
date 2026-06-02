from multiprocessing import freeze_support
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

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

RUN_NAME = "yolo26n_fara_aug"
MODEL_START = "yolo26n.pt"


def save_losses_after_each_epoch(trainer):
    """
    Ultralytics salveaza automat results.csv.
    Functia extrage coloanele de loss si le salveaza separat in loss_dupa_epoch.csv.
    """
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
    print("Antrenez YOLO26n fara augmentari")
    print(f"Dataset YAML: {DATA_YAML}")

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
