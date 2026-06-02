from multiprocessing import freeze_support
from pathlib import Path
import csv

from ultralytics import YOLO

from config_experimente import DATA_YAML, PROJECT_DIR, IMGSZ, BATCH, DEVICE, WORKERS

SPLIT = "val"
OUTPUT_FOLDER_NAME = "metrici_validare"
SUMMARY_CSV_NAME = "rezumat_metrici_validare.csv"

EXPERIMENTS = [
    "yolo11n_fara_aug",
    "yolo11n_cu_aug",
    "yolo26n_fara_aug",
    "yolo26n_cu_aug",
]


def save_global_metrics(metrics, output_csv):
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metrica", "Valoare"])

        for key, value in metrics.results_dict.items():
            writer.writerow([key, value])


def save_per_class_metrics(metrics, output_csv):
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Clasa", "Precision", "Recall", "mAP50", "mAP50-95"])

        names = metrics.names

        for class_id, class_name in names.items():
            try:
                precision, recall, map50, map5095 = metrics.box.class_result(class_id)
                writer.writerow([
                    class_name,
                    round(float(precision), 4),
                    round(float(recall), 4),
                    round(float(map50), 4),
                    round(float(map5095), 4),
                ])
            except Exception:
                writer.writerow([class_name, "N/A", "N/A", "N/A", "N/A"])


def get_metric_value(metrics, wanted_key_part):
    """
    Cauta flexibil valori in results_dict, pentru ca numele cheilor pot varia usor
    intre versiuni de Ultralytics.
    """
    for key, value in metrics.results_dict.items():
        normalized = key.lower().replace(" ", "")
        if wanted_key_part.lower() in normalized:
            try:
                return float(value)
            except Exception:
                return value
    return "N/A"


def main():
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    print(f"Rulez validarea pe split='{SPLIT}'")
    print(f"Dataset YAML: {DATA_YAML}")

    for experiment_name in EXPERIMENTS:
        run_dir = PROJECT_DIR / experiment_name
        model_path = run_dir / "weights" / "best.pt"

        print("\n" + "=" * 80)
        print(f"Model: {experiment_name}")
        print(f"Caut modelul: {model_path}")

        if not model_path.exists():
            print("Modelul nu exista. Sar peste acest experiment.")
            continue

        output_dir = run_dir / OUTPUT_FOLDER_NAME
        output_dir.mkdir(parents=True, exist_ok=True)

        model = YOLO(str(model_path))

        metrics = model.val(
            data=str(DATA_YAML),
            split=SPLIT,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            plots=True,
            project=str(output_dir),
            name="validare_best_model",
        )

        global_csv = output_dir / "metrici_globale_validare.csv"
        per_class_csv = output_dir / "metrici_pe_clase_validare.csv"

        save_global_metrics(metrics, global_csv)
        save_per_class_metrics(metrics, per_class_csv)

        precision = get_metric_value(metrics, "precision")
        recall = get_metric_value(metrics, "recall")
        map50 = get_metric_value(metrics, "map50(b)")
        map5095 = get_metric_value(metrics, "map50-95")

        # fallback pentru alte denumiri ale cheilor
        if map50 == "N/A":
            map50 = get_metric_value(metrics, "map50")

        summary_rows.append([
            experiment_name,
            precision,
            recall,
            map50,
            map5095,
            str(global_csv),
            str(per_class_csv),
        ])

        print("Metrici salvate:")
        print(global_csv)
        print(per_class_csv)

    summary_csv = PROJECT_DIR / SUMMARY_CSV_NAME
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Experiment",
            "Precision",
            "Recall",
            "mAP50",
            "mAP50-95",
            "Fisier metrici globale",
            "Fisier metrici pe clase",
        ])
        writer.writerows(summary_rows)

    print("\nRezumat salvat aici:")
    print(summary_csv)


if __name__ == "__main__":
    freeze_support()
    main()
