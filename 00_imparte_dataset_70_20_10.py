from pathlib import Path
import random
import shutil

from config_experimente import (
    SOURCE_DATASET,
    SPLIT_DATASET,
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    TRAIN_PERCENT,
    VAL_PERCENT,
)

random.seed(42)


def get_all_image_label_pairs():
    """
    Ia toate imaginile existente din datasetul vechi.
    Nu modifica datasetul vechi. Doar citeste din el.
    Cauta in train, val si test, daca aceste foldere exista.
    """
    all_pairs = []
    seen = set()

    for split in ["train", "val", "test"]:
        image_dir = SOURCE_DATASET / "images" / split
        label_dir = SOURCE_DATASET / "labels" / split

        if not image_dir.exists():
            print(f"Folder inexistent, il sar: {image_dir}")
            continue

        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # Cheia include numele, ca sa evitam duplicatele intre train/val/test.
            key = image_path.name
            if key in seen:
                print(f"Imagine duplicata ignorata: {image_path.name}")
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                print(f"ATENTIE: lipseste label pentru {image_path.name}. Imaginea este ignorata.")
                continue

            all_pairs.append((image_path, label_path))
            seen.add(key)

    return all_pairs


def copy_files(pairs, split_name):
    """Copiaza imaginile si labelurile in noul split."""
    images_out = SPLIT_DATASET / "images" / split_name
    labels_out = SPLIT_DATASET / "labels" / split_name

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for image_path, label_path in pairs:
        shutil.copy2(image_path, images_out / image_path.name)
        shutil.copy2(label_path, labels_out / label_path.name)


def create_yaml():
    """Creeaza fisierul YAML nou, cu train, val si test."""
    yaml_path = SPLIT_DATASET / "coco_blind_split.yaml"

    content = f"""path: {SPLIT_DATASET.as_posix()}
train: images/train
val: images/val
test: images/test

names:
"""

    for index, class_name in enumerate(CLASS_NAMES):
        content += f"  {index}: {class_name}\n"

    yaml_path.write_text(content, encoding="utf-8")
    print("\nYAML nou creat:")
    print(yaml_path)


def main():
    print("Dataset sursa:")
    print(SOURCE_DATASET)
    print("\nDataset nou care va fi creat:")
    print(SPLIT_DATASET)

    if SPLIT_DATASET.exists():
        print("\nSTOP: folderul de output exista deja.")
        print("Nu il suprascriu ca sa nu incurc rezultatele.")
        print("Daca vrei sa refaci impartirea, sterge manual folderul:")
        print(SPLIT_DATASET)
        return

    all_pairs = get_all_image_label_pairs()

    if len(all_pairs) == 0:
        print("\nNu am gasit nicio imagine cu label. Verifica SOURCE_DATASET din config_experimente.py")
        return

    random.shuffle(all_pairs)

    total = len(all_pairs)
    n_train = int(total * TRAIN_PERCENT)
    n_val = int(total * VAL_PERCENT)

    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:n_train + n_val]
    test_pairs = all_pairs[n_train + n_val:]

    copy_files(train_pairs, "train")
    copy_files(val_pairs, "val")
    copy_files(test_pairs, "test")
    create_yaml()

    print("\nImpartire finala:")
    print(f"Train: {len(train_pairs)} imagini ({len(train_pairs) / total * 100:.2f}%)")
    print(f"Val:   {len(val_pairs)} imagini ({len(val_pairs) / total * 100:.2f}%)")
    print(f"Test:  {len(test_pairs)} imagini ({len(test_pairs) / total * 100:.2f}%)")
    print(f"Total: {total} imagini")
    print("\nDatasetul vechi NU a fost modificat. S-au facut doar copii.")


if __name__ == "__main__":
    main()
