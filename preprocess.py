import os
import json
import shutil
import random
import argparse
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
IMG_EXTENSIONS = [".jpg", ".jpeg", ".png"]

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess the IP102 dataset for YOLO format.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the IP102 dataset folder.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the preprocessed data.")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    return args

def create_output_folders(output_path):
    os.makedirs(os.path.join(output_path, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "images/test"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels/train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels/val"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels/test"), exist_ok=True)

def load_annotations(dataset_path):
    csv_path = os.path.join(dataset_path, "IP102_annotations.csv")
    annotations = pd.read_csv(csv_path)
    return annotations

def convert_to_yolo_format(row, image_width, image_height):
    # Normalize bounding box coordinates
    xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
    x_center = ((xmin + xmax) / 2) / image_width
    y_center = ((ymin + ymax) / 2) / image_height
    bbox_width = (xmax - xmin) / image_width
    bbox_height = (ymax - ymin) / image_height

    return f"{row['category_id']} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"

def save_yolo_labels(data_split, split_name, dataset_path, output_path):
    for _, row in tqdm(data_split.iterrows(), total=len(data_split), desc=f"Processing {split_name} labels"):
        image_path = os.path.join(dataset_path, "images", row["image_path"])
        label_path = os.path.join(output_path, f"labels/{split_name}", os.path.splitext(row["image_path"])[0] + ".txt")

        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]

        yolo_label = convert_to_yolo_format(row, image_width, image_height)

        with open(label_path, "a") as label_file:
            label_file.write(yolo_label + "\n")

def copy_images(data_split, split_name, dataset_path, output_path):
    for _, row in tqdm(data_split.iterrows(), total=len(data_split), desc=f"Copying {split_name} images"):
        src = os.path.join(dataset_path, "images", row["image_path"])
        dst = os.path.join(output_path, f"images/{split_name}", os.path.basename(row["image_path"]))
        if os.path.exists(src):
            shutil.copy(src, dst)

def augment_images(data_split, split_name, output_path):
    for _, row in tqdm(data_split.iterrows(), total=len(data_split), desc=f"Augmenting {split_name} images"):
        image_path = os.path.join(output_path, f"images/{split_name}", os.path.basename(row["image_path"]))
        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)

        # Augmentation: Horizontal Flip
        flipped_image = cv2.flip(image, 1)
        flipped_path = image_path.replace(".jpg", "_flipped.jpg")
        cv2.imwrite(flipped_path, flipped_image)

        # Augmentation: Rotation
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_path = image_path.replace(".jpg", "_rotated.jpg")
        cv2.imwrite(rotated_path, rotated_image)

        # Augmentation: Brightness adjustment
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 50)
        bright_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        bright_path = image_path.replace(".jpg", "_bright.jpg")
        cv2.imwrite(bright_path, bright_image)

def split_dataset(annotations, output_path, seed):
    random.seed(seed)
    train, temp = train_test_split(annotations, test_size=1 - TRAIN_RATIO, random_state=seed)
    val, test = train_test_split(temp, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO), random_state=seed)

    train.to_csv(os.path.join(output_path, "annotations/train.csv"), index=False)
    val.to_csv(os.path.join(output_path, "annotations/val.csv"), index=False)
    test.to_csv(os.path.join(output_path, "annotations/test.csv"), index=False)

    return train, val, test

def main():
    args = parse_args()
    print("Starting dataset preprocessing...")

    create_output_folders(args.output_path)

    print("Loading annotations...")
    annotations = load_annotations(args.dataset_path)

    print("Splitting dataset into train, validation, and test sets...")
    train, val, test = split_dataset(annotations, args.output_path, args.seed)

    print("Saving YOLO format labels...")
    save_yolo_labels(train, "train", args.dataset_path, args.output_path)
    save_yolo_labels(val, "val", args.dataset_path, args.output_path)
    save_yolo_labels(test, "test", args.dataset_path, args.output_path)

    print("Copying images...")
    copy_images(train, "train", args.dataset_path, args.output_path)
    copy_images(val, "val", args.dataset_path, args.output_path)
    copy_images(test, "test", args.dataset_path, args.output_path)

    if args.augment:
        print("Applying data augmentation...")
        augment_images(train, "train", args.output_path)

    print("Preprocessing complete. Data saved to:", args.output_path)

if __name__ == "__main__":
    main()
