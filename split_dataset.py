"""
Split existing dataset into train and test sets.
"""

import os
import shutil
from pathlib import Path
import random
import argparse


def split_dataset(source_dir: str, train_dir: str, test_dir: str,
                  test_split: float = 0.2, seed: int = 42):
    """
    Split dataset into train and test sets.

    Args:
        source_dir: Directory containing streaks/ and non-streaks/ subdirectories
        train_dir: Output directory for training data
        test_dir: Output directory for test data
        test_split: Fraction of data to use for testing (default: 0.2 = 20%)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    source_path = Path(source_dir)
    train_path = Path(train_dir)
    test_path = Path(test_dir)

    # Create output directories
    for base_dir in [train_path, test_path]:
        for class_name in ['streaks', 'non-streaks']:
            (base_dir / class_name / 'images').mkdir(parents=True, exist_ok=True)
            (base_dir / class_name / 'labels').mkdir(parents=True, exist_ok=True)

    stats = {'train': {'streaks': 0, 'non-streaks': 0},
             'test': {'streaks': 0, 'non-streaks': 0}}

    # Process each class
    for class_name in ['streaks', 'non-streaks']:
        print(f"\nProcessing {class_name}...")

        images_dir = source_path / class_name / 'images'
        labels_dir = source_path / class_name / 'labels'

        if not images_dir.exists():
            print(f"  Warning: {images_dir} does not exist, skipping...")
            continue

        # Get all image files
        image_files = sorted([f for f in images_dir.iterdir() if f.is_file()])
        total_images = len(image_files)

        if total_images == 0:
            print(f"  Warning: No images found in {images_dir}")
            continue

        # Shuffle and split
        random.shuffle(image_files)
        test_size = int(total_images * test_split)
        test_files = image_files[:test_size]
        train_files = image_files[test_size:]

        print(f"  Total images: {total_images}")
        print(f"  Train: {len(train_files)}")
        print(f"  Test: {len(test_files)}")

        # Copy files to train set
        for img_file in train_files:
            # Copy image
            shutil.copy2(img_file, train_path / class_name / 'images' / img_file.name)

            # Copy corresponding label if it exists
            label_file = labels_dir / img_file.with_suffix('.json').name
            if label_file.exists():
                shutil.copy2(label_file, train_path / class_name / 'labels' / label_file.name)

            stats['train'][class_name] += 1

        # Copy files to test set
        for img_file in test_files:
            # Copy image
            shutil.copy2(img_file, test_path / class_name / 'images' / img_file.name)

            # Copy corresponding label if it exists
            label_file = labels_dir / img_file.with_suffix('.json').name
            if label_file.exists():
                shutil.copy2(label_file, test_path / class_name / 'labels' / label_file.name)

            stats['test'][class_name] += 1

    # Print summary
    print(f"\n{'='*60}")
    print("Dataset split complete!")
    print(f"{'='*60}")
    print(f"\nTraining set ({train_dir}):")
    print(f"  Streaks: {stats['train']['streaks']}")
    print(f"  Non-streaks: {stats['train']['non-streaks']}")
    print(f"  Total: {stats['train']['streaks'] + stats['train']['non-streaks']}")

    print(f"\nTest set ({test_dir}):")
    print(f"  Streaks: {stats['test']['streaks']}")
    print(f"  Non-streaks: {stats['test']['non-streaks']}")
    print(f"  Total: {stats['test']['streaks'] + stats['test']['non-streaks']}")

    total = sum(stats['train'].values()) + sum(stats['test'].values())
    test_percentage = (sum(stats['test'].values()) / total * 100) if total > 0 else 0
    print(f"\nTest split: {test_percentage:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train and test sets")
    parser.add_argument("--source", type=str, default="training_dataset",
                       help="Source directory containing the full dataset")
    parser.add_argument("--train-dir", type=str, default="train_dataset",
                       help="Output directory for training data")
    parser.add_argument("--test-dir", type=str, default="test_dataset_new",
                       help="Output directory for test data")
    parser.add_argument("--test-split", type=float, default=0.2,
                       help="Fraction of data to use for testing (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    split_dataset(args.source, args.train_dir, args.test_dir,
                  args.test_split, args.seed)
