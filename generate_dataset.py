"""
Example script to generate a dataset of semisynthetic space debris images.
"""

import os
import json
import argparse
from debris_data_generator import DebrisDataGenerator
import cv2
import numpy as np


def generate_dataset(output_dir: str, num_images: int = 100, seed: int = 42):
    """
    Generate a dataset of debris detection images.

    Args:
        output_dir: Directory to save generated images
        num_images: Number of images to generate
        seed: Random seed for reproducibility
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Initialize generator
    generator = DebrisDataGenerator(image_size=(1024, 1024), seed=seed)

    # Generate images
    print(f"Generating {num_images} images...")
    for i in range(num_images):
        # Vary the number of debris per image
        num_debris = generator.rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])

        # Generate image
        img, debris_params = generator.generate_image(
            num_debris=num_debris,
            num_stars=generator.rng.randint(300, 800),
            curved_debris_prob=0.2,
            noise_level=generator.rng.uniform(0.01, 0.03),
            vignetting_strength=generator.rng.uniform(0.1, 0.3)
        )

        # Save image
        img_filename = f"debris_{i:04d}.png"
        img_path = os.path.join(images_dir, img_filename)
        generator.save_image(img, img_path, format='png')

        # Save labels as JSON
        label_filename = f"debris_{i:04d}.json"
        label_path = os.path.join(labels_dir, label_filename)

        # Convert debris parameters to serializable format
        labels = {
            'image_id': int(i),
            'image_filename': img_filename,
            'image_size': list(generator.image_size),
            'num_debris': int(num_debris),
            'debris': []
        }

        for debris in debris_params:
            debris_label = {
                'start_point': [int(debris['start_point'][0]), int(debris['start_point'][1])],
                'angle': float(debris['angle']) if 'angle' in debris else None,
                'length': int(debris['length']),
                'thickness': int(debris['thickness']),
                'intensity': float(debris['intensity']),
                'type': debris.get('type', 'linear')
            }
            if 'end_point' in debris:
                debris_label['end_point'] = [int(debris['end_point'][0]), int(debris['end_point'][1])]
            if 'curve_amount' in debris:
                debris_label['curve_amount'] = float(debris['curve_amount'])

            labels['debris'].append(debris_label)

        with open(label_path, 'w') as f:
            json.dump(labels, f, indent=2)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_images} images")

    print(f"\nDataset generation complete!")
    print(f"Images saved to: {images_dir}")
    print(f"Labels saved to: {labels_dir}")


def generate_preview_image(output_path: str = "preview.png"):
    """
    Generate a single preview image showing multiple debris streaks.
    """
    generator = DebrisDataGenerator(image_size=(1024, 1024), seed=42)

    # Generate image with multiple debris
    img, debris_params = generator.generate_image(
        num_debris=5,
        num_stars=600,
        curved_debris_prob=0.3,
        noise_level=0.02,
        vignetting_strength=0.2
    )

    # Create annotated version
    img_annotated = (img * 255).astype(np.uint8)
    img_annotated = cv2.cvtColor(img_annotated, cv2.COLOR_GRAY2BGR)

    # Draw bounding boxes and labels
    for idx, debris in enumerate(debris_params):
        start = debris['start_point']

        if 'end_point' in debris:
            end = debris['end_point']
            # Draw line showing the streak
            cv2.line(img_annotated,
                    (start[1], start[0]),
                    (end[1], end[0]),
                    (0, 255, 0), 2)

            # Draw start and end points
            cv2.circle(img_annotated, (start[1], start[0]), 5, (255, 0, 0), -1)
            cv2.circle(img_annotated, (end[1], end[0]), 5, (0, 0, 255), -1)
        else:
            cv2.circle(img_annotated, (start[1], start[0]), 5, (255, 0, 0), -1)

        # Add label
        label = f"Debris {idx+1}"
        cv2.putText(img_annotated, label,
                   (start[1] + 10, start[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Save both versions side by side
    combined = np.hstack([
        cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
        img_annotated
    ])

    cv2.imwrite(output_path, combined)
    print(f"Preview image saved to: {output_path}")
    print(f"  Left: Original image")
    print(f"  Right: Annotated with debris locations")
    print(f"\nDebris details:")
    for idx, debris in enumerate(debris_params):
        print(f"  Debris {idx+1}:")
        print(f"    Type: {debris.get('type', 'linear')}")
        print(f"    Length: {debris['length']} pixels")
        print(f"    Thickness: {debris['thickness']} pixels")
        print(f"    Intensity: {debris['intensity']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate semisynthetic space debris dataset")
    parser.add_argument("--output-dir", type=str, default="debris_dataset",
                       help="Output directory for dataset")
    parser.add_argument("--num-images", type=int, default=100,
                       help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--preview-only", action="store_true",
                       help="Only generate preview image")

    args = parser.parse_args()

    if args.preview_only:
        generate_preview_image()
    else:
        generate_dataset(args.output_dir, args.num_images, args.seed)
        # Also generate a preview
        preview_path = os.path.join(args.output_dir, "preview.png")
        generate_preview_image(preview_path)
