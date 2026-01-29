"""
Run Debris Detection Evaluation

Standalone script to evaluate streak detection performance across an entire dataset.
Imports the streak_detector module and runs comprehensive evaluation.

Usage:
    python run_evaluation.py --dataset-dir debris_dataset
    python run_evaluation.py --dataset-dir debris_dataset --iou-threshold 0.4
    python run_evaluation.py --help
"""

import os
import json
import argparse
import numpy as np
import cv2
from typing import List, Dict, Tuple
from datetime import datetime

# Import the streak detector module
from streak_detector import StreakDetector, Detection


# ============================================================================
# GROUND TRUTH HANDLING
# ============================================================================

class GroundTruth:
    """Represents ground truth debris from labels."""
    def __init__(self, start_point, end_point, debris_type, length, thickness, intensity):
        self.start_point = tuple(start_point)
        self.end_point = tuple(end_point)
        self.debris_type = debris_type
        self.length = length
        self.thickness = thickness
        self.intensity = intensity


def load_ground_truth(label_path: str) -> List[GroundTruth]:
    """Load ground truth from JSON label file."""
    with open(label_path, 'r') as f:
        label_data = json.load(f)
    
    ground_truths = []
    for debris in label_data['debris']:
        gt = GroundTruth(
            start_point=debris['start_point'],
            end_point=debris['end_point'],
            debris_type=debris.get('type', 'linear'),
            length=debris['length'],
            thickness=debris['thickness'],
            intensity=debris['intensity']
        )
        ground_truths.append(gt)
    
    return ground_truths


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class EvaluationMetrics:
    """Container for evaluation metrics."""
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.average_iou = 0.0
    
    def calculate(self):
        """Calculate precision, recall, and F1 from TP/FP/FN counts."""
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        else:
            self.precision = 0.0
        
        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        else:
            self.recall = 0.0
        
        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = 0.0
    
    def __str__(self):
        return (f"Precision: {self.precision:.3f} | Recall: {self.recall:.3f} | "
                f"F1: {self.f1_score:.3f} | Avg IoU: {self.average_iou:.3f}")


# ============================================================================
# MATCHING AND EVALUATION
# ============================================================================

def line_segment_iou(det: Detection, gt: GroundTruth, thickness: int = 5) -> float:
    """
    Calculate IoU between detected and ground truth line segments.
    
    Args:
        det: Detection object
        gt: Ground truth object
        thickness: Line thickness for rasterization
        
    Returns:
        IoU score between 0 and 1
    """
    # Determine canvas size
    all_y = [det.start_point[0], det.end_point[0], gt.start_point[0], gt.end_point[0]]
    all_x = [det.start_point[1], det.end_point[1], gt.start_point[1], gt.end_point[1]]
    
    min_y, max_y = max(0, min(all_y) - 10), max(all_y) + 10
    min_x, max_x = max(0, min(all_x) - 10), max(all_x) + 10
    
    height = max_y - min_y + 1
    width = max_x - min_x + 1
    
    # Create masks
    det_mask = np.zeros((height, width), dtype=np.uint8)
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Shift coordinates to local canvas
    det_start = (det.start_point[1] - min_x, det.start_point[0] - min_y)
    det_end = (det.end_point[1] - min_x, det.end_point[0] - min_y)
    gt_start = (gt.start_point[1] - min_x, gt.start_point[0] - min_y)
    gt_end = (gt.end_point[1] - min_x, gt.end_point[0] - min_y)
    
    # Draw lines
    cv2.line(det_mask, det_start, det_end, 255, thickness)
    cv2.line(gt_mask, gt_start, gt_end, 255, max(thickness, gt.thickness))
    
    # Calculate IoU
    intersection = np.logical_and(det_mask > 0, gt_mask > 0).sum()
    union = np.logical_or(det_mask > 0, gt_mask > 0).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def match_detections(detections: List[Detection], 
                    ground_truths: List[GroundTruth],
                    iou_threshold: float = 0.3) -> Tuple[List, List, List]:
    """
    Match detections to ground truth using greedy IoU matching.
    
    Returns:
        Tuple of (matches, unmatched_detections, unmatched_ground_truths)
    """
    if len(detections) == 0 or len(ground_truths) == 0:
        return [], list(range(len(detections))), list(range(len(ground_truths)))
    
    # Build similarity matrix
    similarity_matrix = np.zeros((len(detections), len(ground_truths)))
    
    for i, det in enumerate(detections):
        for j, gt in enumerate(ground_truths):
            similarity_matrix[i, j] = line_segment_iou(det, gt)
    
    # Greedy matching
    matches = []
    matched_dets = set()
    matched_gts = set()
    
    # Sort by IoU score
    coords = [(i, j, similarity_matrix[i, j]) 
              for i in range(len(detections)) 
              for j in range(len(ground_truths))]
    coords.sort(key=lambda x: x[2], reverse=True)
    
    for det_idx, gt_idx, score in coords:
        if det_idx in matched_dets or gt_idx in matched_gts:
            continue
        
        if score >= iou_threshold:
            matches.append((det_idx, gt_idx, score))
            matched_dets.add(det_idx)
            matched_gts.add(gt_idx)
    
    unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
    unmatched_gts = [i for i in range(len(ground_truths)) if i not in matched_gts]
    
    return matches, unmatched_dets, unmatched_gts


def evaluate_single_image(detections: List[Detection],
                         ground_truths: List[GroundTruth],
                         iou_threshold: float = 0.3) -> EvaluationMetrics:
    """Evaluate detections for a single image."""
    matches, unmatched_dets, unmatched_gts = match_detections(
        detections, ground_truths, iou_threshold
    )
    
    metrics = EvaluationMetrics()
    metrics.true_positives = len(matches)
    metrics.false_positives = len(unmatched_dets)
    metrics.false_negatives = len(unmatched_gts)
    
    if len(matches) > 0:
        ious = [score for _, _, score in matches]
        metrics.average_iou = np.mean(ious)
    else:
        metrics.average_iou = 0.0
    
    metrics.calculate()
    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(img_path: str,
                     detections: List[Detection],
                     ground_truths: List[GroundTruth],
                     matches: List,
                     output_path: str):
    """
    Visualize evaluation results.
    Green = TP, Red = FP, Yellow = FN
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    
    img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Get matched indices
    matched_det_indices = {det_idx for det_idx, _, _ in matches}
    matched_gt_indices = {gt_idx for _, gt_idx, _ in matches}
    
    # Draw False Negatives (missed ground truth) in yellow
    for idx, gt in enumerate(ground_truths):
        if idx not in matched_gt_indices:
            cv2.line(img_vis,
                    (gt.start_point[1], gt.start_point[0]),
                    (gt.end_point[1], gt.end_point[0]),
                    (0, 255, 255), 3)  # Yellow - thicker to show it's missed
    
    # Draw False Positives (wrong detections) in red
    for idx, det in enumerate(detections):
        if idx not in matched_det_indices:
            cv2.line(img_vis,
                    (det.start_point[1], det.start_point[0]),
                    (det.end_point[1], det.end_point[0]),
                    (0, 0, 255), 2)  # Red
    
    # Draw True Positives (correct detections) in green
    for det_idx, gt_idx, iou in matches:
        det = detections[det_idx]
        cv2.line(img_vis,
                (det.start_point[1], det.start_point[0]),
                (det.end_point[1], det.end_point[0]),
                (0, 255, 0), 2)  # Green
        
        # Add IoU score
        mid_y = (det.start_point[0] + det.end_point[0]) // 2
        mid_x = (det.start_point[1] + det.end_point[1]) // 2
        cv2.putText(img_vis, f"{iou:.2f}",
                   (mid_x, mid_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.imwrite(output_path, img_vis)


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def run_evaluation(dataset_dir: str,
                  detector: StreakDetector,
                  iou_threshold: float = 0.3,
                  save_visualizations: bool = True,
                  save_results: bool = True) -> Dict:
    """
    Run evaluation across entire dataset.
    
    Args:
        dataset_dir: Directory containing images/ and labels/ subdirectories
        detector: StreakDetector instance to use
        iou_threshold: IoU threshold for matching
        save_visualizations: Whether to save visualization images
        save_results: Whether to save results to JSON
        
    Returns:
        Dictionary with evaluation results
    """
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    if not os.path.exists(labels_dir):
        raise ValueError(f"Labels directory not found: {labels_dir}")
    
    # Create output directories
    if save_visualizations:
        vis_dir = os.path.join(dataset_dir, 'evaluation_results')
        os.makedirs(vis_dir, exist_ok=True)
    
    # Get image files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    
    if len(image_files) == 0:
        raise ValueError(f"No PNG images found in {images_dir}")
    
    print("="*70)
    print("DEBRIS DETECTION EVALUATION")
    print("="*70)
    print(f"Dataset: {dataset_dir}")
    print(f"Images: {len(image_files)}")
    print(f"IoU Threshold: {iou_threshold}")
    print(f"Detector: threshold={detector.threshold}, min_length={detector.min_length}, min_aspect={detector.min_aspect}")
    print("="*70)
    
    # Accumulate metrics
    overall_metrics = EvaluationMetrics()
    per_image_results = []
    
    print(f"\nProcessing images...")
    
    for idx, img_filename in enumerate(image_files):
        # Paths
        img_path = os.path.join(images_dir, img_filename)
        label_path = os.path.join(labels_dir, img_filename.replace('.png', '.json'))
        
        if not os.path.exists(label_path):
            print(f"  Warning: No label file for {img_filename}, skipping")
            continue
        
        # Load ground truth
        ground_truths = load_ground_truth(label_path)
        
        # Run detection
        detections = detector.detect(img_path)
        
        # Evaluate
        matches, unmatched_dets, unmatched_gts = match_detections(
            detections, ground_truths, iou_threshold
        )
        img_metrics = evaluate_single_image(detections, ground_truths, iou_threshold)
        
        # Accumulate
        overall_metrics.true_positives += img_metrics.true_positives
        overall_metrics.false_positives += img_metrics.false_positives
        overall_metrics.false_negatives += img_metrics.false_negatives
        
        # Store result
        per_image_results.append({
            'filename': img_filename,
            'num_detections': len(detections),
            'num_ground_truth': len(ground_truths),
            'true_positives': img_metrics.true_positives,
            'false_positives': img_metrics.false_positives,
            'false_negatives': img_metrics.false_negatives,
            'precision': img_metrics.precision,
            'recall': img_metrics.recall,
            'f1_score': img_metrics.f1_score,
            'average_iou': img_metrics.average_iou
        })
        
        # Visualize
        if save_visualizations:
            vis_path = os.path.join(vis_dir, f"eval_{img_filename}")
            visualize_results(img_path, detections, ground_truths, matches, vis_path)
        
        # Progress
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(image_files)} images")
    
    # Calculate overall metrics
    overall_metrics.calculate()
    
    # Calculate average IoU
    all_ious = [r['average_iou'] for r in per_image_results if r['true_positives'] > 0]
    if all_ious:
        overall_metrics.average_iou = np.mean(all_ious)
    
    # Compile results
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_dir': dataset_dir,
        'num_images': len(per_image_results),
        'iou_threshold': iou_threshold,
        'detector_params': {
            'threshold': detector.threshold,
            'min_length': detector.min_length,
            'min_aspect': detector.min_aspect
        },
        'overall_metrics': {
            'true_positives': overall_metrics.true_positives,
            'false_positives': overall_metrics.false_positives,
            'false_negatives': overall_metrics.false_negatives,
            'precision': overall_metrics.precision,
            'recall': overall_metrics.recall,
            'f1_score': overall_metrics.f1_score,
            'average_iou': overall_metrics.average_iou
        },
        'per_image_results': per_image_results
    }
    
    # Print report
    print_report(results)
    
    # Save results
    if save_results:
        results_path = os.path.join(dataset_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {results_path}")
    
    if save_visualizations:
        print(f"✓ Visualizations saved to: {vis_dir}/")
    
    return results


# ============================================================================
# REPORTING
# ============================================================================

def print_report(results: Dict):
    """Print detailed evaluation report."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    overall = results['overall_metrics']
    
    print(f"\nOVERALL METRICS:")
    print(f"  True Positives:  {overall['true_positives']}")
    print(f"  False Positives: {overall['false_positives']}")
    print(f"  False Negatives: {overall['false_negatives']}")
    print(f"\n  Precision: {overall['precision']:.3f}")
    print(f"  Recall:    {overall['recall']:.3f}")
    print(f"  F1 Score:  {overall['f1_score']:.3f}")
    print(f"  Avg IoU:   {overall['average_iou']:.3f}")
    
    # Per-image stats
    per_img = results['per_image_results']
    if per_img:
        precisions = [r['precision'] for r in per_img 
                     if r['true_positives'] + r['false_positives'] > 0]
        recalls = [r['recall'] for r in per_img
                  if r['true_positives'] + r['false_negatives'] > 0]
        
        print(f"\nPER-IMAGE STATISTICS:")
        if precisions:
            print(f"  Avg Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
        if recalls:
            print(f"  Avg Recall:    {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
        
        # Worst performing
        print(f"\nWORST 5 IMAGES (by F1 score):")
        sorted_results = sorted(per_img, key=lambda x: x['f1_score'])
        for result in sorted_results[:5]:
            print(f"  {result['filename']}: F1={result['f1_score']:.3f} "
                  f"(P={result['precision']:.3f}, R={result['recall']:.3f})")
    
    print("\n" + "="*70)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate debris streak detection on a dataset"
    )
    parser.add_argument(
        "--dataset-dir", 
        type=str, 
        required=True,
        help="Path to dataset directory (containing images/ and labels/)"
    )
    parser.add_argument(
        "--iou-threshold", 
        type=float, 
        default=0.3,
        help="IoU threshold for matching detections to ground truth (default: 0.3)"
    )
    parser.add_argument(
        "--detection-threshold", 
        type=float, 
        default=0.4,
        help="Brightness threshold for detection (0-1, default: 0.4)"
    )
    parser.add_argument(
        "--min-length", 
        type=int, 
        default=40,
        help="Minimum streak length in pixels (default: 40)"
    )
    parser.add_argument(
        "--min-aspect", 
        type=float, 
        default=3.0,
        help="Minimum aspect ratio for streaks (default: 3.0)"
    )
    parser.add_argument(
        "--no-visualizations", 
        action="store_true",
        help="Skip saving visualization images (faster)"
    )
    parser.add_argument(
        "--no-save", 
        action="store_true",
        help="Don't save results JSON"
    )
    
    args = parser.parse_args()
    
    # Create detector with specified parameters
    detector = StreakDetector(
        threshold=int(args.detection_threshold * 255),  # Convert 0-1 to 0-255
        min_length=args.min_length,
        min_aspect=args.min_aspect
    )
    
    # Run evaluation
    results = run_evaluation(
        dataset_dir=args.dataset_dir,
        detector=detector,
        iou_threshold=args.iou_threshold,
        save_visualizations=not args.no_visualizations,
        save_results=not args.no_save
    )