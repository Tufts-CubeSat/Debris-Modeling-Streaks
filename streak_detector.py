"""Detector with black/white (0/1) detection and then detecting lines/connected components"""

import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Detection:
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    confidence: float = 1.0
    length: int = 0
    angle: float = 0.0
    
    def __post_init__(self):
        dy = self.end_point[0] - self.start_point[0]
        dx = self.end_point[1] - self.start_point[1]
        self.length = int(np.sqrt(dy**2 + dx**2))
        self.angle = np.degrees(np.arctan2(dy, dx))


class StreakDetector:
    def __init__(self, threshold: int = 100, min_length: int = 40, min_aspect: float = 3.0):
        self.threshold = threshold
        self.min_length = min_length
        self.min_aspect = min_aspect
    
    def detect(self, image_path: str) -> List[Detection]:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return []
        
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        
        _, binary = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        detections = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            aspect = max(w, h) / (min(w, h) + 1e-6)
            
            if max(w, h) < self.min_length or aspect < self.min_aspect:
                continue
            
            points = np.column_stack(np.where(labels == i))
            if len(points) < 2:
                continue
            
            dists = np.sum((points[:, None] - points[None, :])**2, axis=2)
            i1, i2 = np.unravel_index(np.argmax(dists), dists.shape)
            
            detections.append(Detection(
                start_point=tuple(points[i1]),
                end_point=tuple(points[i2])
            ))
        
        return detections


def visualize_detections(image_path: str, detections: List[Detection], output_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for det in detections:
        cv2.line(img_vis, (det.start_point[1], det.start_point[0]),
                (det.end_point[1], det.end_point[0]), (0, 255, 0), 2)
        cv2.circle(img_vis, (det.start_point[1], det.start_point[0]), 4, (255, 0, 0), -1)
        cv2.circle(img_vis, (det.end_point[1], det.end_point[0]), 4, (0, 0, 255), -1)
    
    cv2.imwrite(output_path, img_vis)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python streak_detector.py <image_path> [output_path]")
        sys.exit(1)
    
    detector = StreakDetector(threshold=100, min_length=40, min_aspect=3.0)
    detections = detector.detect(sys.argv[1])
    
    print(f"Found {len(detections)} streaks")
    for i, det in enumerate(detections, 1):
        print(f"  {i}. Length: {det.length}px, Angle: {det.angle:.1f}°")
    
    if len(sys.argv) > 2:
        visualize_detections(sys.argv[1], detections, sys.argv[2])
        print(f"Saved to: {sys.argv[2]}")