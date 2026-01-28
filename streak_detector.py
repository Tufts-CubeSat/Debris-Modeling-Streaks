"""
Debris Streak Detection Module

A standalone module for detecting space debris streaks in astronomical images.
Can be imported and used in any application.

Usage:
    from streak_detector import StreakDetector, Detection
    
    detector = StreakDetector(threshold=0.4, min_length=40)
    detections = detector.detect(image_path)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Detection:
    """Represents a detected debris streak."""
    start_point: Tuple[int, int]  # (y, x)
    end_point: Tuple[int, int]    # (y, x)
    confidence: float = 1.0
    length: Optional[int] = None
    angle: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived properties."""
        if self.length is None:
            dy = self.end_point[0] - self.start_point[0]
            dx = self.end_point[1] - self.start_point[1]
            self.length = int(np.sqrt(dy**2 + dx**2))
        
        if self.angle is None:
            dy = self.end_point[0] - self.start_point[0]
            dx = self.end_point[1] - self.start_point[1]
            self.angle = np.degrees(np.arctan2(dy, dx))


class StreakDetector:
    """
    Detects debris streaks in astronomical images.
    
    This is a template implementation using Hough Line Transform.
    Replace the detection logic with your own algorithm!
    """
    
    def __init__(self, 
                 threshold: float = 0.4,
                 min_length: int = 40,
                 max_gap: int = 10,
                 hough_threshold: int = 20,
                 canny_low: int = 50,
                 canny_high: int = 150):
        """
        Initialize the streak detector.
        
        Args:
            threshold: Brightness threshold for streak detection (0-1)
            min_length: Minimum streak length in pixels
            max_gap: Maximum gap allowed in a streak
            hough_threshold: Minimum votes for Hough line detection
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
        """
        self.threshold = threshold
        self.min_length = min_length
        self.max_gap = max_gap
        self.hough_threshold = hough_threshold
        self.canny_low = canny_low
        self.canny_high = canny_high
    
    def detect(self, image_path: str) -> List[Detection]:
        """
        Detect debris streaks in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of Detection objects
        """
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Preprocess
        img_processed = self._preprocess(img)
        
        # Detect lines
        lines = self._detect_lines(img_processed)
        
        # Convert to Detection objects
        detections = self._lines_to_detections(lines)
        
        # Post-process (filter, merge, etc.)
        detections = self._postprocess(detections)
        
        return detections
    
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for streak detection.
        
        Args:
            img: Input image (may be 8-bit or 16-bit)
            
        Returns:
            8-bit preprocessed image
        """
        # Convert to 8-bit if needed (images from generator are 16-bit)
        if img.dtype == np.uint16:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply threshold to isolate bright features
        _, thresh = cv2.threshold(
            img, 
            int(self.threshold * 255), 
            255, 
            cv2.THRESH_BINARY
        )
        
        # Morphological closing to connect nearby bright pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return closed
    
    def _detect_lines(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect lines using Hough Line Transform.
        
        Args:
            img: Preprocessed binary image
            
        Returns:
            Array of detected lines or None
        """
        # Edge detection
        edges = cv2.Canny(img, self.canny_low, self.canny_high)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,                      # Distance resolution in pixels
            theta=np.pi/180,            # Angle resolution in radians
            threshold=self.hough_threshold,  # Min votes
            minLineLength=self.min_length,   # Min line length
            maxLineGap=self.max_gap          # Max gap between line segments
        )
        
        return lines
    
    def _lines_to_detections(self, lines: Optional[np.ndarray]) -> List[Detection]:
        """
        Convert Hough lines to Detection objects.
        
        Args:
            lines: Array of lines from HoughLinesP
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        if lines is None:
            return detections
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Convert to (y, x) format for consistency with ground truth
            detection = Detection(
                start_point=(y1, x1),
                end_point=(y2, x2),
                confidence=1.0  # Could compute based on line strength
            )
            detections.append(detection)
        
        return detections
    
    def _postprocess(self, detections: List[Detection]) -> List[Detection]:
        """
        Post-process detections (filter, merge, etc.).
        
        Args:
            detections: Raw detections
            
        Returns:
            Filtered/processed detections
        """
        # Filter out very short lines (noise)
        filtered = [d for d in detections if d.length >= self.min_length]
        
        # TODO: Add more sophisticated post-processing:
        # - Merge nearby parallel lines
        # - Remove lines that are likely stars (point sources)
        # - Filter based on orientation (if debris has expected direction)
        # - Non-maximum suppression
        
        return filtered


# ============================================================================
# ALTERNATIVE DETECTION METHODS
# ============================================================================

class TemplateMatchingDetector(StreakDetector):
    """
    Streak detector using template matching.
    Inherit from StreakDetector and override detection method.
    """
    
    def __init__(self, 
                 threshold: float = 0.6,
                 template_length: int = 50,
                 template_thickness: int = 2,
                 **kwargs):
        super().__init__(**kwargs)
        self.template_threshold = threshold
        self.template_length = template_length
        self.template_thickness = template_thickness
        self.template = self._create_template()
    
    def _create_template(self) -> np.ndarray:
        """Create a line template for matching."""
        template = np.zeros(
            (self.template_thickness * 4, self.template_length), 
            dtype=np.uint8
        )
        center_y = template.shape[0] // 2
        cv2.line(
            template, 
            (0, center_y), 
            (self.template_length-1, center_y), 
            255, 
            self.template_thickness
        )
        return template
    
    def _detect_lines(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Detect lines using template matching."""
        # Match template
        result = cv2.matchTemplate(img, self.template, cv2.TM_CCOEFF_NORMED)
        
        # Find matches above threshold
        locations = np.where(result >= self.template_threshold)
        
        # Convert to line format
        lines = []
        for y, x in zip(*locations):
            x1, y1 = x, y + self.template.shape[0] // 2
            x2, y2 = x + self.template_length, y + self.template.shape[0] // 2
            lines.append([[x1, y1, x2, y2]])
        
        return np.array(lines) if lines else None


class RadonTransformDetector(StreakDetector):
    """
    Streak detector using Radon transform.
    Good for detecting lines at various angles.
    """
    
    def __init__(self, 
                 threshold: float = 0.5,
                 num_angles: int = 180,
                 **kwargs):
        super().__init__(**kwargs)
        self.radon_threshold = threshold
        self.num_angles = num_angles
    
    def _detect_lines(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Detect lines using Radon transform."""
        try:
            from skimage.transform import radon, iradon
            from scipy.signal import find_peaks
        except ImportError:
            raise ImportError(
                "Radon transform requires scikit-image. "
                "Install with: pip install scikit-image scipy"
            )
        
        # Compute Radon transform
        theta = np.linspace(0, 180, self.num_angles, endpoint=False)
        sinogram = radon(img, theta=theta)
        
        # Find peaks in sinogram (indicates lines)
        lines = []
        for angle_idx in range(sinogram.shape[1]):
            column = sinogram[:, angle_idx]
            peaks, properties = find_peaks(
                column, 
                height=self.radon_threshold * column.max()
            )
            
            # Convert peaks back to line segments
            # TODO: Implement inverse Radon to get line endpoints
            # This is a placeholder
        
        return np.array(lines) if lines else None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def visualize_detections(image_path: str, 
                        detections: List[Detection],
                        output_path: str):
    """
    Visualize detections on the image.
    
    Args:
        image_path: Path to original image
        detections: List of Detection objects
        output_path: Where to save visualization
    """
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert to color
    img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw detections
    for idx, det in enumerate(detections):
        # Draw line
        cv2.line(
            img_vis,
            (det.start_point[1], det.start_point[0]),  # Convert (y,x) to (x,y)
            (det.end_point[1], det.end_point[0]),
            (0, 255, 0),  # Green
            2
        )
        
        # Draw endpoints
        cv2.circle(img_vis, (det.start_point[1], det.start_point[0]), 
                  4, (255, 0, 0), -1)  # Blue start
        cv2.circle(img_vis, (det.end_point[1], det.end_point[0]), 
                  4, (0, 0, 255), -1)  # Red end
        
        # Add label
        label = f"{idx+1}: L={det.length}px"
        cv2.putText(
            img_vis, label,
            (det.start_point[1] + 10, det.start_point[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
        )
    
    # Save
    cv2.imwrite(output_path, img_vis)


def detect_and_visualize(image_path: str, 
                        output_path: str,
                        detector: Optional[StreakDetector] = None) -> List[Detection]:
    """
    Convenience function to detect and visualize in one call.
    
    Args:
        image_path: Input image path
        output_path: Output visualization path
        detector: StreakDetector instance (creates default if None)
        
    Returns:
        List of detections
    """
    if detector is None:
        detector = StreakDetector()
    
    detections = detector.detect(image_path)
    visualize_detections(image_path, detections, output_path)
    
    return detections


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the streak detector.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python streak_detector.py <image_path> [output_path]")
        print("\nExample:")
        print("  python streak_detector.py debris_dataset/images/debris_0000.png output.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "detection_result.png"
    
    # Create detector
    detector = StreakDetector(
        threshold=0.4,
        min_length=40,
        max_gap=10,
        hough_threshold=20
    )
    
    # Detect streaks
    print(f"Processing: {image_path}")
    detections = detector.detect(image_path)
    
    # Print results
    print(f"\nFound {len(detections)} streaks:")
    for i, det in enumerate(detections, 1):
        print(f"  {i}. Length: {det.length}px, "
              f"Angle: {det.angle:.1f}°, "
              f"Confidence: {det.confidence:.2f}")
    
    # Visualize
    visualize_detections(image_path, detections, output_path)
    print(f"\nVisualization saved to: {output_path}")