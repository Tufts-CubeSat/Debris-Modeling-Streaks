"""
Semisynthetic Space Debris Detection Data Generator

Generates realistic astronomical images with space debris streaks for training
detection models.
"""

import numpy as np
from typing import Tuple, Optional, List
import cv2


class DebrisDataGenerator:
    """Generate semisynthetic images of space debris as gray streaks."""

    def __init__(self, image_size: Tuple[int, int] = (1024, 1024), seed: Optional[int] = None):
        """
        Initialize the debris data generator.

        Args:
            image_size: (height, width) of generated images
            seed: Random seed for reproducibility
        """
        self.image_size = image_size
        self.rng = np.random.RandomState(seed)

    def generate_star_field(self, num_stars: int = 500,
                           star_intensity_range: Tuple[float, float] = (0.3, 1.0)) -> np.ndarray:
        """
        Generate a realistic star field background.

        Args:
            num_stars: Number of stars to generate
            star_intensity_range: Min and max intensity for stars (0-1)

        Returns:
            Star field image as float array [0, 1]
        """
        img = np.zeros(self.image_size, dtype=np.float32)

        for _ in range(num_stars):
            # Random position
            y = self.rng.randint(0, self.image_size[0])
            x = self.rng.randint(0, self.image_size[1])

            # Random intensity with power-law distribution (more dim stars)
            intensity = self.rng.beta(2, 5)
            intensity = star_intensity_range[0] + intensity * (star_intensity_range[1] - star_intensity_range[0])

            # Random size (most stars are point sources, some are slightly larger)
            size = self.rng.choice([1, 2, 3], p=[0.7, 0.25, 0.05])

            if size == 1:
                img[y, x] = intensity
            else:
                # Create small Gaussian blob for larger stars
                y_min = max(0, y - size)
                y_max = min(self.image_size[0], y + size + 1)
                x_min = max(0, x - size)
                x_max = min(self.image_size[1], x + size + 1)

                yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
                gaussian = np.exp(-((yy - y)**2 + (xx - x)**2) / (2 * (size/2)**2))
                img[y_min:y_max, x_min:x_max] = np.maximum(
                    img[y_min:y_max, x_min:x_max],
                    gaussian * intensity
                )

        return img

    def add_debris_streak(self, img: np.ndarray,
                         start_point: Optional[Tuple[int, int]] = None,
                         angle: Optional[float] = None,
                         length: Optional[int] = None,
                         thickness: Optional[int] = None,
                         intensity: Optional[float] = None,
                         motion_blur: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Add a debris streak to the image.

        Args:
            img: Input image
            start_point: (y, x) starting position, random if None
            angle: Angle in degrees (0-360), random if None
            length: Length of streak in pixels, random if None
            thickness: Thickness of streak, random if None
            intensity: Base intensity of streak (0-1), random if None
            motion_blur: Apply motion blur effect for realism

        Returns:
            Modified image and dictionary with streak parameters
        """
        h, w = self.image_size

        # Generate random parameters if not provided
        if start_point is None:
            margin = 100  # Keep streaks away from edges
            start_point = (
                self.rng.randint(margin, h - margin),
                self.rng.randint(margin, w - margin)
            )

        if angle is None:
            angle = self.rng.uniform(0, 360)

        if length is None:
            length = self.rng.randint(50, 300)

        if thickness is None:
            thickness = self.rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])

        if intensity is None:
            # Gray streaks - typically dimmer than bright stars
            intensity = self.rng.uniform(0.4, 0.8)

        # Calculate end point
        angle_rad = np.radians(angle)
        end_point = (
            int(start_point[0] + length * np.sin(angle_rad)),
            int(start_point[1] + length * np.cos(angle_rad))
        )

        # Create streak on separate layer
        streak_layer = np.zeros_like(img)

        # Draw the streak with varying intensity along length (fade at ends)
        num_points = max(length, 100)
        for i in range(num_points):
            t = i / num_points
            y = int(start_point[0] + t * (end_point[0] - start_point[0]))
            x = int(start_point[1] + t * (end_point[1] - start_point[1]))

            if 0 <= y < h and 0 <= x < w:
                # Fade intensity at the ends
                fade = 1.0 - abs(2 * t - 1) ** 1.5
                local_intensity = intensity * fade

                # Draw with thickness
                for dy in range(-thickness, thickness + 1):
                    for dx in range(-thickness, thickness + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            # Gaussian falloff from center
                            dist = np.sqrt(dy**2 + dx**2)
                            if thickness > 0:
                                weight = np.exp(-(dist**2) / (2 * thickness**2))
                            else:
                                weight = 1.0 if dist == 0 else 0.0
                            streak_layer[ny, nx] = max(
                                streak_layer[ny, nx],
                                local_intensity * weight
                            )

        # Apply motion blur if requested
        if motion_blur and length > 10:
            blur_size = min(15, max(3, thickness * 2))
            kernel = np.zeros((blur_size, blur_size))
            # Create directional blur kernel
            kernel_angle = angle_rad + np.pi / 2
            for i in range(blur_size):
                ky = int(blur_size // 2 + (i - blur_size // 2) * np.sin(kernel_angle) * 0.5)
                kx = int(blur_size // 2 + (i - blur_size // 2) * np.cos(kernel_angle) * 0.5)
                if 0 <= ky < blur_size and 0 <= kx < blur_size:
                    kernel[ky, kx] += 1
            kernel = kernel / (kernel.sum() + 1e-10)
            streak_layer = cv2.filter2D(streak_layer, -1, kernel)

        # Composite the streak
        result = np.maximum(img, streak_layer)

        # Store parameters
        params = {
            'start_point': start_point,
            'end_point': end_point,
            'angle': angle,
            'length': length,
            'thickness': thickness,
            'intensity': intensity
        }

        return result, params

    def add_curved_debris_streak(self, img: np.ndarray,
                                 start_point: Optional[Tuple[int, int]] = None,
                                 curve_amount: Optional[float] = None,
                                 length: Optional[int] = None,
                                 thickness: Optional[int] = None,
                                 intensity: Optional[float] = None) -> Tuple[np.ndarray, dict]:
        """
        Add a curved debris streak (e.g., tumbling debris or long exposure).

        Args:
            img: Input image
            start_point: Starting position
            curve_amount: How much the streak curves (-1 to 1)
            length: Approximate length
            thickness: Thickness of streak
            intensity: Base intensity

        Returns:
            Modified image and parameters
        """
        h, w = self.image_size

        if start_point is None:
            margin = 100
            start_point = (
                self.rng.randint(margin, h - margin),
                self.rng.randint(margin, w - margin)
            )

        if curve_amount is None:
            curve_amount = self.rng.uniform(-0.3, 0.3)

        if length is None:
            length = self.rng.randint(80, 350)

        if thickness is None:
            thickness = self.rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])

        if intensity is None:
            intensity = self.rng.uniform(0.4, 0.8)

        # Generate curved path
        streak_layer = np.zeros_like(img)
        base_angle = self.rng.uniform(0, 2 * np.pi)

        num_points = max(length * 2, 100)
        for i in range(num_points):
            t = i / num_points

            # Parametric curve with varying curvature
            angle = base_angle + curve_amount * np.sin(t * np.pi)
            distance = t * length

            y = int(start_point[0] + distance * np.sin(angle))
            x = int(start_point[1] + distance * np.cos(angle))

            if 0 <= y < h and 0 <= x < w:
                # Fade at ends
                fade = 1.0 - abs(2 * t - 1) ** 1.5
                local_intensity = intensity * fade

                for dy in range(-thickness, thickness + 1):
                    for dx in range(-thickness, thickness + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            dist = np.sqrt(dy**2 + dx**2)
                            weight = np.exp(-(dist**2) / (2 * max(thickness, 1)**2))
                            streak_layer[ny, nx] = max(
                                streak_layer[ny, nx],
                                local_intensity * weight
                            )

        result = np.maximum(img, streak_layer)

        params = {
            'start_point': start_point,
            'curve_amount': curve_amount,
            'length': length,
            'thickness': thickness,
            'intensity': intensity,
            'type': 'curved'
        }

        return result, params

    def add_noise(self, img: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
        """
        Add realistic sensor noise to the image.

        Args:
            img: Input image
            noise_level: Standard deviation of Gaussian noise

        Returns:
            Noisy image
        """
        noise = self.rng.normal(0, noise_level, img.shape)
        noisy = img + noise
        return np.clip(noisy, 0, 1)

    def add_vignetting(self, img: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """
        Add vignetting effect (darkening at edges).

        Args:
            img: Input image
            strength: Strength of vignetting (0-1)

        Returns:
            Image with vignetting
        """
        h, w = self.image_size
        y, x = np.ogrid[:h, :w]

        # Distance from center
        center_y, center_x = h / 2, w / 2
        max_dist = np.sqrt(center_y**2 + center_x**2)
        dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)

        # Radial falloff
        vignette = 1 - strength * (dist / max_dist) ** 2

        return img * vignette

    def generate_image(self, num_debris: int = 1,
                      num_stars: int = 500,
                      curved_debris_prob: float = 0.2,
                      noise_level: float = 0.02,
                      vignetting_strength: float = 0.2) -> Tuple[np.ndarray, List[dict]]:
        """
        Generate a complete semisynthetic image with debris streaks.

        Args:
            num_debris: Number of debris streaks to add
            num_stars: Number of background stars
            curved_debris_prob: Probability of curved vs straight streaks
            noise_level: Amount of sensor noise
            vignetting_strength: Strength of edge darkening

        Returns:
            Generated image (float32, 0-1 range) and list of debris parameters
        """
        # Start with star field
        img = self.generate_star_field(num_stars=num_stars)

        # Add debris streaks
        debris_params = []
        for _ in range(num_debris):
            if self.rng.random() < curved_debris_prob:
                img, params = self.add_curved_debris_streak(img)
            else:
                img, params = self.add_debris_streak(img)
            debris_params.append(params)

        # Add realistic effects
        img = self.add_vignetting(img, strength=vignetting_strength)
        img = self.add_noise(img, noise_level=noise_level)

        return img, debris_params

    def save_image(self, img: np.ndarray, filename: str, format: str = 'png'):
        """
        Save generated image to file.

        Args:
            img: Image to save (0-1 float range)
            filename: Output filename
            format: Image format ('png' or 'fits')
        """
        if format == 'png':
            # Convert to 16-bit for better dynamic range
            img_16bit = (img * 65535).astype(np.uint16)
            cv2.imwrite(filename, img_16bit)
        elif format == 'fits':
            try:
                from astropy.io import fits
                hdu = fits.PrimaryHDU(img)
                hdu.writeto(filename, overwrite=True)
            except ImportError:
                raise ImportError("astropy required for FITS format. Install with: pip install astropy")
        else:
            raise ValueError(f"Unsupported format: {format}")
