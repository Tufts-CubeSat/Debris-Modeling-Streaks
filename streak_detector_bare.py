import math
import sys
# You can download 'png.py' (PyPNG) and put it in the same folder. 
# It is a single file, ~60KB.
import png 

class StreakDetector:
    def __init__(self, threshold=120, min_length=30, min_aspect=2.5):
        self.threshold = threshold
        self.min_length = min_length
        self.min_aspect = min_aspect

    def load_png_to_gray(self, file_path):
        """Reads a PNG and converts to a 2D grayscale list."""
        r = png.Reader(file_path)
        width, height, pixels, meta = r.read()
        
        # Handle RGB or RGBA from GoPro by averaging or taking one channel
        planes = meta['planes']
        gray_img = []
        for row in pixels:
            # Step through row based on number of color channels
            gray_row = [row[i] if planes == 1 else (row[i] + row[i+1] + row[i+2]) // 3 
                        for i in range(0, len(row), planes)]
            gray_img.append(gray_row)
        return gray_img, width, height

    def detect(self, image_path):
        pixels, width, height = self.load_png_to_gray(image_path)
        visited = [[False for _ in range(width)] for _ in range(height)]
        detections = []

        for y in range(height):
            for x in range(width):
                # Look for pixels above brightness threshold
                if pixels[y][x] > self.threshold and not visited[y][x]:
                    component = self._flood_fill(pixels, x, y, visited, width, height)
                    
                    if len(component) < self.min_length:
                        continue

                    # Geometric filtering
                    start, end, length = self._analyze_component(component)
                    
                    # Basic aspect ratio check (bounding box)
                    min_x = min(p[0] for p in component)
                    max_x = max(p[0] for p in component)
                    min_y = min(p[1] for p in component)
                    max_y = max(p[1] for p in component)
                    
                    bw = (max_x - min_x) + 1
                    bh = (max_y - min_y) + 1
                    aspect = max(bw, bh) / max(min(bw, bh), 1)

                    if length >= self.min_length and aspect >= self.min_aspect:
                        detections.append({
                            'start': start,
                            'end': end,
                            'length': round(length, 2)
                        })
        return detections

    def _flood_fill(self, pixels, x, y, visited, width, height):
        """Standard 8-connectivity flood fill."""
        component = []
        stack = [(x, y)]
        while stack:
            curr_x, curr_y = stack.pop()
            if 0 <= curr_x < width and 0 <= curr_y < height:
                if not visited[curr_y][curr_x] and pixels[curr_y][curr_x] > self.threshold:
                    visited[curr_y][curr_x] = True
                    component.append((curr_x, curr_y))
                    # Check all 8 neighbors
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0: continue
                            stack.append((curr_x + dx, curr_y + dy))
        return component

    def _analyze_component(self, component):
        """Finds the endpoints by looking for the two points furthest apart."""
        if not component: return (0,0), (0,0), 0
        
        # For efficiency on a slow CPU, we sample points if the blob is huge
        step = max(1, len(component) // 100)
        sample = component[::step]
        
        max_dist_sq = -1
        p1, p2 = sample[0], sample[0]
        
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                d_sq = (sample[i][0] - sample[j][0])**2 + (sample[i][1] - sample[j][1])**2
                if d_sq > max_dist_sq:
                    max_dist_sq = d_sq
                    p1, p2 = sample[i], sample[j]
        
        return p1, p2, math.sqrt(max_dist_sq)

# Usage
if __name__ == "__main__":
    detector = StreakDetector(threshold=150)
    found = detector.detect("gopro_capture.png")
    print(f"Detected {len(found)} streaks.")
    for s in found:
        print(f"Streak: {s['start']} to {s['end']} | Length: {s['length']}px")