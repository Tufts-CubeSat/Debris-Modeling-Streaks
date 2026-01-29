# Debris-Modeling-Streaks

## V.1.
- Convert pixels to 1 or 0 (black or white)
- Identify streaks

### How to Use:
- **Generate a Single Preview:** Run `python generate_dataset.py --preview-only`. This generates preview.png, showing the image side-by-side with an annotated version. Use this to visually verify that the streak intensity and star density look realistic enough.
- **Generate a Full Dataset:** Run `python generate_dataset.py` for a default of 100 images, or something more specific like `python generate_dataset.py --num-images 500 --output-dir my_debris_data` to create 
  - /images/: 16-bit PNG files of the star fields.
  - /labels/: JSON files containing the ground truth for every streak.
- **Adjusting Realism:** Modify the generate_image parameters in `generate_dataset.py`
- **Anallyzing Results** Run the evaluation using `python run_evaluation.py --dataset-dir debris_dataset`. That will:
  - Import StreakDetector from streak_detector.py
  - Run it on all images in debris_dataset/images/
  - Compare detections to ground truth in debris_dataset/labels/
  - Print precision, recall, F1 score
  - Save visualizations to debris_dataset/evaluation_results/
  - Save detailed results to debris_dataset/evaluation_results.json

## Notes:
- `streak_detector.py`: a simplified cv2 and numpy based detector program that can be run with `run_evaluation.py` to assess & compare streak detection to json ground-truth labels
- `streak_detector_bare.py`: a prototype for what an actual on-board program could look like
  - By using pypng (about 60KB) and standard Python lists, we avoid the 50MB+ overhead of OpenCV and NumPy.
  - The load_png_to_gray function handles standard GoPro RGB output by averaging the color channels into a single "Black and White" intensity value.
  - The _analyze_component function includes a sampling step (len(component) // 100) to ensure that even if a streak is very large, the endpoint calculation won't hang your basic processor.

## Long term plan:
1. Develop the program in Python
2. Write a new version in c/c++ or rust
