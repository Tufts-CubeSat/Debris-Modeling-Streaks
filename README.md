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

## Long term plan:
1. Develop the program in Python
2. Write a new version in c/c++ or rust
