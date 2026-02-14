Tasks:
    Decompress the jpeg
    Convert to pbm
    Run image processing on pbm

B
B
B
B
BB
B
B
Here's the code that Gemini suggested as a starting point:

```
#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h" // A single tiny header file

void process_image(const char* filename) {
    int width, height, channels;
    // Load image into a simple byte array (R,G,B,R,G,B...)
    unsigned char *img = stbi_load(filename, &width, &height, &channels, 3);
    
    if (img == NULL) return;

    // Create a buffer for the 1-bit output (or just reuse the input)
    for (int i = 0; i < width * height; i++) {
        int r = img[i * 3];
        int g = img[i * 3 + 1];
        int b = img[i * 3 + 2];

        // 1. Convert to Grayscale (Luminance)
        float gray = (0.299f * r) + (0.587f * g) + (0.114f * b);

        // 2. Threshold to Black (0) or White (255)
        unsigned char bw = (gray > 128) ? 255 : 0;

        // Store result (here we just overwrite the first channel for simplicity)
        img[i] = bw;
    }

    // Now 'img' contains your processed data. 
    // On a CubeSat, you'd likely transmit this raw or save it.
    stbi_image_free(img);
}
```
