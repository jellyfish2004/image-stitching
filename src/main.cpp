#include "image.h"
#include "pyramid.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s image1\n", argv[0]);
        return 1;
    }

    // ── 1. Load images ─────────────────────────────────────────
    printf("=== Loading images ===\n");
    Image img1 = load_image(argv[1]);
    printf("Image 1: %d x %d x %d\n", img1.width, img1.height, img1.channels);

    printf("--- Processing Image 1 ---\n");
    Image gray1 = rgb_to_grayscale(img1);

    GaussianPyramid gp = build_gaussian_pyramid(gray1, 4, 6, 1.6f);
    DoGPyramid dogp = build_dog_pyramid(gp);

    // save DoG images
    for(int o = 0; o < dogp.octaves.size(); ++o) {
        for(int s = 0; s < dogp.octaves[o].size(); ++s) {
            char filename[64];
            std::sprintf(filename, "dog_%d_%d.png", o, s);
            save_image(filename, dogp.octaves[o][s]);
        }
    }

    return 0;
}