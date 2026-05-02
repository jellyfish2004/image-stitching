#include "filter.h"
#include <cmath>
#include <algorithm>

// gaussian kernels, convolutions, up/down sampling
// parallel - openacc convolutions and up/down sampling

std::vector<float> gaussian_kernel_1d(float sigma) {
    int radius = static_cast<int>(std::ceil(3.0f * sigma));
    int size   = 2 * radius + 1;

    std::vector<float> kernel(size);
    float sum = 0.0f;

    for (int i = 0; i < size; ++i) {
        // eg. rad=3, -3, -2, -1, 0, 1, 2, 3
        float x = static_cast<float>(i - radius);
        kernel[i] = std::exp(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel[i];
    }

    // Normalise
    for (int i = 0; i < size; ++i) {
        kernel[i] /= sum;
    }

    return kernel;
}

// index - clamping for padding with border values
static inline float clamped_at(const Image& img, int x, int y, int c) {
    x = std::clamp(x, 0, img.width  - 1);
    y = std::clamp(y, 0, img.height - 1);
    return img.data[(y * img.width + x) * img.channels + c];
}


Image convolve_horizontal(const Image& img, const std::vector<float>& kernel) {
    int radius = static_cast<int>(kernel.size()) / 2;
    Image dst(img.width, img.height, img.channels);

    // Parallelizable: each (y, x) is independent
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            for (int ch = 0; ch < img.channels; ++ch) {
                float val = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    val += clamped_at(img, x + k, y, ch) * kernel[k + radius];
                }
                dst.at(x, y, ch) = val;
            }
        }
    }

    return dst;
}

Image convolve_vertical(const Image& img, const std::vector<float>& kernel) {
    int radius = static_cast<int>(kernel.size()) / 2;
    Image dst(img.width, img.height, img.channels);

    // Parallelizable: each (y, x) is independent
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            for (int ch = 0; ch < img.channels; ++ch) {
                float val = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    val += clamped_at(img, x, y + k, ch) * kernel[k + radius];
                }
                dst.at(x, y, ch) = val;
            }
        }
    }

    return dst;
}

Image gaussian_blur(const Image& img, float sigma) {
    auto kernel = gaussian_kernel_1d(sigma);
    Image tmp = convolve_horizontal(img, kernel);
    return convolve_vertical(tmp, kernel);
}


Image downsample_2x(const Image& img) {
    int w = img.width  / 2;
    int h = img.height / 2;
    Image dst(w, h, img.channels);

    // Parallelizable: per output pixel
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < img.channels; ++c) {
                dst.at(x, y, c) = img.at(x * 2, y * 2, c); // sampling every other pixel
            }
        }
    }

    return dst;
}

// bilinear interp
Image upsample_2x(const Image& img) {
    int w = img.width  * 2;
    int h = img.height * 2;
    Image dst(w, h, img.channels);

    // Parallelizable: per output pixel
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // Map output (x,y) back to source coords
            // idea - center of output pixel: (x+0.5)/2, then -0.5 to move back from the center of input to the actual coordinate position
            float sx = (x + 0.5f) / 2.0f - 0.5f;
            float sy = (y + 0.5f) / 2.0f - 0.5f;

            int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, img.width  - 1);
            int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, img.height - 1);
            int x1 = std::clamp(x0 + 1, 0, img.width  - 1);
            int y1 = std::clamp(y0 + 1, 0, img.height - 1);

            float fx = sx - std::floor(sx); // fractions used for interp
            float fy = sy - std::floor(sy);

            for (int c = 0; c < img.channels; ++c) {
                float v = img.at(x0, y0, c) * (1 - fx) * (1 - fy)
                        + img.at(x1, y0, c) *      fx  * (1 - fy)
                        + img.at(x0, y1, c) * (1 - fx) *      fy
                        + img.at(x1, y1, c) *      fx  *      fy;
                dst.at(x, y, c) = v;
            }
        }
    }

    return dst;
}
