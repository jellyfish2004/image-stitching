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

#pragma acc routine seq
static inline float clamped_at_ptr(const float* data, int w, int h, int c_channels, int x, int y, int c) {
    x = (x < 0) ? 0 : ((x >= w) ? w - 1 : x);
    y = (y < 0) ? 0 : ((y >= h) ? h - 1 : y);
    return data[(y * w + x) * c_channels + c];
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

    int w = img.width;
    int h = img.height;
    int c_channels = img.channels;
    const float* in_data = img.data.data();
    float* out_data = dst.data.data();
    const float* k_data = kernel.data();
    int k_size = kernel.size();

    // Parallelizable: each (y, x) is independent
    #pragma acc parallel loop collapse(3) copyin(in_data[0:w*h*c_channels], k_data[0:k_size]) copyout(out_data[0:w*h*c_channels])
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int ch = 0; ch < c_channels; ++ch) {
                float val = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    val += clamped_at_ptr(in_data, w, h, c_channels, x + k, y, ch) * k_data[k + radius];
                }
                out_data[(y * w + x) * c_channels + ch] = val;
            }
        }
    }

    return dst;
}

Image convolve_vertical(const Image& img, const std::vector<float>& kernel) {
    int radius = static_cast<int>(kernel.size()) / 2;
    Image dst(img.width, img.height, img.channels);

    int w = img.width;
    int h = img.height;
    int c_channels = img.channels;
    const float* in_data = img.data.data();
    float* out_data = dst.data.data();
    const float* k_data = kernel.data();
    int k_size = kernel.size();

    // Parallelizable: each (y, x) is independent
    #pragma acc parallel loop collapse(3) copyin(in_data[0:w*h*c_channels], k_data[0:k_size]) copyout(out_data[0:w*h*c_channels])
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int ch = 0; ch < c_channels; ++ch) {
                float val = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    val += clamped_at_ptr(in_data, w, h, c_channels, x, y + k, ch) * k_data[k + radius];
                }
                out_data[(y * w + x) * c_channels + ch] = val;
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

    int src_w = img.width;
    int c_channels = img.channels;
    const float* in_data = img.data.data();
    float* out_data = dst.data.data();

    // Parallelizable: per output pixel
    #pragma acc parallel loop collapse(3) copyin(in_data[0:src_w*img.height*c_channels]) copyout(out_data[0:w*h*c_channels])
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < c_channels; ++c) {
                out_data[(y * w + x) * c_channels + c] = in_data[((y * 2) * src_w + (x * 2)) * c_channels + c];
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

    int src_w = img.width;
    int src_h = img.height;
    int c_channels = img.channels;
    const float* in_data = img.data.data();
    float* out_data = dst.data.data();

    // Parallelizable: per output pixel
    #pragma acc parallel loop collapse(3) copyin(in_data[0:src_w*src_h*c_channels]) copyout(out_data[0:w*h*c_channels])
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < c_channels; ++c) {
                // Map output (x,y) back to source coords
                // idea - center of output pixel: (x+0.5)/2, then -0.5 to move back from the center of input to the actual coordinate position
                float sx = (x + 0.5f) / 2.0f - 0.5f;
                float sy = (y + 0.5f) / 2.0f - 0.5f;

                int x0 = (int)floorf(sx);
                x0 = x0 < 0 ? 0 : (x0 >= src_w ? src_w - 1 : x0);
                int y0 = (int)floorf(sy);
                y0 = y0 < 0 ? 0 : (y0 >= src_h ? src_h - 1 : y0);
                
                int x1 = x0 + 1;
                x1 = x1 < 0 ? 0 : (x1 >= src_w ? src_w - 1 : x1);
                int y1 = y0 + 1;
                y1 = y1 < 0 ? 0 : (y1 >= src_h ? src_h - 1 : y1);

                float fx = sx - floorf(sx);
                float fy = sy - floorf(sy);

                float v00 = in_data[(y0 * src_w + x0) * c_channels + c];
                float v10 = in_data[(y0 * src_w + x1) * c_channels + c];
                float v01 = in_data[(y1 * src_w + x0) * c_channels + c];
                float v11 = in_data[(y1 * src_w + x1) * c_channels + c];

                float v = v00 * (1.0f - fx) * (1.0f - fy)
                        + v10 *        fx   * (1.0f - fy)
                        + v01 * (1.0f - fx) *        fy
                        + v11 *        fx   *        fy;
                out_data[(y * w + x) * c_channels + c] = v;
            }
        }
    }

    return dst;
}
