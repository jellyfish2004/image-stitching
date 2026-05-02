#pragma once

#include "image.h"
#include <vector>

// Returns a 1D Gaussian kernel with radius = ceil(3*sigma).
std::vector<float> gaussian_kernel_1d(float sigma);

// Each pass is independent per-row / per-column, trivially parallel.
Image convolve_horizontal(const Image& img, const std::vector<float>& kernel);
Image convolve_vertical  (const Image& img, const std::vector<float>& kernel);

// Full Gaussian Blur
Image gaussian_blur(const Image& img, float sigma);

// Downsample by factor 2 (nearest-neighbour subsampling).
// Parallelizable: per output pixel.
Image downsample_2x(const Image& img);

// Upsample by factor 2 (bilinear interpolation).
// Parallelizable: per output pixel.
Image upsample_2x(const Image& img);