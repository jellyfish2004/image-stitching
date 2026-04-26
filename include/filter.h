#pragma once

#include "image.h"
#include <vector>

// kernel generation
std::vector<float> gaussian_kernel_1d(float sigma);

// convolution (separable)
Image convolve_horizontal(const Image& img, const std::vector<float>& kernel);
Image convolve_vertical(const Image& img, const std::vector<float>& kernel);

// full blur (calls both)
Image gaussian_blur(const Image& img, float sigma);