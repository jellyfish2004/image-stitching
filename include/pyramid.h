#pragma once

#include "image.h"
#include <vector>

// Gaussian Pyramid
// gauss.octaves[o][s] stores the Gaussian-blurred image at octave o, scale s.
// Each octave has (scales_per_octave + 3) images.
// Octave 0 is at full resolution; each subsequent octave is downsampled 2x.
struct GaussianPyramid {
    int num_octaves;
    int scales_per_octave;      // S  (e.g. 5)
    float sigma0;               // base sigma (e.g. 1.6)
    // octaves[o][s], where s ∈ [0, S+2]
    std::vector<std::vector<Image>> octaves;
};

// Difference-of-Gaussian Pyramid
// dog.octaves[o][s] = gauss[o][s+1] − gauss[o][s]
// Each octave has (scales_per_octave + 2) DoG images.
struct DoGPyramid {
    // octaves[o][s], where s ∈ [0, S+1]
    std::vector<std::vector<Image>> octaves;
};

// Builders
// Build Gaussian pyramid from a single-channel (grayscale) base image.
// Parallelization:
//   • Each convolution is parallel per-pixel (separable rows/cols).
//   • DoG subtraction is embarrassingly parallel per-pixel.
//   • Octaves are sequential (each depends on downsampled previous).
GaussianPyramid build_gaussian_pyramid(const Image& base,
                                        int num_octaves,
                                        int scales_per_octave,
                                        float sigma0);

DoGPyramid build_dog_pyramid(const GaussianPyramid& gauss);
