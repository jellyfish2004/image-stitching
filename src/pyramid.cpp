#include "pyramid.h"
#include "filter.h"

#include <cmath>


GaussianPyramid build_gaussian_pyramid(const Image& base,
                                        int num_octaves,
                                        int scales_per_octave,
                                        float sigma0) {
    GaussianPyramid pyr;
    pyr.num_octaves       = num_octaves;
    pyr.scales_per_octave = scales_per_octave;
    pyr.sigma0            = sigma0;
    pyr.octaves.resize(num_octaves);

    // Scale factor between consecutive levels: k = 2^(1/S)
    const float k = std::pow(2.0f, 1.0f / static_cast<float>(scales_per_octave));

    // S+3 -> S+2 after DoG -> +2 is padding for top and bottom DoGs during comparison
    int num_scales = scales_per_octave + 3;

    // Octave 0
    pyr.octaves[0].resize(num_scales);
    pyr.octaves[0][0] = gaussian_blur(base, sigma0);

    for (int s = 1; s < num_scales; ++s) {
        float sig      = sigma0 * std::pow(k, static_cast<float>(s));
        float sig_prev = sigma0 * std::pow(k, static_cast<float>(s - 1));
        // G(s1) * G(s2) = G(sqrt(s1^2 + s2^2))
        float sig_inc  = std::sqrt(sig * sig - sig_prev * sig_prev);
        pyr.octaves[0][s] = gaussian_blur(pyr.octaves[0][s - 1], sig_inc);
    }

    // Higher octaves
    // Each octave's base = downsample of the S-th image from previous octave
    for (int o = 1; o < num_octaves; ++o) {
        pyr.octaves[o].resize(num_scales);
        pyr.octaves[o][0] = downsample_2x(pyr.octaves[o - 1][scales_per_octave]);

        for (int s = 1; s < num_scales; ++s) {
            float sig      = sigma0 * std::pow(k, static_cast<float>(s));
            float sig_prev = sigma0 * std::pow(k, static_cast<float>(s - 1));
            float sig_inc  = std::sqrt(sig * sig - sig_prev * sig_prev);
            pyr.octaves[o][s] = gaussian_blur(pyr.octaves[o][s - 1], sig_inc);
        }
    }

    return pyr;
}

// Build DoG Pyramid
// DoG is used to approximate LoG
DoGPyramid build_dog_pyramid(const GaussianPyramid& gauss) {
    DoGPyramid dog;
    dog.octaves.resize(gauss.num_octaves);

    for (int o = 0; o < gauss.num_octaves; ++o) {
        int num_dog = static_cast<int>(gauss.octaves[o].size()) - 1;
        dog.octaves[o].resize(num_dog);

        for (int s = 0; s < num_dog; ++s) {
            const Image& g_hi = gauss.octaves[o][s + 1];
            const Image& g_lo = gauss.octaves[o][s];

            Image diff(g_lo.width, g_lo.height, 1);

            // Parallelizable: per-pixel subtraction, embarrassingly parallel
            int n = g_lo.width * g_lo.height;
            for (int i = 0; i < n; ++i) {
                diff.data[i] = g_hi.data[i] - g_lo.data[i];
            }

            dog.octaves[o][s] = std::move(diff);
        }
    }

    return dog;
}
