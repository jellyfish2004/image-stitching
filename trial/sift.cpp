#include "sift.h"
#include "filter.h"

#include <cmath>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// pixel access, single channel
static inline float cat(const Image& img, int x, int y) {
    x = std::clamp(x, 0, img.width  - 1);
    y = std::clamp(y, 0, img.height - 1);
    return img.data[y * img.width + x];
}


// Check if (x,y) at scale s in octave o is a 3D local extremum
// among its 26 neighbours (3×3×3 cube minus center).
static bool is_extremum(const DoGPyramid& dog, int o, int s,
                         int x, int y, float contrast_thresh) {
    float v = cat(dog.octaves[o][s], x, y);
    if (std::fabs(v) <= contrast_thresh) return false;

    bool is_max = true, is_min = true;

    for (int ds = -1; ds <= 1 && (is_max || is_min); ++ds) {
        for (int dy = -1; dy <= 1 && (is_max || is_min); ++dy) {
            for (int dx = -1; dx <= 1 && (is_max || is_min); ++dx) {
                if (ds == 0 && dy == 0 && dx == 0) continue;
                float n = cat(dog.octaves[o][s + ds], x + dx, y + dy);
                if (n >= v) is_max = false;
                if (n <= v) is_min = false;
            }
        }
    }

    return is_max || is_min;
}

// Reject edge responses using the 2D Hessian determinant ratio test.
// Returns true if the keypoint is NOT on an edge (i.e., it should be kept).
static bool passes_edge_test(const Image& dog_layer, int x, int y,
                              float edge_thresh) {
    // use central diff approximations for hessian
    float Dxx = cat(dog_layer, x + 1, y) + cat(dog_layer, x - 1, y)
              - 2.0f * cat(dog_layer, x, y);
    float Dyy = cat(dog_layer, x, y + 1) + cat(dog_layer, x, y - 1)
              - 2.0f * cat(dog_layer, x, y);
    float Dxy = (cat(dog_layer, x + 1, y + 1) - cat(dog_layer, x - 1, y + 1)
               - cat(dog_layer, x + 1, y - 1) + cat(dog_layer, x - 1, y - 1))
              / 4.0f;

    float trace = Dxx + Dyy;
    float det   = Dxx * Dyy - Dxy * Dxy;
    float r     = edge_thresh;

    if (det <= 0.0f) return false;
    return (trace * trace / det) < ((r + 1.0f) * (r + 1.0f) / r);
}

std::vector<Keypoint> detect_extrema(const DoGPyramid& dog,
                                      int scales_per_octave,
                                      float sigma0,
                                      float contrast_thresh,
                                      float edge_thresh) {
    (void)scales_per_octave;  // used implicitly via DoG structure
    std::vector<Keypoint> keypoints;

    int num_octaves = static_cast<int>(dog.octaves.size());

    // Parallelizable: the (o, s, y, x) loop nest.
    // Each pixel check is independent. Use thread-local vectors + merge.
    for (int o = 0; o < num_octaves; ++o) {
        int w = dog.octaves[o][0].width;
        int h = dog.octaves[o][0].height;
        float octave_scale = static_cast<float>(1 << o);

        // s ranges over the "interior" DoG images where we have
        // scale neighbours above and below: [1, num_dog-2]
        int num_dog = static_cast<int>(dog.octaves[o].size());

        for (int s = 1; s < num_dog - 1; ++s) {
            for (int y = 1; y < h - 1; ++y) {
                for (int x = 1; x < w - 1; ++x) {
                    if (!is_extremum(dog, o, s, x, y, contrast_thresh))
                        continue;
                    if (!passes_edge_test(dog.octaves[o][s], x, y, edge_thresh))
                        continue;

                    Keypoint kp{};
                    kp.x         = static_cast<float>(x) * octave_scale;
                    kp.y         = static_cast<float>(y) * octave_scale;
                    kp.scale     = sigma0 * std::pow(SIFT_K, static_cast<float>(s))
                                 * octave_scale;
                    kp.octave    = o;
                    kp.scale_idx = s;
                    kp.angle     = 0.0f;

                    keypoints.push_back(kp);
                }
            }
        }
    }

    return keypoints;
}

// Compute the dominant orientation for a single keypoint.
// Returns the peak angle in radians.
// steps"
// 1. compute gradient direction and magnitude around the keypoint
// 2. create bins based on orientation, use gaussian weighting based on the distance from it
// 3. smooth the histogram
// 4. find the dominant orientation
// essentially, find the dominant gradient direction
static float compute_single_orientation(const Image& blur, int x, int y,
                                         float sigma) {
    float hist[SIFT_ORI_BINS] = {};
    int rad = static_cast<int>(3.0f * SIFT_ORI_SIGMA_FACTOR * sigma);
    float weight_sigma = SIFT_ORI_SIGMA_FACTOR * sigma;

    for (int dy = -rad; dy <= rad; ++dy) {
        for (int dx = -rad; dx <= rad; ++dx) {
            float gx = cat(blur, x + dx + 1, y + dy) - cat(blur, x + dx - 1, y + dy);
            float gy = cat(blur, x + dx, y + dy + 1) - cat(blur, x + dx, y + dy - 1);
            float mag = std::sqrt(gx * gx + gy * gy);
            float ori = std::atan2(gy, gx);
            float w   = std::exp(-(dx * dx + dy * dy) /
                                  (2.0f * weight_sigma * weight_sigma));

            // [-pi, pi] -> [0, 2pi] -> [0, 1] -> [0, N_bins]. wrap around if required
            int bin = static_cast<int>((ori + static_cast<float>(M_PI)) /
                      (2.0f * static_cast<float>(M_PI)) * SIFT_ORI_BINS);
            if (bin >= SIFT_ORI_BINS) bin = 0;
            if (bin < 0) bin = SIFT_ORI_BINS - 1;

            hist[bin] += w * mag;
        }
    }

    // Smooth the histogram (6 passes of circular moving average)
    // essentially just conv with [0.25, 0.5, 0.25]
    for (int i = 0; i < 6; ++i) {
        float prev = hist[SIFT_ORI_BINS - 1];
        for (int b = 0; b < SIFT_ORI_BINS; ++b) {
            float tmp = hist[b];
            hist[b] = 0.25f * prev + 0.5f * hist[b]
                    + 0.25f * hist[(b + 1) % SIFT_ORI_BINS];
            prev = tmp;
        }
    }

    // Find peak bin
    int peak_bin = 0;
    for (int b = 1; b < SIFT_ORI_BINS; ++b) {
        if (hist[b] > hist[peak_bin]) peak_bin = b;
    }

    // bin+0.5 -> center of range of bin, * 2pi / N_bins -> angle range, -pi -> back to original angle
    return (static_cast<float>(peak_bin) + 0.5f) / SIFT_ORI_BINS
         * 2.0f * static_cast<float>(M_PI) - static_cast<float>(M_PI);
}

void assign_orientations(std::vector<Keypoint>& keypoints,
                          const GaussianPyramid& gauss) {
    // Parallelizable: each keypoint is independent.
    // For secondary-peak duplicates, would need thread-local vectors.
    for (auto& kp : keypoints) {
        int o = kp.octave;
        int s = kp.scale_idx;
        float octave_scale = static_cast<float>(1 << o);

        // Keypoint position in octave-local coordinates
        int lx = static_cast<int>(kp.x / octave_scale);
        int ly = static_cast<int>(kp.y / octave_scale);

        float local_sigma = kp.scale / octave_scale;

        kp.angle = compute_single_orientation(gauss.octaves[o][s],
                                               lx, ly, local_sigma);
    }
}


// Compute the 128-dim SIFT descriptor for a single keypoint.
static void compute_single_descriptor(const Image& blur,
                                       float x, float y,
                                       float scale, float angle,
                                       float* desc) {
    std::memset(desc, 0, SIFT_DESC_SIZE * sizeof(float));

    float cos_a = std::cos(-angle);
    float sin_a = std::sin(-angle);
    float sbins = scale * 3.0f;   // pixel spacing per histogram cell

    // in 4x4 grid, each cell has 4x4 pixels
    // for each point in this 16x16 area, in the direction of the gradient so that it's rotationally invariant,
    // we find the bin like before and add it. each cell in the 4x4 grid has 8 bins, so we get a 128 dim vector
    for (int gy = 0; gy < SIFT_DESC_GRID; ++gy) {
        for (int gx = 0; gx < SIFT_DESC_GRID; ++gx) {
            for (int sy = 0; sy < 4; ++sy) {
                for (int sx = 0; sx < 4; ++sx) {
                    // Sample point in rotated frame
                    float rx = ((gx - 1.5f) * 4 + sx) * sbins / 4.0f;
                    float ry = ((gy - 1.5f) * 4 + sy) * sbins / 4.0f;
                    float px = x + cos_a * rx - sin_a * ry;
                    float py = y + sin_a * rx + cos_a * ry;

                    float dx = cat(blur, static_cast<int>(px) + 1, static_cast<int>(py))
                             - cat(blur, static_cast<int>(px) - 1, static_cast<int>(py));
                    float dy = cat(blur, static_cast<int>(px), static_cast<int>(py) + 1)
                             - cat(blur, static_cast<int>(px), static_cast<int>(py) - 1);
                    float mag = std::sqrt(dx * dx + dy * dy);
                    float ori = std::atan2(dy, dx) - angle;

                    while (ori < 0.0f)                            ori += 2.0f * static_cast<float>(M_PI);
                    while (ori >= 2.0f * static_cast<float>(M_PI)) ori -= 2.0f * static_cast<float>(M_PI);

                    int bin = static_cast<int>(ori / (2.0f * static_cast<float>(M_PI))
                            * SIFT_DESC_HIST_BINS);
                    if (bin >= SIFT_DESC_HIST_BINS) bin = 0;

                    int didx = (gy * SIFT_DESC_GRID + gx) * SIFT_DESC_HIST_BINS + bin;
                    desc[didx] += mag;
                }
            }
        }
    }

    // L2-normalise
    float norm = 0.0f;
    for (int i = 0; i < SIFT_DESC_SIZE; ++i) norm += desc[i] * desc[i];
    norm = std::sqrt(norm) + 1e-7f;
    for (int i = 0; i < SIFT_DESC_SIZE; ++i) desc[i] /= norm;

    // Clamp at 0.2 and renormalise (illumination robustness)
    for (int i = 0; i < SIFT_DESC_SIZE; ++i) {
        if (desc[i] > 0.2f) desc[i] = 0.2f;
    }
    norm = 0.0f;
    for (int i = 0; i < SIFT_DESC_SIZE; ++i) norm += desc[i] * desc[i];
    norm = std::sqrt(norm) + 1e-7f;
    for (int i = 0; i < SIFT_DESC_SIZE; ++i) desc[i] /= norm;
}

void compute_descriptors(std::vector<Keypoint>& keypoints,
                          const GaussianPyramid& gauss) {
    // Parallelizable: each keypoint is independent.
    for (auto& kp : keypoints) {
        int o = kp.octave;
        int s = kp.scale_idx;
        float octave_scale = static_cast<float>(1 << o);

        float lx    = kp.x / octave_scale;
        float ly    = kp.y / octave_scale;
        float lsigma = kp.scale / octave_scale;

        compute_single_descriptor(gauss.octaves[o][s],
                                   lx, ly, lsigma,
                                   kp.angle, kp.desc);
    }
}

std::vector<Keypoint> sift_detect_and_describe(const Image& gray,
                                                int num_octaves,
                                                int scales_per_octave,
                                                float sigma0,
                                                float contrast_thresh,
                                                float edge_thresh) {
    printf("[SIFT] Building Gaussian pyramid (%d octaves, %d scales)...\n",
           num_octaves, scales_per_octave);
    GaussianPyramid gauss = build_gaussian_pyramid(gray, num_octaves,
                                                    scales_per_octave, sigma0);

    printf("[SIFT] Building DoG pyramid...\n");
    DoGPyramid dog = build_dog_pyramid(gauss);

    printf("[SIFT] Detecting extrema...\n");
    std::vector<Keypoint> keypoints = detect_extrema(dog, scales_per_octave,
                                                      sigma0, contrast_thresh,
                                                      edge_thresh);
    printf("[SIFT] Found %zu candidate keypoints.\n", keypoints.size());

    printf("[SIFT] Assigning orientations...\n");
    assign_orientations(keypoints, gauss);

    printf("[SIFT] Computing descriptors...\n");
    compute_descriptors(keypoints, gauss);

    printf("[SIFT] Final keypoint count: %zu\n", keypoints.size());
    return keypoints;
}
