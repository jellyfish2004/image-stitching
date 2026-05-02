#pragma once

#include "image.h"
#include "pyramid.h"
#include <vector>

// ── Constants (defaults match serial.c) ────────────────────────
constexpr int   SIFT_DESC_SIZE       = 128;   // 4 × 4 × 8
constexpr int   SIFT_DESC_HIST_BINS  = 8;
constexpr int   SIFT_DESC_GRID      = 4;
constexpr int   SIFT_ORI_BINS       = 36;
constexpr float SIFT_ORI_SIGMA_FACTOR = 1.5f;
constexpr float SIFT_K              = 1.4142f; // sqrt(2)

// ── Keypoint ───────────────────────────────────────────────────
struct Keypoint {
    float x, y;                     // position in original image coords
    float scale;                    // sigma at detection
    float angle;                    // dominant orientation (radians)
    int   octave;                   // octave index (for back-reference)
    int   scale_idx;                // scale index within octave
    float desc[SIFT_DESC_SIZE];     // L2-normalised descriptor
};

// ── Full SIFT Pipeline ─────────────────────────────────────────
// Detects keypoints, assigns orientations, computes descriptors.
// This is the main entry point — calls the sub-steps below.
//
// Parallelization (high level):
//   • Two images can run through this entirely in parallel (OpenMP sections / MPI ranks).
std::vector<Keypoint> sift_detect_and_describe(
    const Image& gray,
    int   num_octaves        = 4,
    int   scales_per_octave  = 5,
    float sigma0             = 1.6f,
    float contrast_thresh    = 0.015f,
    float edge_thresh        = 10.0f);

// ── Sub-steps (exposed for testing & fine-grained parallelization) ──

// Detect 3D extrema in DoG pyramid and filter by contrast + edge response.
// Parallelizable: per-pixel within each octave/scale.
std::vector<Keypoint> detect_extrema(
    const DoGPyramid& dog,
    int   scales_per_octave,
    float sigma0,
    float contrast_thresh,
    float edge_thresh);

// Assign dominant orientation(s) to each keypoint.
// Parallelizable: per-keypoint (independent histogram computation).
// Note: may add duplicate keypoints for secondary peaks (>80% of dominant).
void assign_orientations(
    std::vector<Keypoint>& keypoints,
    const GaussianPyramid& gauss);

// Compute 128-dim SIFT descriptor for each keypoint.
// Parallelizable: per-keypoint.
void compute_descriptors(
    std::vector<Keypoint>& keypoints,
    const GaussianPyramid& gauss);
