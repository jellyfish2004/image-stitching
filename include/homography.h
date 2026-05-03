#pragma once

#include "sift.h"
#include "match.h"
#include <vector>
#include <array>

// 3x3 Matrix (row-major)
using Mat3 = std::array<float, 9>;

// Point Transforms
void apply_homography(const Mat3& H, float x, float y,
                      float& ox, float& oy);

Mat3 invert_homography(const Mat3& H);

// DLT (Direct Linear Transform)
// Compute homography from exactly 4 point correspondences.
// Uses Hartley normalisation for numerical stability.
// Returns false if the configuration is degenerate.
bool compute_homography_dlt(const float pts1[][2],
                            const float pts2[][2],
                            Mat3& H);

// RANSAC
// Robustly estimate homography from noisy matches.
// Returns number of inliers (0 on failure).
int ransac_homography(const std::vector<Keypoint>& kp1,
                      const std::vector<Keypoint>& kp2,
                      const std::vector<Match>& matches,
                      Mat3& H_out,
                      int   iterations = 1000,
                      float thresh     = 10.0f);
