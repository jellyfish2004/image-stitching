#pragma once

#include "image.h"
#include "sift.h"
#include "match.h"
#include <string>
#include <vector>

// ── Visualization ──────────────────────────────────────────────
// Save image with SIFT keypoints drawn as green circles + orientation ticks.
void save_keypoints_image(const Image& img,
                          const std::vector<Keypoint>& kps,
                          const std::string& path);

// Save side-by-side image with coloured match lines between keypoints.
void save_matches_image(const Image& img1,
                        const Image& img2,
                        const std::vector<Keypoint>& kp1,
                        const std::vector<Keypoint>& kp2,
                        const std::vector<Match>& matches,
                        const std::string& path);
