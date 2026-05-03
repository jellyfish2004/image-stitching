#pragma once

#include "sift.h"
#include <vector>

// Match
struct Match {
    int   idx1, idx2;   // indices into keypoint arrays (image 1, image 2)
    float dist;         // L2 descriptor distance
};

std::vector<Match> match_features(
    const std::vector<Keypoint>& kp1,
    const std::vector<Keypoint>& kp2,
    float ratio_thresh = 0.75f);
