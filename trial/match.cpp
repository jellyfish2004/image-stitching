#include "match.h"

#include <cmath>
#include <cfloat>
#include <cstdio>

static float descriptor_dist_sq(const float* a, const float* b) {
    float sum = 0.0f;
    for (int i = 0; i < SIFT_DESC_SIZE; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}


// for each kp1, kp2 pair we compute distances and check if the ratio of the best and second best distance is less than the ratio_thresh.
// best/2nd best ratio is used to ensure that the pt doesn't match to multiple points -> ambiguous
std::vector<Match> match_features(const std::vector<Keypoint>& kp1,
                                   const std::vector<Keypoint>& kp2,
                                   float ratio_thresh) {
    printf("[MATCH] Running Lowe ratio test (threshold=%.2f)...\n", ratio_thresh);

    std::vector<Match> matches;
    int n1 = static_cast<int>(kp1.size());
    int n2 = static_cast<int>(kp2.size());

    // Parallelizable: outer loop over kp1 is embarrassingly parallel
    for (int i = 0; i < n1; ++i) {
        float best1 = FLT_MAX, best2 = FLT_MAX;
        int   idx1  = -1;

        for (int j = 0; j < n2; ++j) {
            float d = descriptor_dist_sq(kp1[i].desc, kp2[j].desc);
            if (d < best1) {
                best2 = best1;
                best1 = d;
                idx1  = j;
            } else if (d < best2) {
                best2 = d;
            }
        }

        if (best2 < 1e-10f) continue;

        if (std::sqrt(best1) / std::sqrt(best2) < ratio_thresh) {
            Match m;
            m.idx1 = i;
            m.idx2 = idx1;
            m.dist = std::sqrt(best1);
            matches.push_back(m);
        }
    }

    printf("[MATCH] Found %zu matches after ratio test.\n", matches.size());
    return matches;
}
