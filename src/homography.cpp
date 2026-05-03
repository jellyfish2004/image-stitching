#include "homography.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>

// [x' y' w]T = H [x y 1]T
void apply_homography(const Mat3& H, float x, float y,
                      float& ox, float& oy) {
    float w = H[6] * x + H[7] * y + H[8];
    ox = (H[0] * x + H[1] * y + H[2]) / w;
    oy = (H[3] * x + H[4] * y + H[5]) / w;
}

// matrix inverse
Mat3 invert_homography(const Mat3& H) {
    const float* h = H.data();
    float det = h[0] * (h[4] * h[8] - h[5] * h[7])
              - h[1] * (h[3] * h[8] - h[5] * h[6])
              + h[2] * (h[3] * h[7] - h[4] * h[6]);

    Mat3 inv;
    inv[0] = (h[4] * h[8] - h[5] * h[7]) / det;
    inv[1] = (h[2] * h[7] - h[1] * h[8]) / det;
    inv[2] = (h[1] * h[5] - h[2] * h[4]) / det;
    inv[3] = (h[5] * h[6] - h[3] * h[8]) / det;
    inv[4] = (h[0] * h[8] - h[2] * h[6]) / det;
    inv[5] = (h[2] * h[3] - h[0] * h[5]) / det;
    inv[6] = (h[3] * h[7] - h[4] * h[6]) / det;
    inv[7] = (h[1] * h[6] - h[0] * h[7]) / det;
    inv[8] = (h[0] * h[4] - h[1] * h[3]) / det;
    return inv;
}

// JACOBI SVD (for small 9x9 symmetric system)
// get eigenvalues and eigenvectors of A
// A = VDV^T
static void jacobi_svd(double A[9][9], double d[9], double V[9][9], int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            V[i][j] = (i == j) ? 1.0 : 0.0;

    for (int sweep = 0; sweep < 50; ++sweep) {
        double off = 0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                off += A[i][j] * A[i][j];
        if (off < 1e-20) break;

        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                if (std::fabs(A[p][q]) < 1e-15) continue;
                double theta = 0.5 * (A[q][q] - A[p][p]) / A[p][q];
                double t = (theta >= 0)
                    ? 1.0 / (theta + std::sqrt(1.0 + theta * theta))
                    : 1.0 / (theta - std::sqrt(1.0 + theta * theta));
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = t * c;

                double App = A[p][p], Aqq = A[q][q], Apq = A[p][q];
                A[p][p] = App - t * Apq;
                A[q][q] = Aqq + t * Apq;
                A[p][q] = A[q][p] = 0;

                for (int r = 0; r < n; ++r) {
                    if (r == p || r == q) continue;
                    double arp = A[r][p], arq = A[r][q];
                    A[r][p] = A[p][r] = c * arp - s * arq;
                    A[r][q] = A[q][r] = s * arp + c * arq;
                }
                for (int r = 0; r < n; ++r) {
                    double vrp = V[r][p], vrq = V[r][q];
                    V[r][p] = c * vrp - s * vrq;
                    V[r][q] = s * vrp + c * vrq;
                }
            }
        }
    }
    for (int i = 0; i < n; ++i) d[i] = A[i][i];
}

//  DLT (with Hartley normalisation)
bool compute_homography_dlt(const float pts1[][2],
                            const float pts2[][2],
                            Mat3& H) {
    // Centroids
    float cx1 = 0, cy1 = 0, cx2 = 0, cy2 = 0;
    for (int i = 0; i < 4; ++i) {
        cx1 += pts1[i][0]; cy1 += pts1[i][1];
        cx2 += pts2[i][0]; cy2 += pts2[i][1];
    }
    cx1 /= 4; cy1 /= 4; cx2 /= 4; cy2 /= 4;

    // Mean distances
    float d1 = 0, d2 = 0;
    for (int i = 0; i < 4; ++i) {
        d1 += std::sqrt((pts1[i][0] - cx1) * (pts1[i][0] - cx1) +
                        (pts1[i][1] - cy1) * (pts1[i][1] - cy1));
        d2 += std::sqrt((pts2[i][0] - cx2) * (pts2[i][0] - cx2) +
                        (pts2[i][1] - cy2) * (pts2[i][1] - cy2));
    }
    d1 /= 4; d2 /= 4;
    if (d1 < 1e-8f || d2 < 1e-8f) return false;

    float s1 = std::sqrt(2.0f) / d1, s2 = std::sqrt(2.0f) / d2;

    float T1[9] = { s1, 0, -s1*cx1,  0, s1, -s1*cy1,  0, 0, 1 };

    // Normalised points
    float np1[4][2], np2[4][2];
    for (int i = 0; i < 4; ++i) {
        np1[i][0] = s1 * (pts1[i][0] - cx1);
        np1[i][1] = s1 * (pts1[i][1] - cy1);
        np2[i][0] = s2 * (pts2[i][0] - cx2);
        np2[i][1] = s2 * (pts2[i][1] - cy2);
    }

    // Build ATA (9×9)
    double A[9][9] = {};
    for (int i = 0; i < 4; ++i) {
        double x1 = np1[i][0], y1 = np1[i][1];
        double x2 = np2[i][0], y2 = np2[i][1];
        double r0[9] = { 0,  0,  0, -x1, -y1, -1,  y2*x1,  y2*y1,  y2};
        double r1[9] = {x1, y1,  1,   0,   0,  0, -x2*x1, -x2*y1, -x2};
        for (int m = 0; m < 9; ++m)
            for (int n = 0; n < 9; ++n)
                A[m][n] += r0[m] * r0[n] + r1[m] * r1[n];
    }

    double d[9], V[9][9];
    jacobi_svd(A, d, V, 9);

    // Smallest eigenvalue → homography
    int min_idx = 0;
    for (int i = 1; i < 9; ++i)
        if (d[i] < d[min_idx]) min_idx = i;

    float Hn[9];
    for (int i = 0; i < 9; ++i) Hn[i] = static_cast<float>(V[i][min_idx]);

    // Denormalise: H = T2_inv * Hn * T1
    float T2inv[9] = { 1/s2, 0, cx2,  0, 1/s2, cy2,  0, 0, 1 };

    float tmp[9] = {};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 3; ++k)
                tmp[r*3+c] += Hn[r*3+k] * T1[k*3+c];

    H.fill(0);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 3; ++k)
                H[r*3+c] += T2inv[r*3+k] * tmp[k*3+c];

    // Normalise so H[8] = 1
    if (std::fabs(H[8]) > 1e-10f) {
        float sc = H[8];
        for (int i = 0; i < 9; ++i) H[i] /= sc;
    }
    return true;
}


//  RANSAC
// Simple LCG RNG (thread-local state for parallelisation)
static unsigned int lcg_state = 42;
static int rand_int(int max) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return static_cast<int>((lcg_state >> 1) % static_cast<unsigned>(max));
}

int ransac_homography(const std::vector<Keypoint>& kp1,
                      const std::vector<Keypoint>& kp2,
                      const std::vector<Match>& matches,
                      Mat3& H_out,
                      int iterations,
                      float thresh) {
    int nm = static_cast<int>(matches.size());
    printf("[RANSAC] Running %d iterations (threshold=%.1fpx, %d matches)...\n",
           iterations, thresh, nm);

    if (nm < 4) {
        printf("[RANSAC] Too few matches.\n");
        return 0;
    }

    int best_inliers = 0;
    Mat3 best_H;

    // Parallelizable: each iteration is independent.
    // OpenMP: #pragma omp parallel for with thread-local best + reduction.
    // Each thread needs its own RNG state.
    for (int iter = 0; iter < iterations; ++iter) {
        // Pick 4 random distinct matches
        int idx[4];
        for (int k = 0; k < 4; ++k) {
            int r;
            bool dup;
            do {
                r = rand_int(nm);
                dup = false;
                for (int l = 0; l < k; ++l)
                    if (idx[l] == r) { dup = true; break; }
            } while (dup);
            idx[k] = r;
        }

        float p1[4][2], p2[4][2];
        for (int k = 0; k < 4; ++k) {
            p1[k][0] = kp1[matches[idx[k]].idx1].x;
            p1[k][1] = kp1[matches[idx[k]].idx1].y;
            p2[k][0] = kp2[matches[idx[k]].idx2].x;
            p2[k][1] = kp2[matches[idx[k]].idx2].y;
        }

        Mat3 H;
        if (!compute_homography_dlt(p1, p2, H)) continue;

        // Count inliers (parallelizable per-match)
        int inliers = 0;
        for (int m = 0; m < nm; ++m) {
            float x1 = kp1[matches[m].idx1].x;
            float y1 = kp1[matches[m].idx1].y;
            float x2 = kp2[matches[m].idx2].x;
            float y2 = kp2[matches[m].idx2].y;
            float px, py;
            apply_homography(H, x1, y1, px, py);
            float err = std::sqrt((px - x2) * (px - x2) +
                                  (py - y2) * (py - y2));
            if (err < thresh) ++inliers;
        }

        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_H = H;
        }
    }

    printf("[RANSAC] Best model: %d inliers out of %d matches.\n",
           best_inliers, nm);

    if (best_inliers < 4) {
        printf("[RANSAC] Not enough inliers.\n");
        return 0;
    }

    H_out = best_H;
    return best_inliers;
}
