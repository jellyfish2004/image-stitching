#include "viz.h"

#include <cmath>
#include <cstdio>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Drawing Primitives ─────────────────────────────────────────

static void draw_circle(Image& img, int cx, int cy, int r,
                         float R, float G, float B) {
    for (int y = cy - r; y <= cy + r; ++y) {
        for (int x = cx - r; x <= cx + r; ++x) {
            int dx = x - cx, dy = y - cy;
            int d2 = dx * dx + dy * dy;
            if (d2 <= r * r && d2 >= (r - 1) * (r - 1)) {
                if (x < 0 || x >= img.width || y < 0 || y >= img.height)
                    continue;
                if (img.channels >= 3) {
                    img.at(x, y, 0) = R;
                    img.at(x, y, 1) = G;
                    img.at(x, y, 2) = B;
                }
            }
        }
    }
}

static void draw_line(Image& img, int x0, int y0, int x1, int y1,
                       float R, float G, float B) {
    int dx = std::abs(x1 - x0), dy = std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    for (;;) {
        if (x0 >= 0 && x0 < img.width && y0 >= 0 && y0 < img.height) {
            if (img.channels >= 3) {
                img.at(x0, y0, 0) = R;
                img.at(x0, y0, 1) = G;
                img.at(x0, y0, 2) = B;
            }
        }
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 <  dx) { err += dx; y0 += sy; }
    }
}

// ── Save Keypoints Image ──────────────────────────────────────

void save_keypoints_image(const Image& img,
                          const std::vector<Keypoint>& kps,
                          const std::string& path) {
    printf("[VIS] Saving keypoints image: %s (%zu keypoints)\n",
           path.c_str(), kps.size());

    Image out = img;  // copy
    constexpr int radius = 5;

    for (const auto& kp : kps) {
        int cx = static_cast<int>(kp.x);
        int cy = static_cast<int>(kp.y);

        // Orientation tick (yellow)
        int ex = cx + static_cast<int>(radius * std::cos(kp.angle));
        int ey = cy + static_cast<int>(radius * std::sin(kp.angle));
        draw_line(out, cx, cy, ex, ey, 1.0f, 1.0f, 0.0f);

        // Green circle
        draw_circle(out, cx, cy, radius, 0.0f, 1.0f, 0.0f);
    }

    save_image(path, out);
}

// ── Save Matches Image ────────────────────────────────────────

void save_matches_image(const Image& img1,
                        const Image& img2,
                        const std::vector<Keypoint>& kp1,
                        const std::vector<Keypoint>& kp2,
                        const std::vector<Match>& matches,
                        const std::string& path) {
    printf("[VIS] Saving match lines image: %s (%zu matches)\n",
           path.c_str(), matches.size());

    int w1 = img1.width, h1 = img1.height;
    int w2 = img2.width, h2 = img2.height;
    int out_w = w1 + w2;
    int out_h = std::max(h1, h2);
    int ch = img1.channels;

    Image out(out_w, out_h, ch);

    // Copy img1 to left
    for (int y = 0; y < h1; ++y)
        for (int x = 0; x < w1; ++x)
            for (int c = 0; c < ch; ++c)
                out.at(x, y, c) = img1.at(x, y, c);

    // Copy img2 to right
    for (int y = 0; y < h2; ++y)
        for (int x = 0; x < w2; ++x)
            for (int c = 0; c < ch; ++c)
                out.at(w1 + x, y, c) = img2.at(x, y, c);

    // Draw match lines with cycling colours
    float colours[][3] = {
        {1.0f, 0.0f, 0.0f},     // red
        {0.0f, 1.0f, 1.0f},     // cyan
        {1.0f, 0.647f, 0.0f},   // orange
        {1.0f, 0.0f, 1.0f},     // magenta
        {0.0f, 0.784f, 0.0f}    // green
    };
    int nc = 5;

    for (size_t m = 0; m < matches.size(); ++m) {
        int x1 = static_cast<int>(kp1[matches[m].idx1].x);
        int y1 = static_cast<int>(kp1[matches[m].idx1].y);
        int x2 = static_cast<int>(kp2[matches[m].idx2].x) + w1;
        int y2 = static_cast<int>(kp2[matches[m].idx2].y);

        float* col = colours[m % nc];
        draw_line(out, x1, y1, x2, y2, col[0], col[1], col[2]);
        draw_circle(out, x1, y1, 4, col[0], col[1], col[2]);
        draw_circle(out, x2, y2, 4, col[0], col[1], col[2]);
    }

    save_image(path, out);
}
