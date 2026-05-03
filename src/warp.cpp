#include "warp.h"

#include <cmath>
#include <cstdio>
#include <algorithm>

//  CANVAS SIZE
Canvas compute_canvas_size(const Mat3& H,
                           const Image& img1,
                           const Image& img2) {
    int w1 = img1.width, h1 = img1.height;
    int w2 = img2.width, h2 = img2.height;

    // Project img1 corners through H
    float corners[4][2] = {
        {0, 0},
        {static_cast<float>(w1 - 1), 0},
        {0, static_cast<float>(h1 - 1)},
        {static_cast<float>(w1 - 1), static_cast<float>(h1 - 1)}
    };

    float minx = 0, miny = 0;
    float maxx = static_cast<float>(w2 - 1);
    float maxy = static_cast<float>(h2 - 1);

    for (int c = 0; c < 4; ++c) {
        float px, py;
        apply_homography(H, corners[c][0], corners[c][1], px, py);
        minx = std::min(minx, px);
        miny = std::min(miny, py);
        maxx = std::max(maxx, px);
        maxy = std::max(maxy, py);
    }

    Canvas canvas;
    canvas.width    = static_cast<int>(std::ceil(maxx - minx)) + 1;
    canvas.height   = static_cast<int>(std::ceil(maxy - miny)) + 1;
    canvas.offset_x = -minx;
    canvas.offset_y = -miny;

    printf("[WARP] Output canvas: %d x %d (offset: %.1f, %.1f)\n",
           canvas.width, canvas.height, canvas.offset_x, canvas.offset_y);

    return canvas;
}

// INVERSE WARP + BILINEAR INTERPOLATION
Image warp_image(const Image& img1,
                 const Mat3& H_inv,
                 const Canvas& canvas) {
    printf("[WARP] Warping image (%d x %d) onto canvas...\n",
           img1.width, img1.height);

    int w1 = img1.width, h1 = img1.height;
    int ch = img1.channels;
    Image warped(canvas.width, canvas.height, ch);

    // Fill with -1 to mark "no data" pixels (used by blending)
    std::fill(warped.data.begin(), warped.data.end(), -1.0f);

    // Parallelizable: per output pixel - embarrassingly parallel.
    for (int y = 0; y < canvas.height; ++y) {
        for (int x = 0; x < canvas.width; ++x) {
            // Map canvas pixel back to img1 coordinates
            float cx = static_cast<float>(x) - canvas.offset_x;
            float cy = static_cast<float>(y) - canvas.offset_y;

            float sx, sy;
            apply_homography(H_inv, cx, cy, sx, sy);

            // Check bounds (need 1 pixel margin for bilinear)
            if (sx < 0 || sx >= w1 - 1 || sy < 0 || sy >= h1 - 1)
                continue;

            // Bilinear interpolation
            int ix = static_cast<int>(sx);
            int iy = static_cast<int>(sy);
            float fx = sx - ix;
            float fy = sy - iy;

            for (int c = 0; c < ch; ++c) {
                float v = img1.at(ix,     iy,     c) * (1 - fx) * (1 - fy)
                        + img1.at(ix + 1, iy,     c) *      fx  * (1 - fy)
                        + img1.at(ix,     iy + 1, c) * (1 - fx) *      fy
                        + img1.at(ix + 1, iy + 1, c) *      fx  *      fy;
                warped.at(x, y, c) = v;
            }
        }
    }

    return warped;
}
