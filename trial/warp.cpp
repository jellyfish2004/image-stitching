#include "warp.h"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <omp.h>

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
    int canvas_w = canvas.width, canvas_h = canvas.height;
    float offset_x = canvas.offset_x, offset_y = canvas.offset_y;

    Image warped(canvas_w, canvas_h, ch);

    // Fill with -1 to mark "no data" pixels (used by blending)
    std::fill(warped.data.begin(), warped.data.end(), -1.0f);

    // Raw pointers for GPU-friendly access (no object methods inside acc region)
    const float* img1_data   = img1.data.data();
    float*       warped_data = warped.data.data();

    // H_inv as flat array for GPU transfer
    float h[9];
    for (int i = 0; i < 9; ++i) h[i] = H_inv[i];

    // Each (x, y) output pixel is fully independent:
    // reads from img1_data (read-only), writes to unique warped_data location.
    // collapse(2) fuses y and x into one flat iteration space for the GPU.
    // copyin: img1_data and h are read-only on device.
    // copyout: warped_data is written on device, transferred back to host.
    #pragma acc parallel loop collapse(2) \
        copyin(img1_data[0 : w1 * h1 * ch], h[0:9]) \
        copyout(warped_data[0 : canvas_w * canvas_h * ch]) \
        firstprivate(w1, h1, ch, canvas_w, canvas_h, offset_x, offset_y)
    for (int y = 0; y < canvas_h; ++y) {
        for (int x = 0; x < canvas_w; ++x) {
            // Map canvas pixel back to img1 coordinates
            float cx = static_cast<float>(x) - offset_x;
            float cy = static_cast<float>(y) - offset_y;

            // apply_homography inlined for GPU compatibility
            // (device kernels cannot call host-side functions)
            float w_val = h[6] * cx + h[7] * cy + h[8];
            float sx    = (h[0] * cx + h[1] * cy + h[2]) / w_val;
            float sy    = (h[3] * cx + h[4] * cy + h[5]) / w_val;

            // Check bounds (need 1 pixel margin for bilinear)
            if (sx < 0 || sx >= w1 - 1 || sy < 0 || sy >= h1 - 1)
                continue;

            // Bilinear interpolation
            int ix = static_cast<int>(sx);
            int iy = static_cast<int>(sy);
            float fx = sx - ix;
            float fy = sy - iy;

            #pragma omp simd
            for (int c = 0; c < ch; ++c) {
                float v =
                    img1_data[(iy     * w1 + ix    ) * ch + c] * (1 - fx) * (1 - fy)
                  + img1_data[(iy     * w1 + ix + 1) * ch + c] *      fx  * (1 - fy)
                  + img1_data[((iy+1) * w1 + ix    ) * ch + c] * (1 - fx) *      fy
                  + img1_data[((iy+1) * w1 + ix + 1) * ch + c] *      fx  *      fy;
                warped_data[(y * canvas_w + x) * ch + c] = v;
            }
        }
    }

    return warped;
}
