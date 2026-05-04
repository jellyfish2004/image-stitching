#include "blend.h"
#include "filter.h"

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>

Image blend_simple(const Image& warped1,
                   const Image& img2,
                   const Canvas& canvas) {
    printf("[BLEND] Simple weighted average blend...\n");

    int out_w = canvas.width;
    int out_h = canvas.height;
    int ch    = img2.channels;

    // Accumulation buffers
    std::vector<float> acc(out_w * out_h * ch, 0.0f);
    std::vector<float> wsum(out_w * out_h, 0.0f);

    // Accumulate warped image 1 (pixels with data have value >= 0)
    // Parallelizable: per-pixel
    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            if (warped1.at(x, y, 0) < 0.0f) continue;  // sentinel: no data
            for (int c = 0; c < ch; ++c) {
                acc[(y * out_w + x) * ch + c] += warped1.at(x, y, c);
            }
            wsum[y * out_w + x] += 1.0f;
        }
    }

    // Accumulate image 2 at its canvas position
    // Parallelizable: per-pixel
    int w2 = img2.width, h2 = img2.height;
    int ox = static_cast<int>(canvas.offset_x);
    int oy = static_cast<int>(canvas.offset_y);

    for (int y = 0; y < h2; ++y) {
        for (int x = 0; x < w2; ++x) {
            int cx = x + ox;
            int cy = y + oy;
            if (cx < 0 || cx >= out_w || cy < 0 || cy >= out_h) continue;
            for (int c = 0; c < ch; ++c) {
                acc[(cy * out_w + cx) * ch + c] += img2.at(x, y, c);
            }
            wsum[cy * out_w + cx] += 1.0f;
        }
    }

    // Normalise
    Image result(out_w, out_h, ch);
    // Parallelizable: per-pixel
    for (int i = 0; i < out_w * out_h; ++i) {
        float w = wsum[i];
        if (w > 0.0f) {
            for (int c = 0; c < ch; ++c) {
                result.data[i * ch + c] = acc[i * ch + c] / w;
            }
        }
    }

    return result;
}
