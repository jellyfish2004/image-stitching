#pragma once

#include "image.h"
#include "homography.h"

// Canvas
// Describes the output canvas that fits both the warped image 1 and image 2.
struct Canvas {
    int   width, height;
    float offset_x, offset_y;   // translation so that img2's (0,0) maps here
};

// Compute output canvas by projecting img1's corners through H
// and taking the bounding-box union with img2.
Canvas compute_canvas_size(const Mat3& H,
                           const Image& img1,
                           const Image& img2);

// Inverse Warp
// For each output pixel, apply H_inv to find the source location in img1,
// then bilinearly interpolate.
//
// Parallelization: per output pixel — embarrassingly parallel.
// Perfect for OpenMP parallel for and OpenACC kernels.
Image warp_image(const Image& img1,
                 const Mat3& H_inv,
                 const Canvas& canvas);
