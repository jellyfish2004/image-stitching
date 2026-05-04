#pragma once

#include "image.h"
#include "warp.h"

// Simple Blend
// Weighted average in the overlap region.
// Parallelizable: per output pixel.
Image blend_simple(const Image& warped1,
                   const Image& img2,
                   const Canvas& canvas);

