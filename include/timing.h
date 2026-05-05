#pragma once
#include <chrono>

extern double total_t_dog;
extern double total_t_gauss;
extern double total_t_downsample;
extern double total_t_upsample;
extern double total_t_convolve;

inline double get_time() {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}
