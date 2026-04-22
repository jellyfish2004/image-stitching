#pragma once

#include<vector>
#include<string>

struct Image {
    int width;
    int height;
    int channels; // 1 or 3
    std::vector<float> data;

    Image();
    Image(int w, int h, int c);

    float& at(int x, int y, int c);
    const float& at(int x, int y, int c) const;
};

// I/O
Image load_image(const std::string& path, bool force_grayscale = false);
void save_image(const std::string& path, const Image& img);

// utilities
Image rgb_to_grayscale(const Image& img);