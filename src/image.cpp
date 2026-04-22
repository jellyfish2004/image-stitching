#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "image.h"
#include "stb_image.h"
#include "stb_image_write.h"

#include <stdexcept>

Image::Image() : width(0), height(0), channels(0) {}
Image::Image(int w, int h, int c) : width(w), height(h), channels(c), data(w * h * c, 0.0f) {}

float& Image::at(int x, int y, int c) {
    return data[(y * width + x) * channels + c];
}

const float& Image::at(int x, int y, int c) const {
    return data[(y * width + x) * channels + c];
}

Image load_image(const std::string& path, bool force_grayscale) {
    int w, h, c;
    unsigned char* raw = stbi_load(path.c_str(), &w, &h, &c, 0);

    if (!raw) {
        throw std::runtime_error("Failed to load image: " + path);
    }

    int target_c = force_grayscale ? 1 : c;
    Image img(w, h, target_c);

    if (force_grayscale && c >= 3) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int idx = (y * w + x) * c;

                float r = raw[idx + 0] / 255.0f;
                float g = raw[idx + 1] / 255.0f;
                float b = raw[idx + 2] / 255.0f;

                img.at(x, y, 0) = 0.299f * r + 0.587f * g + 0.114f * b;
            }
        }
    } else {
        for (int i = 0; i < w * h * target_c; ++i) {
            img.data[i] = raw[i] / 255.0f;
        }
    }

    stbi_image_free(raw);
    return img;
}

void save_image(const std::string& path, const Image& img) {
    int size = img.width * img.height * img.channels;

    std::vector<unsigned char> out(size);

    for (int i = 0; i < size; ++i) {
        float v = img.data[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        out[i] = static_cast<unsigned char>(v * 255.0f);
    }

    if (!stbi_write_png(path.c_str(), img.width, img.height, img.channels, out.data(), img.width * img.channels)) {
        throw std::runtime_error("Failed to save image: " + path);
    }
}

Image rgb_to_grayscale(const Image& img) {
    if (img.channels == 1) return img;

    Image gray(img.width, img.height, 1);

    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            float r = img.at(x, y, 0);
            float g = img.at(x, y, 1);
            float b = img.at(x, y, 2);

            gray.at(x, y, 0) = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }

    return gray;
}