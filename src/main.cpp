#include "image.h"
#include <iostream>

int main() {
    std::cout << "Hello\n";
    Image img = load_image("test.jpg");
    std::cout << img.width << " " << img.height << " " << img.channels << "\n";
    Image gray = rgb_to_grayscale(img);
    save_image("gray.png", gray);
    return 0;
}