#include "image.h"
#include "sift.h"
#include "match.h"
#include "homography.h"
#include "warp.h"
#include "blend.h"
#include "viz.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s image1 image2\n", argv[0]);
        return 1;
    }

    printf("Loading images...\n");
    Image img1 = load_image(argv[1]);
    Image img2 = load_image(argv[2]);
    printf("Image 1: %d x %d x %d\n", img1.width, img1.height, img1.channels);
    printf("Image 2: %d x %d x %d\n", img2.width, img2.height, img2.channels);

    printf("\n SIFT Feature Detection & Description...\n");

    printf("Processing Image 1...\n");
    Image gray1 = rgb_to_grayscale(img1);
    auto kp1 = sift_detect_and_describe(gray1);

    printf("Processing Image 2...\n");
    Image gray2 = rgb_to_grayscale(img2);
    auto kp2 = sift_detect_and_describe(gray2);

    printf("\n Feature Matching (Lowe Ratio Test)...\n");
    auto matches = match_features(kp1, kp2);

    if (matches.size() < 4) {
        fprintf(stderr, "Not enough matches (%zu) to compute homography.\n",
                matches.size());
        return 1;
    }

    printf("\n Homography Estimation (RANSAC + DLT/SVD)...");
    Mat3 H;
    int inliers = ransac_homography(kp1, kp2, matches, H);
    if (inliers == 0) {
        fprintf(stderr, "Homography estimation failed.\n");
        return 1;
    }

    printf("Homography matrix:\n");
    for (int r = 0; r < 3; ++r) {
        printf("       [ %8.4f %8.4f %8.4f ]\n",
               H[r * 3], H[r * 3 + 1], H[r * 3 + 2]);
    }

    printf("\n Perspective Warp...\n");
    Canvas canvas = compute_canvas_size(H, img1, img2);
    Mat3 H_inv = invert_homography(H);
    Image warped = warp_image(img1, H_inv, canvas);

    printf("\n Panorama Blend...\n");
    Image result = blend_simple(warped, img2, canvas);

    printf("\nSaving outputs...\n");
    save_image("stitched.png", result);

    save_keypoints_image(img1, kp1, "keypoints1.png");
    save_keypoints_image(img2, kp2, "keypoints2.png");
    save_matches_image(img1, img2, kp1, kp2, matches, "matches.png");

    printf("\n Done!\n");
    printf("  stitched.png     - final stitched panorama\n");
    printf("  keypoints1.png   - image1 with SIFT keypoints\n");
    printf("  keypoints2.png   - image2 with SIFT keypoints\n");
    printf("  matches.png      - feature match lines\n");

    return 0;
}