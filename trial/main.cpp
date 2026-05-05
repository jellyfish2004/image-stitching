#include "image.h"
#include "sift.h"
#include "match.h"
#include "homography.h"
#include "warp.h"
#include "blend.h"
#include "viz.h"

#include <cstdio>
#include <cstdlib>
#include "timing.h"

double total_t_dog = 0.0;
double total_t_gauss = 0.0;
double total_t_downsample = 0.0;
double total_t_upsample = 0.0;
double total_t_convolve = 0.0;

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s image1 image2\n", argv[0]);
        return 1;
    }

    double t_total_start = get_time();

    printf("Loading images...\n");
    double t0 = get_time();
    Image img1 = load_image(argv[1]);
    Image img2 = load_image(argv[2]);
    printf("Image 1: %d x %d x %d\n", img1.width, img1.height, img1.channels);
    printf("Image 2: %d x %d x %d\n", img2.width, img2.height, img2.channels);
    printf("  [Timing] Image loading: %.4f s\n", get_time() - t0);

    printf("\n SIFT Feature Detection & Description...\n");

    printf("Processing Image 1...\n");
    t0 = get_time();
    Image gray1 = rgb_to_grayscale(img1);
    auto kp1 = sift_detect_and_describe(gray1);
    printf("  [Timing] SIFT Image 1: %.4f s\n", get_time() - t0);

    printf("Processing Image 2...\n");
    t0 = get_time();
    Image gray2 = rgb_to_grayscale(img2);
    auto kp2 = sift_detect_and_describe(gray2);
    printf("  [Timing] SIFT Image 2: %.4f s\n", get_time() - t0);

    printf("\n Feature Matching (Lowe Ratio Test)...\n");
    t0 = get_time();
    auto matches = match_features(kp1, kp2);
    printf("  [Timing] Matching: %.4f s\n", get_time() - t0);

    if (matches.size() < 4) {
        fprintf(stderr, "Not enough matches (%zu) to compute homography.\n",
                matches.size());
        return 1;
    }

    printf("\n Homography Estimation (RANSAC + DLT/SVD)...");
    t0 = get_time();
    Mat3 H;
    int inliers = ransac_homography(kp1, kp2, matches, H);
    if (inliers == 0) {
        fprintf(stderr, "\nHomography estimation failed.\n");
        return 1;
    }
    printf(" done.\n  [Timing] Homography: %.4f s\n", get_time() - t0);

    printf("Homography matrix:\n");
    for (int r = 0; r < 3; ++r) {
        printf("       [ %8.4f %8.4f %8.4f ]\n",
               H[r * 3], H[r * 3 + 1], H[r * 3 + 2]);
    }

    printf("\n Perspective Warp...\n");
    t0 = get_time();
    Canvas canvas = compute_canvas_size(H, img1, img2);
    Mat3 H_inv = invert_homography(H);
    Image warped = warp_image(img1, H_inv, canvas);
    printf("  [Timing] Warp: %.4f s\n", get_time() - t0);

    printf("\n Panorama Blend...\n");
    t0 = get_time();
    Image result = blend_simple(warped, img2, canvas);
    printf("  [Timing] Blend: %.4f s\n", get_time() - t0);


    printf("\nSaving outputs...\n");
    t0 = get_time();
    save_image("stitched.png", result);

    save_keypoints_image(img1, kp1, "keypoints1.png");
    save_keypoints_image(img2, kp2, "keypoints2.png");
    save_matches_image(img1, img2, kp1, kp2, matches, "matches.png");
    printf("  [Timing] Save outputs: %.4f s\n", get_time() - t0);

    printf("\n  [Timing] Total execution time: %.4f s\n", get_time() - t_total_start);

    printf("\n  [Accumulated Timings]\n");
    printf("    Gaussian Pyramid : %.4f s\n", total_t_gauss);
    printf("    DoG Pyramid      : %.4f s\n", total_t_dog);
    printf("    Convolutions     : %.4f s\n", total_t_convolve);
    printf("    Downsampling     : %.4f s\n", total_t_downsample);
    printf("    Upsampling       : %.4f s\n", total_t_upsample);

    printf("\n Done!\n");
    printf("  stitched.png     - final stitched panorama\n");
    printf("  keypoints1.png   - image1 with SIFT keypoints\n");
    printf("  keypoints2.png   - image2 with SIFT keypoints\n");
    printf("  matches.png      - feature match lines\n");

    return 0;
}