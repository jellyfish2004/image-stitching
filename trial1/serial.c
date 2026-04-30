/*
 * ============================================================
 *  IMAGE STITCHING IN SERIAL C
 *  Steps:
 *    1. Load images (PPM format)
 *    2. Convert to grayscale
 *    3. SIFT (DoG, Keypoints, Orientation, Descriptors)
 *    4. Feature Matching (Lowe's Ratio Test)
 *    5. Homography via RANSAC + SVD
 *    6. Perspective Warp + Blend
 *    7. Save visualizations (keypoints on each image,
 *       match lines image, final stitched result)
 * ============================================================
 *
 *  Compile:
 *    gcc -O2 -o image_stitch image_stitch.c -lm
 *
 *  Usage:
 *    ./image_stitch image1.ppm image2.ppm
 *
 *  Output files:
 *    keypoints1.ppm      - image1 with SIFT keypoints drawn
 *    keypoints2.ppm      - image2 with SIFT keypoints drawn
 *    matches.ppm         - side-by-side match lines
 *    stitched.ppm        - final perspective-warped panorama
 *
 *  Note: Input images must be in PPM (P6 binary) format.
 *        Convert with: convert input.jpg input.ppm
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ============================================================
 *  CONSTANTS & TUNING PARAMETERS
 * ============================================================ */

/* SIFT Scale-Space */
#define SIFT_OCTAVES        4       /* number of octaves */
#define SIFT_SCALES         5       /* scales per octave (s+3 DoG layers) */
#define SIFT_SIGMA          1.6     /* base blur sigma */
#define SIFT_K              1.4142  /* scale factor between levels (sqrt(2)) */

/* Keypoint detection */
#define DOG_THRESHOLD       0.015   /* contrast threshold for DoG extrema */
#define EDGE_THRESHOLD      10.0    /* Hessian edge ratio threshold */
#define MAX_KEYPOINTS       2000    /* maximum keypoints per image */

/* Descriptor */
#define DESC_HIST_BINS      8       /* orientation bins per histogram cell */
#define DESC_GRID           4       /* 4x4 histogram grid */
#define DESC_SIZE           128     /* 4*4*8 = 128 */
#define ORI_BINS            36      /* orientation histogram bins */
#define ORI_SIGMA_FACTOR    1.5     /* sigma multiplier for orientation */

/* Matching */
#define LOWE_RATIO          0.75f   /* ratio test threshold */
#define MAX_MATCHES         500

/* RANSAC */
#define RANSAC_ITERS        1000
#define RANSAC_THRESH       10.0     /* reprojection error threshold (pixels) */
#define MIN_INLIERS         4      /* minimum inliers to accept homography */

/* Drawing */
#define CIRCLE_RADIUS       5       /* keypoint circle radius in pixels */
#define PI 3.14159265358979323846


/* ============================================================
 *  DATA STRUCTURES
 * ============================================================ */

typedef struct {
    unsigned char *data;    /* packed RGB bytes: R,G,B,R,G,B,... */
    int width, height;
} Image;

typedef struct {
    float x, y;            /* subpixel keypoint location */
    float scale;           /* scale (sigma) at which detected */
    float angle;           /* dominant orientation in radians */
    float desc[DESC_SIZE]; /* L2-normalized SIFT descriptor */
} Keypoint;

typedef struct {
    int idx1, idx2;        /* indices into kp arrays */
    float dist;            /* descriptor distance */
} Match;

/* ============================================================
 *  PPM IMAGE I/O
 * ============================================================ */

/* Load a P6 (binary PPM) image */
Image *load_ppm(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }

    char magic[3];
    fgets(magic, 3, f);
    if (magic[0] != 'P' || magic[1] != '6') {
        fprintf(stderr, "Not a P6 PPM: %s\n", path); fclose(f); return NULL;
    }

    /* Skip comments */
    int c = fgetc(f);
    while (c == '#') { while (fgetc(f) != '\n'); c = fgetc(f); }
    ungetc(c, f);

    Image *img = malloc(sizeof(Image));
    int maxval;
    fscanf(f, "%d %d %d", &img->width, &img->height, &maxval);
    fgetc(f); /* consume newline after header */

    size_t sz = (size_t)img->width * img->height * 3;
    img->data = malloc(sz);
    fread(img->data, 1, sz, f);
    fclose(f);
    return img;
}

/* Save a P6 PPM image */
void save_ppm(const char *path, Image *img) {
    FILE *f = fopen(path, "wb");
    fprintf(f, "P6\n%d %d\n255\n", img->width, img->height);
    fwrite(img->data, 1, (size_t)img->width * img->height * 3, f);
    fclose(f);
    printf("Saved: %s\n", path);
}

/* Clone an image */
Image *clone_image(Image *src) {
    Image *dst = malloc(sizeof(Image));
    dst->width = src->width; dst->height = src->height;
    size_t sz = (size_t)src->width * src->height * 3;
    dst->data = malloc(sz);
    memcpy(dst->data, src->data, sz);
    return dst;
}

void free_image(Image *img) { if (img) { free(img->data); free(img); } }

/* ============================================================
 *  GRAYSCALE FLOAT BUFFER HELPERS
 * ============================================================ */

typedef struct { float *d; int w, h; } GrayF;

GrayF *new_gray(int w, int h) {
    GrayF *g = malloc(sizeof(GrayF));
    g->w = w; g->h = h;
    g->d = calloc((size_t)w * h, sizeof(float));
    return g;
}

void free_gray(GrayF *g) { if (g) { free(g->d); free(g); } }

static inline float gat(GrayF *g, int x, int y) {
    if (x < 0) x = 0; if (x >= g->w) x = g->w - 1;
    if (y < 0) y = 0; if (y >= g->h) y = g->h - 1;
    return g->d[y * g->w + x];
}

static inline void gset(GrayF *g, int x, int y, float v) {
    if (x >= 0 && x < g->w && y >= 0 && y < g->h)
        g->d[y * g->w + x] = v;
}

/* Convert RGB image to float grayscale [0..1] */
GrayF *rgb_to_gray(Image *img) {
    GrayF *g = new_gray(img->width, img->height);
    for (int i = 0; i < img->width * img->height; i++) {
        float r = img->data[3*i+0] / 255.0f;
        float gr = img->data[3*i+1] / 255.0f;
        float b = img->data[3*i+2] / 255.0f;
        g->d[i] = 0.299f*r + 0.587f*gr + 0.114f*b;
    }
    return g;
}

/* ============================================================
 *  GAUSSIAN BLUR (separable 1D kernel)
 * ============================================================ */

/* Build a 1D Gaussian kernel of radius r */
static float *make_kernel(float sigma, int *radius_out) {
    int r = (int)ceil(3.0 * sigma);
    *radius_out = r;
    int sz = 2 * r + 1;
    float *k = malloc(sz * sizeof(float));
    float sum = 0;
    for (int i = 0; i < sz; i++) {
        int x = i - r;
        k[i] = expf(-(float)(x*x) / (2.0f * sigma * sigma));
        sum += k[i];
    }
    for (int i = 0; i < sz; i++) k[i] /= sum;
    return k;
}

/* Apply separable Gaussian blur to a GrayF buffer */
GrayF *gaussian_blur(GrayF *src, float sigma) {
    int r;
    float *k = make_kernel(sigma, &r);
    int w = src->w, h = src->h;

    /* Horizontal pass */
    GrayF *tmp = new_gray(w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            float val = 0;
            for (int i = -r; i <= r; i++)
                val += gat(src, x+i, y) * k[i+r];
            gset(tmp, x, y, val);
        }

    /* Vertical pass */
    GrayF *dst = new_gray(w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            float val = 0;
            for (int i = -r; i <= r; i++)
                val += gat(tmp, x, y+i) * k[i+r];
            gset(dst, x, y, val);
        }

    free_gray(tmp);
    free(k);
    return dst;
}

/* ============================================================
 *  STEP 3A: BUILD DOG SCALE-SPACE
 *  For each octave:
 *    - build (SIFT_SCALES+3) Gaussian blurred images
 *    - subtract adjacent pairs to get (SIFT_SCALES+2) DoG images
 * ============================================================ */

/* Downsample by 2 (nearest neighbour) */
GrayF *downsample(GrayF *src) {
    int w = src->w / 2, h = src->h / 2;
    GrayF *dst = new_gray(w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            gset(dst, x, y, gat(src, x*2, y*2));
    return dst;
}

/* Compute DoG pyramid.
   dog[octave][scale] has dimensions (w>>octave) x (h>>octave)
   Returns number of DoG layers per octave = SIFT_SCALES+2 */
void build_dog_pyramid(GrayF *base,
                        GrayF *dog[SIFT_OCTAVES][SIFT_SCALES+2],
                        GrayF *gauss[SIFT_OCTAVES][SIFT_SCALES+3]) {
    /* ---- Octave 0: blur the base image ---- */
    gauss[0][0] = gaussian_blur(base, SIFT_SIGMA);
    for (int s = 1; s < SIFT_SCALES+3; s++) {
        /* sigma between consecutive levels: k^s * sigma */
        float sig = SIFT_SIGMA * powf(SIFT_K, s);
        /* incremental blur: only need to add the difference */
        float sig_prev = SIFT_SIGMA * powf(SIFT_K, s-1);
        float sig_inc = sqrtf(sig*sig - sig_prev*sig_prev);
        gauss[0][s] = gaussian_blur(gauss[0][s-1], sig_inc);
    }

    /* ---- Higher octaves: downsample top-of-stack ---- */
    for (int o = 1; o < SIFT_OCTAVES; o++) {
        gauss[o][0] = downsample(gauss[o-1][SIFT_SCALES]);
        for (int s = 1; s < SIFT_SCALES+3; s++) {
            float sig = SIFT_SIGMA * powf(SIFT_K, s);
            float sig_prev = SIFT_SIGMA * powf(SIFT_K, s-1);
            float sig_inc = sqrtf(sig*sig - sig_prev*sig_prev);
            gauss[o][s] = gaussian_blur(gauss[o][s-1], sig_inc);
        }
    }

    /* ---- Compute DoG = difference of adjacent Gaussians ---- */
    for (int o = 0; o < SIFT_OCTAVES; o++) {
        int w = gauss[o][0]->w, h = gauss[o][0]->h;
        for (int s = 0; s < SIFT_SCALES+2; s++) {
            dog[o][s] = new_gray(w, h);
            for (int i = 0; i < w*h; i++)
                dog[o][s]->d[i] = gauss[o][s+1]->d[i] - gauss[o][s]->d[i];
        }
    }
}

/* ============================================================
 *  STEP 3B: DETECT DOG EXTREMA (Keypoint Candidates)
 *  A pixel is a keypoint if it is the min or max of its
 *  3x3x3 neighbourhood across adjacent DoG scales.
 * ============================================================ */

static int is_extremum(GrayF *dog[SIFT_OCTAVES][SIFT_SCALES+2],
                        int o, int s, int x, int y) {
    float v = dog[o][s]->d[y * dog[o][s]->w + x];
    int above_thresh = fabsf(v) > DOG_THRESHOLD;
    if (!above_thresh) return 0;

    int is_max = 1, is_min = 1;
    for (int ds = -1; ds <= 1 && (is_max || is_min); ds++)
        for (int dy = -1; dy <= 1 && (is_max || is_min); dy++)
            for (int dx = -1; dx <= 1 && (is_max || is_min); dx++) {
                if (ds == 0 && dy == 0 && dx == 0) continue;
                float n = gat(dog[o][s+ds], x+dx, y+dy);
                if (n >= v) is_max = 0;
                if (n <= v) is_min = 0;
            }
    return (is_max || is_min);
}

/* Reject edge responses via Hessian determinant ratio */
static int is_not_edge(GrayF *dog_layer, int x, int y) {
    float Dxx = gat(dog_layer,x+1,y) + gat(dog_layer,x-1,y) - 2*gat(dog_layer,x,y);
    float Dyy = gat(dog_layer,x,y+1) + gat(dog_layer,x,y-1) - 2*gat(dog_layer,x,y);
    float Dxy = (gat(dog_layer,x+1,y+1) - gat(dog_layer,x-1,y+1)
               - gat(dog_layer,x+1,y-1) + gat(dog_layer,x-1,y-1)) / 4.0f;
    float trace = Dxx + Dyy;
    float det   = Dxx*Dyy - Dxy*Dxy;
    float r = EDGE_THRESHOLD;
    if (det <= 0) return 0;
    return (trace*trace / det) < ((r+1)*(r+1)/r);
}

/* ============================================================
 *  STEP 3C: ORIENTATION ASSIGNMENT
 *  Build 36-bin gradient orientation histogram in a patch
 *  around the keypoint; pick dominant peak(s).
 * ============================================================ */

static float compute_orientation(GrayF *blur, int x, int y, float sigma) {
    float hist[ORI_BINS];
    memset(hist, 0, sizeof(hist));
    int rad = (int)(3.0 * ORI_SIGMA_FACTOR * sigma);
    float weight_sigma = ORI_SIGMA_FACTOR * sigma;

    for (int dy = -rad; dy <= rad; dy++)
        for (int dx = -rad; dx <= rad; dx++) {
            float gx = gat(blur, x+dx+1, y+dy) - gat(blur, x+dx-1, y+dy);
            float gy = gat(blur, x+dx, y+dy+1) - gat(blur, x+dx, y+dy-1);
            float mag  = sqrtf(gx*gx + gy*gy);
            float ori  = atan2f(gy, gx);
            float w    = expf(-(float)(dx*dx+dy*dy)/(2.0f*weight_sigma*weight_sigma));
            int bin = (int)((ori + (float)PI) / (2.0f*(float)PI) * ORI_BINS);
            if (bin >= ORI_BINS) bin = 0;
            if (bin < 0) bin = ORI_BINS - 1;
            hist[bin] += w * mag;
        }

    /* Smooth the histogram */
    for (int i = 0; i < 6; i++) {
        float prev = hist[ORI_BINS-1];
        for (int b = 0; b < ORI_BINS; b++) {
            float tmp = hist[b];
            hist[b] = 0.25f*prev + 0.5f*hist[b]
                    + 0.25f*hist[(b+1)%ORI_BINS];
            prev = tmp;
        }
    }

    /* Find peak */
    int peak_bin = 0;
    for (int b = 1; b < ORI_BINS; b++)
        if (hist[b] > hist[peak_bin]) peak_bin = b;

    return ((float)peak_bin + 0.5f) / ORI_BINS * 2.0f * (float)PI - (float)PI;
}

/* ============================================================
 *  STEP 3D: COMPUTE SIFT DESCRIPTOR
 *  Sample a 16x16 patch rotated to the keypoint orientation.
 *  Divide into 4x4 cells of 8-bin orientation histograms.
 *  Normalise, clamp at 0.2, renormalise -> 128-dim vector.
 * ============================================================ */

static void compute_descriptor(GrayF *blur, float x, float y,
                                float scale, float angle,
                                float *desc) {
    memset(desc, 0, DESC_SIZE * sizeof(float));
    float cos_a = cosf(-angle), sin_a = sinf(-angle);
    float sbins = scale * 3.0f; /* pixel spacing per histogram cell */

    for (int gy = 0; gy < DESC_GRID; gy++)
        for (int gx = 0; gx < DESC_GRID; gx++)
            for (int sy = 0; sy < 4; sy++)
                for (int sx = 0; sx < 4; sx++) {
                    /* Sample point in rotated frame */
                    float rx = ((gx - 1.5f) * 4 + sx) * sbins / 4.0f;
                    float ry = ((gy - 1.5f) * 4 + sy) * sbins / 4.0f;
                    float px = x + cos_a * rx - sin_a * ry;
                    float py = y + sin_a * rx + cos_a * ry;

                    float dx = gat(blur, (int)px+1, (int)py) - gat(blur, (int)px-1, (int)py);
                    float dy = gat(blur, (int)px, (int)py+1) - gat(blur, (int)px, (int)py-1);
                    float mag = sqrtf(dx*dx + dy*dy);
                    float ori = atan2f(dy, dx) - angle;
                    while (ori < 0)          ori += 2.0f*(float)PI;
                    while (ori >= 2.0f*(float)PI) ori -= 2.0f*(float)PI;

                    int bin = (int)(ori / (2.0f*(float)PI) * DESC_HIST_BINS);
                    if (bin >= DESC_HIST_BINS) bin = 0;

                    int didx = (gy * DESC_GRID + gx) * DESC_HIST_BINS + bin;
                    desc[didx] += mag;
                }

    /* L2-normalise */
    float norm = 0;
    for (int i = 0; i < DESC_SIZE; i++) norm += desc[i]*desc[i];
    norm = sqrtf(norm) + 1e-7f;
    for (int i = 0; i < DESC_SIZE; i++) desc[i] /= norm;

    /* Clamp and renormalise */
    for (int i = 0; i < DESC_SIZE; i++)
        if (desc[i] > 0.2f) desc[i] = 0.2f;
    norm = 0;
    for (int i = 0; i < DESC_SIZE; i++) norm += desc[i]*desc[i];
    norm = sqrtf(norm) + 1e-7f;
    for (int i = 0; i < DESC_SIZE; i++) desc[i] /= norm;
}

/* ============================================================
 *  STEP 3 (MASTER): DETECT & DESCRIBE SIFT KEYPOINTS
 * ============================================================ */

int sift_detect(Image *img, Keypoint **kp_out) {
    printf("[SIFT] Converting to grayscale...\n");
    GrayF *base = rgb_to_gray(img);

    printf("[SIFT] Building DoG pyramid (%d octaves, %d scales)...\n",
           SIFT_OCTAVES, SIFT_SCALES);
    GrayF *dog[SIFT_OCTAVES][SIFT_SCALES+2];
    GrayF *gauss[SIFT_OCTAVES][SIFT_SCALES+3];
    build_dog_pyramid(base, dog, gauss);

    Keypoint *kps = malloc(MAX_KEYPOINTS * sizeof(Keypoint));
    int nkp = 0;

    printf("[SIFT] Detecting extrema in DoG pyramid...\n");
    for (int o = 0; o < SIFT_OCTAVES && nkp < MAX_KEYPOINTS; o++) {
        int w = dog[o][0]->w, h = dog[o][0]->h;
        float octave_scale = (float)(1 << o); /* pixel spacing for this octave */

        for (int s = 1; s <= SIFT_SCALES && nkp < MAX_KEYPOINTS; s++)
            for (int y = 1; y < h-1 && nkp < MAX_KEYPOINTS; y++)
                for (int x = 1; x < w-1 && nkp < MAX_KEYPOINTS; x++) {
                    if (!is_extremum(dog, o, s, x, y)) continue;
                    if (!is_not_edge(dog[o][s], x, y))  continue;

                    /* Map back to original image coordinates */
                    float scale = SIFT_SIGMA * powf(SIFT_K, (float)s);
                    float angle = compute_orientation(gauss[o][s], x, y, scale);

                    kps[nkp].x     = (float)x * octave_scale;
                    kps[nkp].y     = (float)y * octave_scale;
                    kps[nkp].scale = scale * octave_scale;
                    kps[nkp].angle = angle;

                    /* Use the Gaussian blur layer at this scale for descriptor */
                    compute_descriptor(gauss[o][s],
                                       (float)x, (float)y,
                                       scale, angle,
                                       kps[nkp].desc);
                    nkp++;
                }
    }

    /* Free pyramid */
    for (int o = 0; o < SIFT_OCTAVES; o++) {
        for (int s = 0; s < SIFT_SCALES+3; s++) free_gray(gauss[o][s]);
        for (int s = 0; s < SIFT_SCALES+2; s++) free_gray(dog[o][s]);
    }
    free_gray(base);

    printf("[SIFT] Found %d keypoints.\n", nkp);
    *kp_out = kps;
    return nkp;
}

/* ============================================================
 *  STEP 4: FEATURE MATCHING (Lowe's Ratio Test)
 *  For each keypoint in image 1, find the 2 nearest neighbours
 *  in image 2 using L2 distance. Accept if dist1/dist2 < ratio.
 * ============================================================ */

static float descriptor_dist(float *a, float *b) {
    float s = 0;
    for (int i = 0; i < DESC_SIZE; i++) { float d = a[i]-b[i]; s += d*d; }
    return s; /* squared L2 distance; ratio test is scale-invariant */
}

int match_features(Keypoint *kp1, int n1, Keypoint *kp2, int n2,
                   Match **matches_out) {
    printf("[MATCH] Running Lowe ratio test (threshold=%.2f)...\n", LOWE_RATIO);

    Match *matches = malloc(MAX_MATCHES * sizeof(Match));
    int nm = 0;

    for (int i = 0; i < n1 && nm < MAX_MATCHES; i++) {
        float best1 = FLT_MAX, best2 = FLT_MAX;
        int   idx1  = -1;

        for (int j = 0; j < n2; j++) {
            float d = descriptor_dist(kp1[i].desc, kp2[j].desc);
            if (d < best1) { best2 = best1; best1 = d; idx1 = j; }
            else if (d < best2) { best2 = d; }
        }

        if (best2 < 1e-10f) continue;
        if (sqrtf(best1) / sqrtf(best2) < LOWE_RATIO) {
            matches[nm].idx1 = i;
            matches[nm].idx2 = idx1;
            matches[nm].dist = sqrtf(best1);
            nm++;
        }
    }

    printf("[MATCH] Found %d matches after ratio test.\n", nm);
    *matches_out = matches;
    return nm;
}

/* ============================================================
 *  STEP 5A: COMPUTE HOMOGRAPHY VIA SVD
 *  Solve Ah = 0 using the Direct Linear Transform (DLT).
 *  SVD via Jacobi method (works fine for small 9x9 systems).
 * ============================================================ */

/* Simple Jacobi SVD for NxN symmetric matrix.
   Returns eigenvectors in V; eigenvalues in d.
   We only need the last column of V (smallest eigenvalue). */
static void jacobi_svd(double A[9][9], double d[9], double V[9][9], int n) {
    /* Initialise V = I */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            V[i][j] = (i==j) ? 1.0 : 0.0;

    /* Symmetric eigendecomposition (Jacobi) */
    for (int sweep = 0; sweep < 50; sweep++) {
        double off = 0;
        for (int i = 0; i < n; i++)
            for (int j = i+1; j < n; j++) off += A[i][j]*A[i][j];
        if (off < 1e-20) break;

        for (int p = 0; p < n-1; p++)
            for (int q = p+1; q < n; q++) {
                if (fabs(A[p][q]) < 1e-15) continue;
                double theta = 0.5*(A[q][q]-A[p][p]) / A[p][q];
                double t = (theta >= 0) ?
                    1.0/(theta + sqrt(1.0+theta*theta)) :
                    1.0/(theta - sqrt(1.0+theta*theta));
                double c = 1.0/sqrt(1.0+t*t), s = t*c;

                /* Update A */
                double App=A[p][p], Aqq=A[q][q], Apq=A[p][q];
                A[p][p] = App - t*Apq;
                A[q][q] = Aqq + t*Apq;
                A[p][q] = A[q][p] = 0;
                for (int r = 0; r < n; r++) {
                    if (r==p||r==q) continue;
                    double arp=A[r][p], arq=A[r][q];
                    A[r][p]=A[p][r]= c*arp - s*arq;
                    A[r][q]=A[q][r]= s*arp + c*arq;
                }
                /* Update V */
                for (int r = 0; r < n; r++) {
                    double vrp=V[r][p], vrq=V[r][q];
                    V[r][p] = c*vrp - s*vrq;
                    V[r][q] = s*vrp + c*vrq;
                }
            }
    }
    for (int i = 0; i < n; i++) d[i] = A[i][i];
}

/* Compute homography from 4 point correspondences using DLT.
   pts1, pts2: arrays of 4 (x,y) pairs.
   H: output 3x3 homography (row-major). */
// static int compute_homography_dlt(float pts1[][2], float pts2[][2], float H[9]) {
//     /* Build 8x9 matrix A */
//     double A[9][9] = {{0}};
//     for (int i = 0; i < 4; i++) {
//         double x1=pts1[i][0], y1=pts1[i][1];
//         double x2=pts2[i][0], y2=pts2[i][1];
//         /* row 1 */
//         double r0[9] = {0,-x1,-y1,-1,0,0,y2*x1,y2*y1,y2};
//         /* row 2 */
//         double r1[9] = {x1,y1,1,0,0,0,-x2*x1,-x2*y1,-x2};
//         /* Build A^T A */
//         for (int m = 0; m < 9; m++)
//             for (int n = 0; n < 9; n++) {
//                 A[m][n] += r0[m]*r0[n] + r1[m]*r1[n];
//             }
//     }

//     double d[9], V[9][9];
//     jacobi_svd(A, d, V, 9);

//     /* Find index of smallest eigenvalue */
//     int min_idx = 0;
//     for (int i = 1; i < 9; i++) if (d[i] < d[min_idx]) min_idx = i;

//     for (int i = 0; i < 9; i++) H[i] = (float)V[i][min_idx];

//     /* Normalise so H[8]=1 */
//     if (fabsf(H[8]) > 1e-10f) {
//         float s = H[8];
//         for (int i = 0; i < 9; i++) H[i] /= s;
//     }
//     return 1;
// }

static int compute_homography_dlt(float pts1[][2], float pts2[][2], float H[9]) {
    /* --- Hartley Normalization for numerical stability --- */
    /* Compute centroids */
    float cx1=0, cy1=0, cx2=0, cy2=0;
    for (int i = 0; i < 4; i++) {
        cx1 += pts1[i][0]; cy1 += pts1[i][1];
        cx2 += pts2[i][0]; cy2 += pts2[i][1];
    }
    cx1/=4; cy1/=4; cx2/=4; cy2/=4;

    /* Compute mean distance from centroid */
    float d1=0, d2=0;
    for (int i = 0; i < 4; i++) {
        d1 += sqrtf((pts1[i][0]-cx1)*(pts1[i][0]-cx1)+(pts1[i][1]-cy1)*(pts1[i][1]-cy1));
        d2 += sqrtf((pts2[i][0]-cx2)*(pts2[i][0]-cx2)+(pts2[i][1]-cy2)*(pts2[i][1]-cy2));
    }
    d1/=4; d2/=4;
    if (d1 < 1e-8f || d2 < 1e-8f) return 0;

    float s1 = sqrtf(2.0f)/d1, s2 = sqrtf(2.0f)/d2;

    /* Normalization matrices T1, T2 (3x3 row-major) */
    float T1[9] = { s1, 0, -s1*cx1,  0, s1, -s1*cy1,  0, 0, 1 };
    float T2[9] = { s2, 0, -s2*cx2,  0, s2, -s2*cy2,  0, 0, 1 };

    /* Normalize points */
    float np1[4][2], np2[4][2];
    for (int i = 0; i < 4; i++) {
        np1[i][0] = s1*(pts1[i][0]-cx1);
        np1[i][1] = s1*(pts1[i][1]-cy1);
        np2[i][0] = s2*(pts2[i][0]-cx2);
        np2[i][1] = s2*(pts2[i][1]-cy2);
    }

    /* --- Build 8x9 matrix A (DLT) --- */
    double A[9][9] = {{0}};
    for (int i = 0; i < 4; i++) {
        double x1=np1[i][0], y1=np1[i][1];
        double x2=np2[i][0], y2=np2[i][1];
        double r0[9] = { 0,  0,  0, -x1, -y1, -1,  y2*x1,  y2*y1,  y2};
        double r1[9] = {x1, y1,  1,   0,   0,  0, -x2*x1, -x2*y1, -x2};
        for (int m = 0; m < 9; m++)
            for (int n = 0; n < 9; n++)
                A[m][n] += r0[m]*r0[n] + r1[m]*r1[n];
    }

    /* --- Jacobi SVD --- */
    double d[9], V[9][9];
    jacobi_svd(A, d, V, 9);

    int min_idx = 0;
    for (int i = 1; i < 9; i++) if (d[i] < d[min_idx]) min_idx = i;

    /* Reshape into Hn (normalized homography) */
    float Hn[9];
    for (int i = 0; i < 9; i++) Hn[i] = (float)V[i][min_idx];

    /* --- Denormalize: H = T2_inv * Hn * T1 --- */
    /* T2_inv (inverse of T2) */
    float T2inv[9] = {
        1.0f/s2,    0,    cx2,
           0,    1.0f/s2, cy2,
           0,       0,     1
    };

    /* Hn * T1 -> tmp */
    float tmp[9] = {0};
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            for (int k = 0; k < 3; k++)
                tmp[r*3+c] += Hn[r*3+k] * T1[k*3+c];

    /* T2inv * tmp -> H */
    for (int i = 0; i < 9; i++) H[i] = 0;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            for (int k = 0; k < 3; k++)
                H[r*3+c] += T2inv[r*3+k] * tmp[k*3+c];

    /* Normalize so H[8] = 1 */
    if (fabsf(H[8]) > 1e-10f) {
        float s = H[8];
        for (int i = 0; i < 9; i++) H[i] /= s;
    }
    return 1;
}


/* Apply homography to a point */
static void apply_homography(float H[9], float x, float y, float *ox, float *oy) {
    float w  = H[6]*x + H[7]*y + H[8];
    *ox = (H[0]*x + H[1]*y + H[2]) / w;
    *oy = (H[3]*x + H[4]*y + H[5]) / w;
}

/* ============================================================
 *  STEP 5B: RANSAC TO FIND THE BEST HOMOGRAPHY
 * ============================================================ */

/* Simple LCG random number in [0, max) */
static unsigned int rand_state = 42;
static int rand_int(int max) {
    rand_state = rand_state * 1664525u + 1013904223u;
    return (int)((rand_state >> 1) % (unsigned)max);
}

int ransac_homography(Keypoint *kp1, Keypoint *kp2,
                      Match *matches, int nm,
                      float H_out[9]) {
    printf("[RANSAC] Running %d iterations (threshold=%.1fpx)...\n",
           RANSAC_ITERS, RANSAC_THRESH);

    if (nm < 4) { printf("[RANSAC] Too few matches.\n"); return 0; }

    int best_inliers = 0;
    float best_H[9];

    for (int iter = 0; iter < RANSAC_ITERS; iter++) {
        /* Pick 4 random distinct matches */
        int idx[4];
        for (int k = 0; k < 4; k++) {
            int r;
            int dup;
            do {
                r = rand_int(nm);
                dup = 0;
                for (int l = 0; l < k; l++) if (idx[l]==r) { dup=1; break; }
            } while (dup);
            idx[k] = r;
        }

        float p1[4][2], p2[4][2];
        for (int k = 0; k < 4; k++) {
            p1[k][0] = kp1[matches[idx[k]].idx1].x;
            p1[k][1] = kp1[matches[idx[k]].idx1].y;
            p2[k][0] = kp2[matches[idx[k]].idx2].x;
            p2[k][1] = kp2[matches[idx[k]].idx2].y;
        }

        float H[9];
        if (!compute_homography_dlt(p1, p2, H)) continue;

        /* Count inliers */
        int inliers = 0;
        for (int m = 0; m < nm; m++) {
            float x1 = kp1[matches[m].idx1].x, y1 = kp1[matches[m].idx1].y;
            float x2 = kp2[matches[m].idx2].x, y2 = kp2[matches[m].idx2].y;
            float px, py;
            apply_homography(H, x1, y1, &px, &py);
            float err = sqrtf((px-x2)*(px-x2) + (py-y2)*(py-y2));
            if (err < RANSAC_THRESH) inliers++;
        }

        if (inliers > best_inliers) {
            best_inliers = inliers;
            memcpy(best_H, H, 9 * sizeof(float));
        }
    }

    printf("[RANSAC] Best model: %d inliers out of %d matches.\n",
           best_inliers, nm);

    if (best_inliers < MIN_INLIERS) {
        printf("[RANSAC] Not enough inliers to stitch.\n");
        return 0;
    }

    memcpy(H_out, best_H, 9 * sizeof(float));
    return best_inliers;
}

/* ============================================================
 *  STEP 6: PERSPECTIVE WARP + BLEND
 *  Warp image1 onto the plane of image2 using H.
 *  Canvas is sized to fit both images.
 *  Simple linear blend in the overlap region.
 * ============================================================ */

Image *perspective_warp_and_blend(Image *img1, Image *img2, float H[9]) {
    printf("[WARP] Computing output canvas size...\n");

    int w1=img1->width, h1=img1->height;
    int w2=img2->width, h2=img2->height;

    /* Project corners of img1 through H */
    float corners[4][2] = {{0,0},{(float)(w1-1),0},
                            {0,(float)(h1-1)},{(float)(w1-1),(float)(h1-1)}};
    float minx=0, miny=0, maxx=(float)(w2-1), maxy=(float)(h2-1);

    for (int c = 0; c < 4; c++) {
        float px, py;
        apply_homography(H, corners[c][0], corners[c][1], &px, &py);
        if (px < minx) minx = px;
        if (py < miny) miny = py;
        if (px > maxx) maxx = px;
        if (py > maxy) maxy = py;
    }

    int out_w = (int)ceilf(maxx - minx) + 1;
    int out_h = (int)ceilf(maxy - miny) + 1;
    printf("[WARP] Output canvas: %d x %d\n", out_w, out_h);

    /* Translation to shift canvas so top-left is at (0,0) */
    float tx = -minx, ty = -miny;

    Image *out = malloc(sizeof(Image));
    out->width = out_w; out->height = out_h;
    out->data = calloc((size_t)out_w * out_h * 3, 1);

    /* Compute inverse of H for backward mapping */
    /* Inverse of 3x3 via cofactors */
    float *h = H;
    float det = h[0]*(h[4]*h[8]-h[5]*h[7])
              - h[1]*(h[3]*h[8]-h[5]*h[6])
              + h[2]*(h[3]*h[7]-h[4]*h[6]);
    float Hinv[9];
    Hinv[0] = (h[4]*h[8]-h[5]*h[7])/det;
    Hinv[1] = (h[2]*h[7]-h[1]*h[8])/det;
    Hinv[2] = (h[1]*h[5]-h[2]*h[4])/det;
    Hinv[3] = (h[5]*h[6]-h[3]*h[8])/det;
    Hinv[4] = (h[0]*h[8]-h[2]*h[6])/det;
    Hinv[5] = (h[2]*h[3]-h[0]*h[5])/det;
    Hinv[6] = (h[3]*h[7]-h[4]*h[6])/det;
    Hinv[7] = (h[1]*h[6]-h[0]*h[7])/det;
    Hinv[8] = (h[0]*h[4]-h[1]*h[3])/det;

    printf("[WARP] Warping image 1 onto canvas (backward mapping)...\n");

    /* ---- Warp image 1 through H ---- */
    /* Also store a float accumulation buffer for blending */
    float *acc  = calloc((size_t)out_w * out_h * 3, sizeof(float));
    float *wsum = calloc((size_t)out_w * out_h,     sizeof(float));

    for (int y = 0; y < out_h; y++)
        for (int x = 0; x < out_w; x++) {
            /* Map canvas pixel back to img1 */
            float cx = (float)x - tx, cy = (float)y - ty;
            float sx, sy;
            apply_homography(Hinv, cx, cy, &sx, &sy);

            if (sx < 0 || sx >= w1-1 || sy < 0 || sy >= h1-1) continue;

            /* Bilinear interpolation in img1 */
            int ix = (int)sx, iy = (int)sy;
            float fx = sx - ix, fy = sy - iy;
            for (int c = 0; c < 3; c++) {
                float v =
                    img1->data[(iy   * w1 + ix  )*3+c] * (1-fx)*(1-fy) +
                    img1->data[(iy   * w1 + ix+1)*3+c] *    fx *(1-fy) +
                    img1->data[((iy+1)*w1 + ix  )*3+c] * (1-fx)*fy +
                    img1->data[((iy+1)*w1 + ix+1)*3+c] *    fx *fy;
                acc [(y * out_w + x)*3 + c] += v;
            }
            wsum[y * out_w + x] += 1.0f;
        }

    printf("[WARP] Placing image 2 onto canvas...\n");

    /* ---- Place image 2 directly (with translation) ---- */
    for (int y = 0; y < h2; y++)
        for (int x = 0; x < w2; x++) {
            int ox = (int)(x + tx), oy = (int)(y + ty);
            if (ox < 0 || ox >= out_w || oy < 0 || oy >= out_h) continue;
            for (int c = 0; c < 3; c++)
                acc [(oy * out_w + ox)*3 + c] += img2->data[(y*w2+x)*3+c];
            wsum[oy * out_w + ox] += 1.0f;
        }

    /* ---- Blend (weighted average) ---- */
    for (int i = 0; i < out_w * out_h; i++) {
        float w = wsum[i];
        if (w > 0) {
            out->data[3*i+0] = (unsigned char)(acc[3*i+0]/w);
            out->data[3*i+1] = (unsigned char)(acc[3*i+1]/w);
            out->data[3*i+2] = (unsigned char)(acc[3*i+2]/w);
        }
    }

    free(acc); free(wsum);
    return out;
}

/* ============================================================
 *  VISUALISATION HELPERS
 * ============================================================ */

/* Draw a coloured circle on an RGB image (no external library) */
static void draw_circle(Image *img, int cx, int cy, int r,
                         unsigned char R, unsigned char G, unsigned char B) {
    for (int y = cy-r; y <= cy+r; y++)
        for (int x = cx-r; x <= cx+r; x++) {
            int dx=x-cx, dy=y-cy;
            if (dx*dx+dy*dy <= r*r && dx*dx+dy*dy >= (r-1)*(r-1)) {
                if (x<0||x>=img->width||y<0||y>=img->height) continue;
                img->data[(y*img->width+x)*3+0] = R;
                img->data[(y*img->width+x)*3+1] = G;
                img->data[(y*img->width+x)*3+2] = B;
            }
        }
}

/* Draw a line (Bresenham) */
static void draw_line(Image *img, int x0, int y0, int x1, int y1,
                       unsigned char R, unsigned char G, unsigned char B) {
    int dx=abs(x1-x0), dy=abs(y1-y0);
    int sx=(x0<x1)?1:-1, sy=(y0<y1)?1:-1;
    int err=dx-dy;
    for (;;) {
        if (x0>=0&&x0<img->width&&y0>=0&&y0<img->height) {
            img->data[(y0*img->width+x0)*3+0] = R;
            img->data[(y0*img->width+x0)*3+1] = G;
            img->data[(y0*img->width+x0)*3+2] = B;
        }
        if (x0==x1&&y0==y1) break;
        int e2=2*err;
        if (e2>-dy){err-=dy; x0+=sx;}
        if (e2< dx){err+=dx; y0+=sy;}
    }
}

/* Save image 1 with keypoints drawn (green circles) */
void save_keypoints_image(Image *img, Keypoint *kps, int n,
                           const char *path) {
    printf("[VIS] Saving keypoints image: %s (%d keypoints)\n", path, n);
    Image *out = clone_image(img);
    for (int i = 0; i < n; i++) {
        int cx = (int)kps[i].x, cy = (int)kps[i].y;
        /* Draw orientation tick */
        int ex = cx + (int)(CIRCLE_RADIUS * cosf(kps[i].angle));
        int ey = cy + (int)(CIRCLE_RADIUS * sinf(kps[i].angle));
        draw_line(out, cx, cy, ex, ey, 255, 255, 0);
        draw_circle(out, cx, cy, CIRCLE_RADIUS, 0, 255, 0);
    }
    save_ppm(path, out);
    free_image(out);
}

/* Save side-by-side match visualisation with coloured lines */
void save_matches_image(Image *img1, Image *img2,
                         Keypoint *kp1, Keypoint *kp2,
                         Match *matches, int nm,
                         const char *path) {
    printf("[VIS] Saving match lines image: %s (%d matches)\n", path, nm);

    int w1=img1->width, h1=img1->height;
    int w2=img2->width, h2=img2->height;
    int out_w = w1 + w2, out_h = (h1 > h2) ? h1 : h2;

    Image *out = malloc(sizeof(Image));
    out->width = out_w; out->height = out_h;
    out->data = calloc((size_t)out_w * out_h * 3, 1);

    /* Copy img1 to left */
    for (int y = 0; y < h1; y++)
        for (int x = 0; x < w1; x++) {
            int oi = (y*out_w+x)*3;
            int ii = (y*w1+x)*3;
            out->data[oi+0]=img1->data[ii+0];
            out->data[oi+1]=img1->data[ii+1];
            out->data[oi+2]=img1->data[ii+2];
        }

    /* Copy img2 to right */
    for (int y = 0; y < h2; y++)
        for (int x = 0; x < w2; x++) {
            int oi = (y*out_w + w1+x)*3;
            int ii = (y*w2+x)*3;
            out->data[oi+0]=img2->data[ii+0];
            out->data[oi+1]=img2->data[ii+1];
            out->data[oi+2]=img2->data[ii+2];
        }

    /* Draw match lines with cycling colours */
    unsigned char colours[][3] = {
        {255,0,0},{0,255,255},{255,165,0},{255,0,255},{0,200,0}
    };
    int nc = 5;
    for (int m = 0; m < nm; m++) {
        int x1 = (int)kp1[matches[m].idx1].x;
        int y1 = (int)kp1[matches[m].idx1].y;
        int x2 = (int)kp2[matches[m].idx2].x + w1;
        int y2 = (int)kp2[matches[m].idx2].y;
        unsigned char *col = colours[m % nc];
        draw_line(out, x1, y1, x2, y2, col[0], col[1], col[2]);
        draw_circle(out, x1, y1, 4, col[0], col[1], col[2]);
        draw_circle(out, x2, y2, 4, col[0], col[1], col[2]);
    }

    save_ppm(path, out);
    free_image(out);
}

/* ============================================================
 *  MAIN
 * ============================================================ */

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s image1.ppm image2.ppm\n", argv[0]);
        return 1;
    }

    /* ---- Load images ---- */
    printf("=== Loading images ===\n");
    Image *img1 = load_ppm(argv[1]);
    Image *img2 = load_ppm(argv[2]);
    if (!img1 || !img2) return 1;
    printf("Image 1: %dx%d\n", img1->width, img1->height);
    printf("Image 2: %dx%d\n", img2->width, img2->height);

    /* ---- STEP 3: SIFT on both images ---- */
    printf("\n=== STEP 3: SIFT Feature Detection & Description ===\n");
    Keypoint *kp1, *kp2;
    printf("--- Processing Image 1 ---\n");
    int n1 = sift_detect(img1, &kp1);
    printf("--- Processing Image 2 ---\n");
    int n2 = sift_detect(img2, &kp2);

    /* Save keypoint visualisations */
    save_keypoints_image(img1, kp1, n1, "keypoints1.ppm");
    save_keypoints_image(img2, kp2, n2, "keypoints2.ppm");

    /* ---- STEP 4: Feature Matching ---- */
    printf("\n=== STEP 4: Feature Matching (Lowe Ratio Test) ===\n");
    Match *matches;
    int nm = match_features(kp1, n1, kp2, n2, &matches);

    if (nm < 4) {
        fprintf(stderr, "Not enough matches to compute homography (%d).\n", nm);
        return 1;
    }

    /* Save match visualisation */
    save_matches_image(img1, img2, kp1, kp2, matches, nm, "matches.ppm");

    /* ---- STEP 5: Homography via RANSAC + SVD/DLT ---- */
    printf("\n=== STEP 5: Homography Estimation (RANSAC + DLT/SVD) ===\n");
    float H[9];
    int inliers = ransac_homography(kp1, kp2, matches, nm, H);
    if (!inliers) {
        fprintf(stderr, "Homography estimation failed.\n");
        return 1;
    }

    printf("[HOMO] Homography matrix:\n");
    for (int r = 0; r < 3; r++)
        printf("       [ %8.4f %8.4f %8.4f ]\n", H[r*3], H[r*3+1], H[r*3+2]);

    /* ---- STEP 6: Perspective Warp & Blend ---- */
    printf("\n=== STEP 6: Perspective Warp & Panorama Blend ===\n");
    Image *result = perspective_warp_and_blend(img1, img2, H);
    save_ppm("stitched.ppm", result);

    /* ---- Summary ---- */
    printf("\n=== Done! Output files ===\n");
    printf("  keypoints1.ppm  - image1 with SIFT keypoints\n");
    printf("  keypoints2.ppm  - image2 with SIFT keypoints\n");
    printf("  matches.ppm     - feature match lines\n");
    printf("  stitched.ppm    - final stitched panorama\n");

    /* Clean up */
    free(kp1); free(kp2); free(matches);
    free_image(img1); free_image(img2); free_image(result);
    return 0;
}