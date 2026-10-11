/* 6CS005 Task 4. Usage: ./sobel output_directory image1.png image2.png ...
 * RGBA -> weighted grayscale -> zero-padded 3x3 Sobel.
 * Truncate the Euclidean magnitude, clamp to 255, preserve alpha.
 * PNG I/O follows the lecturer's week 9 LodePNG example.
 */
#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <time.h>
/* LodePNG is compiled as C; nvcc compiles this translation unit as C++. */
#define LODEPNG_NO_COMPILE_CPP
extern "C" {
#include "lodepng.h"
}

__host__ __device__ int grayscale(const unsigned char *image, int x, int y, unsigned int width, unsigned int height) {
    if (x < 0 || y < 0 || (unsigned int)x >= width || (unsigned int)y >= height) return 0;
    size_t index = ((size_t)y * width + x) * 4;
    return (30*(int)image[index] + 59*(int)image[index+1] + 11*(int)image[index+2]) / 100;
}
__global__ void sobelFilter(const unsigned char *input, unsigned char *output, unsigned char *gradient_x, unsigned char *gradient_y, unsigned int width, unsigned int height) {
    size_t pixel = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= (size_t)width * height) return;
    int x = (int)(pixel % width), y = (int)(pixel / width);
    const int gx_kernel[9] = {-1,0,1,-2,0,2,-1,0,1};
    const int gy_kernel[9] = {-1,-2,-1,0,0,0,1,2,1};
    int gx = 0, gy = 0;
    for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        int gray = grayscale(input, x+dx, y+dy, width, height);
        int k = (dy+1)*3+dx+1;
        gx += gray * gx_kernel[k]; gy += gray * gy_kernel[k];
    }
    int magnitude = (int)sqrtf((float)(gx*gx + gy*gy));
    unsigned char edge = (unsigned char)(magnitude > 255 ? 255 : magnitude);
    output[pixel*4] = edge; output[pixel*4+1] = edge; output[pixel*4+2] = edge;
    output[pixel*4+3] = input[pixel*4+3];
    gradient_x[pixel] = (unsigned char)(abs(gx) > 255 ? 255 : abs(gx));
    gradient_y[pixel] = (unsigned char)(abs(gy) > 255 ? 255 : abs(gy));
}
/* Serial check uses the expanded mask equations, independently of kernel loops. */
static void sobel_cpu(const unsigned char *input, unsigned char *output,
                      unsigned char *gradient_x, unsigned char *gradient_y,
                      unsigned int width, unsigned int height) {
    for (unsigned int row = 0; row < height; row++) {
        for (unsigned int col = 0; col < width; col++) {
            int x = (int)col, y = (int)row;
            int a = grayscale(input,x-1,y-1,width,height);
            int b = grayscale(input,x,y-1,width,height);
            int c = grayscale(input,x+1,y-1,width,height);
            int d = grayscale(input,x-1,y,width,height);
            int f = grayscale(input,x+1,y,width,height);
            int g = grayscale(input,x-1,y+1,width,height);
            int h = grayscale(input,x,y+1,width,height);
            int i = grayscale(input,x+1,y+1,width,height);
            int gx = -a+c-2*d+2*f-g+i, gy = -a-2*b-c+g+2*h+i;
            size_t pixel = (size_t)row*width+col;
            int magnitude = (int)sqrtf((float)(gx*gx+gy*gy));
            unsigned char edge = (unsigned char)(magnitude > 255 ? 255 : magnitude);
            output[pixel*4] = output[pixel*4+1] = output[pixel*4+2] = edge;
            output[pixel*4+3] = input[pixel*4+3];
            gradient_x[pixel] = (unsigned char)(abs(gx)>255 ? 255 : abs(gx));
            gradient_y[pixel] = (unsigned char)(abs(gy)>255 ? 255 : abs(gy));
        }
    }
}
static const char *basename_of(const char *path) {
    const char *base = path;
    for (; *path; path++) if (*path == '/' || *path == '\\') base = path+1;
    return base;
}
static double wall_time(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}
#define CUDA_TRY(expression) do { cudaError_t status = (expression); \
    if (status != cudaSuccess) { fprintf(stderr, "%s: %s\n", #expression, cudaGetErrorString(status)); failed = 1; goto cleanup; } } while (0)

static int process_image(const char *path, const char *directory) {
    unsigned char *host_input = NULL, *host_output = NULL, *device_input = NULL, *device_output = NULL;
    unsigned char *host_x = NULL, *host_y = NULL, *device_x = NULL, *device_y = NULL;
    unsigned char *reference = NULL, *reference_x = NULL, *reference_y = NULL;
    unsigned int width = 0, height = 0, png_error = 0, threads = 0, blocks = 0;
    size_t pixels = 0, bytes = 0, name_size = 0;
    const char *prefixes[3] = {"sobel_", "gradient_x_", "gradient_y_"};
    char *output_name = NULL;
    cudaDeviceProp properties;
    cudaEvent_t start_event = NULL, finish_event = NULL;
    float kernel_ms = 0;
    double cpu_start = 0, cpu_seconds = 0;
    double start = wall_time(), transfer_start = 0, device_seconds = 0;
    int failed = 0;
    png_error = lodepng_decode32_file(&host_input, &width, &height, path);
    if (png_error) { fprintf(stderr, "%s: PNG decode %u (%s)\n", path, png_error, lodepng_error_text(png_error)); failed = 1; goto cleanup; }
    if (!width || !height || width >= INT_MAX || height >= INT_MAX || (size_t)width > SIZE_MAX / height / 12) {
        fprintf(stderr, "%s: image dimensions exceed supported size.\n", path); failed = 1; goto cleanup;
    }
    pixels = (size_t)width * height; bytes = pixels * 4;
    host_output = (unsigned char *)malloc(bytes);
    host_x = (unsigned char *)malloc(pixels); host_y = (unsigned char *)malloc(pixels);
    reference = (unsigned char *)malloc(bytes);
    reference_x = (unsigned char *)malloc(pixels); reference_y = (unsigned char *)malloc(pixels);
    name_size = strlen(directory) + strlen(basename_of(path)) + 16;
    output_name = (char *)malloc(name_size);
    if (!host_output || !host_x || !host_y || !reference || !reference_x || !reference_y || !output_name) { fprintf(stderr, "Host allocation failed.\n"); failed = 1; goto cleanup; }
    snprintf(output_name, name_size, "%s/sobel_%s", directory, basename_of(path));
    CUDA_TRY(cudaGetDeviceProperties(&properties, 0));
    threads = properties.maxThreadsPerBlock < 256 ? properties.maxThreadsPerBlock : 256;
    if ((pixels-1)/threads+1 > (size_t)properties.maxGridSize[0]) { fprintf(stderr, "Image exceeds x-grid limit.\n"); failed = 1; goto cleanup; }
    blocks = (unsigned int)((pixels-1)/threads+1);
    CUDA_TRY(cudaMalloc((void **)&device_input, bytes)); CUDA_TRY(cudaMalloc((void **)&device_output, bytes));
    CUDA_TRY(cudaMalloc((void **)&device_x, pixels)); CUDA_TRY(cudaMalloc((void **)&device_y, pixels));
    CUDA_TRY(cudaEventCreate(&start_event)); CUDA_TRY(cudaEventCreate(&finish_event));
    transfer_start = wall_time();
    CUDA_TRY(cudaMemcpy(device_input, host_input, bytes, cudaMemcpyHostToDevice));
    CUDA_TRY(cudaEventRecord(start_event));
    sobelFilter<<<blocks, threads>>>(device_input, device_output, device_x, device_y, width, height);
    CUDA_TRY(cudaGetLastError()); CUDA_TRY(cudaEventRecord(finish_event));
    CUDA_TRY(cudaEventSynchronize(finish_event));
    CUDA_TRY(cudaEventElapsedTime(&kernel_ms, start_event, finish_event));
    CUDA_TRY(cudaMemcpy(host_output, device_output, bytes, cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(host_x, device_x, pixels, cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(host_y, device_y, pixels, cudaMemcpyDeviceToHost));
    device_seconds = wall_time() - transfer_start;
    cpu_start = wall_time();
    sobel_cpu(host_input, reference, reference_x, reference_y, width, height);
    cpu_seconds = wall_time() - cpu_start;
    if (memcmp(host_output, reference, bytes) || memcmp(host_x, reference_x, pixels) || memcmp(host_y, reference_y, pixels)) {
        fprintf(stderr, "%s: CPU/GPU pixel verification failed.\n", path); failed = 1; goto cleanup;
    }
    for (int plane = 0; plane < 3; plane++) {
        snprintf(output_name, name_size, "%s/%s%s", directory, prefixes[plane], basename_of(path));
        if (plane == 0) png_error = lodepng_encode32_file(output_name, host_output, width, height);
        else png_error = lodepng_encode_file(output_name, plane == 1 ? host_x : host_y, width, height, LCT_GREY, 8);
        if (png_error) { fprintf(stderr, "PNG encode %u (%s)\n", png_error, lodepng_error_text(png_error)); failed = 1; goto cleanup; }
    }
    printf("%s: %u x %u | Blocks: %u | Threads/block: %u\n", basename_of(path), width, height, blocks, threads);
    printf("  GPU: %s | Kernel: %.3f ms | Transfers + kernel: %.6f s | Decode-to-encode: %.6f s\n", properties.name, kernel_ms, device_seconds, wall_time()-start);
    printf("  CPU compute: %.6f s | CPU/GPU pixel verification: PASS\n", cpu_seconds);
    for (int plane = 0; plane < 3; plane++)
        printf("  Output: %s/%s%s\n", directory, prefixes[plane], basename_of(path));
cleanup:
    if (device_input && cudaFree(device_input) != cudaSuccess) failed = 1;
    if (device_output && cudaFree(device_output) != cudaSuccess) failed = 1;
    if (device_x && cudaFree(device_x) != cudaSuccess) failed = 1;
    if (device_y && cudaFree(device_y) != cudaSuccess) failed = 1;
    if (start_event && cudaEventDestroy(start_event) != cudaSuccess) failed = 1;
    if (finish_event && cudaEventDestroy(finish_event) != cudaSuccess) failed = 1;
    free(host_input); free(host_output); free(output_name);
    free(host_x); free(host_y); free(reference); free(reference_x); free(reference_y);
    return !failed;
}
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s output_directory image1.png [image2.png ...]\n", argv[0]); return 1; }
    /* Reject duplicate basenames rather than silently overwriting an output. */
    for (int i = 2; i < argc; i++) for (int j = 2; j < i; j++)
        if (!strcmp(basename_of(argv[i]), basename_of(argv[j]))) { fprintf(stderr, "Duplicate image basename: %s\n", basename_of(argv[i])); return 1; }
    if (mkdir(argv[1], 0777) && errno != EEXIST) { perror(argv[1]); return 1; }
    size_t successful = 0;
    for (int i = 2; i < argc; i++) if (process_image(argv[i], argv[1])) successful++;
    printf("Images completed: %zu/%d\n", successful, argc-2);
    return successful == (size_t)(argc-2) ? 0 : 1;
}
