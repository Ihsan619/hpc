/* 6CS005 Task 3. Usage: ./password_cracking passwords.txt
 * One GPU thread uses nested letter/digit loops for one encrypted line.
 * The transform is copied from the lecturer's CryptForCuda.c, without changing
 * the boundary rules. Each invocation writes to a private buffer.
 */
#include <cuda_runtime.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

__host__ __device__ void cudaCrypt(const char *raw, char *encrypted) {
    encrypted[0] = raw[0] + 2; encrypted[1] = raw[0] - 2; encrypted[2] = raw[0] + 1;
    encrypted[3] = raw[1] + 3; encrypted[4] = raw[1] - 3; encrypted[5] = raw[1] - 1;
    encrypted[6] = raw[2] + 2; encrypted[7] = raw[2] - 2;
    encrypted[8] = raw[3] + 4; encrypted[9] = raw[3] - 4; encrypted[10] = '\0';
    for (int i = 0; i < 10; i++) {
        if (i < 6) {
            if (encrypted[i] > 122) encrypted[i] = (encrypted[i] - 122) + 97;
            else if (encrypted[i] < 97) encrypted[i] = (97 - encrypted[i]) + 97;
        } else {
            if (encrypted[i] > 57) encrypted[i] = (encrypted[i] - 57) + 48;
            else if (encrypted[i] < 48) encrypted[i] = (48 - encrypted[i]) + 48;
        }
    }
}
__device__ int same_characters(const char *left, const char *right, int count) {
    for (int i = 0; i < count; i++) if (left[i] != right[i]) return 0;
    return 1;
}
__global__ void crackPasswords(const char *input, char *output, int *found, size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const char *target = input + index * 11;
    char raw[5] = {'a', 'a', '0', '0', '\0'}, encrypted[11];
    found[index] = 0;
    for (int i = 0; i < 5; i++) output[index*5+i] = '\0';
    for (int first = 0; first < 26; first++) {
        raw[0] = (char)('a' + first);
        for (int second = 0; second < 26; second++) {
            raw[1] = (char)('a' + second);
            cudaCrypt(raw, encrypted);
            /* The first six cipher characters depend only on the letters. */
            if (!same_characters(encrypted, target, 6)) continue;
            for (int tens = 0; tens < 10; tens++) {
                raw[2] = (char)('0' + tens);
                for (int units = 0; units < 10; units++) {
                    raw[3] = (char)('0' + units);
                    cudaCrypt(raw, encrypted);
                    if (same_characters(encrypted, target, 10)) {
                        for (int i = 0; i < 5; i++) output[index*5+i] = raw[i];
                        found[index] = 1;
                        return;
                    }
                }
            }
        }
    }
}
static double wall_time(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}
#define CUDA_TRY(expression) do { cudaError_t status = (expression); \
    if (status != cudaSuccess) { fprintf(stderr, "%s: %s\n", #expression, cudaGetErrorString(status)); failed = 1; goto cleanup; } } while (0)

int main(int argc, char **argv) {
    FILE *input = NULL, *output_file = NULL;
    char *host_input = NULL, *device_input = NULL;
    char *host_output = NULL, *device_output = NULL;
    int *host_found = NULL, *device_found = NULL;
    cudaEvent_t start_event = NULL, finish_event = NULL;
    cudaDeviceProp properties;
    size_t count = 0, capacity = 0, line_number = 0, not_found = 0;
    unsigned int threads = 0, blocks = 0;
    int failed = 0;
    float kernel_ms = 0;
    double transfer_start = 0, device_seconds = 0;
    char line[32], verified[11];
    if (argc != 2) { fprintf(stderr, "Usage: %s passwords.txt\n", argv[0]); return 1; }
    input = fopen(argv[1], "r");
    if (!input) { perror(argv[1]); return 1; }
    while (fgets(line, sizeof(line), input)) {
        line_number++;
        size_t length = strcspn(line, "\r\n");
        if (length != 10) { fprintf(stderr, "Line %zu: expected exactly 10 encrypted characters.\n", line_number); failed = 1; goto cleanup; }
        for (size_t i = 0; i < 10; i++) {
            if ((i < 6 && (line[i] < 'a' || line[i] > 'z')) || (i >= 6 && (line[i] < '0' || line[i] > '9'))) {
                fprintf(stderr, "Line %zu: invalid encrypted character.\n", line_number); failed = 1; goto cleanup;
            }
        }
        if (count == capacity) {
            size_t next = capacity ? capacity * 2 : 128;
            if (next < capacity || next > SIZE_MAX / 11) { failed = 1; goto cleanup; }
            char *larger = (char *)realloc(host_input, next * 11);
            if (!larger) { fprintf(stderr, "Host input allocation failed.\n"); failed = 1; goto cleanup; }
            host_input = larger; capacity = next;
        }
        memcpy(host_input+count*11, line, 10); host_input[count*11+10] = '\0'; count++;
    }
    if (ferror(input)) { perror("Reading passwords"); failed = 1; goto cleanup; }
    if (fclose(input)) { input = NULL; failed = 1; goto cleanup; } input = NULL;
    if (!count || count > SIZE_MAX / 5) { fprintf(stderr, "Password file must be nonempty and fit in memory.\n"); failed = 1; goto cleanup; }
    host_output = (char *)malloc(count * 5);
    host_found = (int *)malloc(count * sizeof(int));
    if (!host_output || !host_found) { failed = 1; goto cleanup; }
    CUDA_TRY(cudaGetDeviceProperties(&properties, 0));
    threads = count < 512 ? (unsigned int)((count+1)/2) : 256;
    if (threads > (unsigned int)properties.maxThreadsPerBlock) threads = properties.maxThreadsPerBlock;
    if ((count - 1) / threads + 1 > (size_t)properties.maxGridSize[0]) { fprintf(stderr, "Input exceeds x-grid limit.\n"); failed = 1; goto cleanup; }
    blocks = (unsigned int)((count - 1) / threads + 1);
    if (blocks < 2) blocks = 2;
    CUDA_TRY(cudaMalloc((void **)&device_input, count * 11));
    CUDA_TRY(cudaMalloc((void **)&device_output, count * 5));
    CUDA_TRY(cudaMalloc((void **)&device_found, count * sizeof(int)));
    CUDA_TRY(cudaEventCreate(&start_event)); CUDA_TRY(cudaEventCreate(&finish_event));
    transfer_start = wall_time();
    CUDA_TRY(cudaMemcpy(device_input, host_input, count * 11, cudaMemcpyHostToDevice));
    CUDA_TRY(cudaEventRecord(start_event));
    crackPasswords<<<blocks, threads>>>(device_input, device_output, device_found, count);
    CUDA_TRY(cudaGetLastError()); CUDA_TRY(cudaEventRecord(finish_event));
    CUDA_TRY(cudaEventSynchronize(finish_event));
    CUDA_TRY(cudaEventElapsedTime(&kernel_ms, start_event, finish_event));
    CUDA_TRY(cudaMemcpy(host_output, device_output, count * 5, cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(host_found, device_found, count * sizeof(int), cudaMemcpyDeviceToHost));
    device_seconds = wall_time() - transfer_start;
    /* Re-encryption on the host detects transfer/indexing errors before writing. */
    for (size_t i = 0; i < count; i++) {
        if (!host_found[i]) { not_found++; continue; }
        host_output[i*5+4] = '\0'; cudaCrypt(host_output+i*5, verified);
        if (strcmp(verified, host_input+i*11)) { fprintf(stderr, "Verification failed at line %zu.\n", i+1); failed = 1; goto cleanup; }
    }
    output_file = fopen("decrypted.txt", "w");
    if (!output_file) { perror("decrypted.txt"); failed = 1; goto cleanup; }
    for (size_t i = 0; i < count; i++)
        if (fprintf(output_file, "%s\n", host_found[i] ? host_output+i*5 : "NOT_FOUND") < 0) { failed = 1; goto cleanup; }
    if (fclose(output_file)) { output_file = NULL; failed = 1; goto cleanup; } output_file = NULL;
    printf("GPU: %s | Passwords: %zu | Blocks: %u | Threads/block: %u\n", properties.name, count, blocks, threads);
    printf("Kernel: %.3f ms | Transfers + kernel: %.6f s | Recovered: %zu | Not found: %zu\n", kernel_ms, device_seconds, count-not_found, not_found);
    printf("Output: decrypted.txt | Host re-encryption verification passed\n");
cleanup:
    if (input && fclose(input)) failed = 1;
    if (output_file && fclose(output_file)) failed = 1;
    if (device_input && cudaFree(device_input) != cudaSuccess) failed = 1;
    if (device_output && cudaFree(device_output) != cudaSuccess) failed = 1;
    if (device_found && cudaFree(device_found) != cudaSuccess) failed = 1;
    if (start_event && cudaEventDestroy(start_event) != cudaSuccess) failed = 1;
    if (finish_event && cudaEventDestroy(finish_event) != cudaSuccess) failed = 1;
    free(host_input); free(host_output); free(host_found);
    return failed ? 1 : not_found ? 2 : 0;
}
