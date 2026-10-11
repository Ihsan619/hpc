/* 6CS005 Task 3. Usage: ./password_cracking passwords.txt
 * One GPU thread searches the 26*26*10*10 candidates for one encrypted line.
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

typedef struct { char text[11]; } Encrypted;
typedef struct { char text[5]; } Plain;

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
__global__ void crackPasswords(const Encrypted *input, Plain *output, size_t count) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    char raw[5] = {0}, encrypted[11];
    output[index].text[0] = '\0';
    for (int candidate = 0; candidate < 67600; candidate++) {
        raw[0] = 'a' + candidate / 2600;
        raw[1] = 'a' + candidate / 100 % 26;
        raw[2] = '0' + candidate / 10 % 10;
        raw[3] = '0' + candidate % 10;
        cudaCrypt(raw, encrypted);
        int equal = 1;
        for (int i = 0; i < 10; i++) if (encrypted[i] != input[index].text[i]) { equal = 0; break; }
        if (equal) {
            for (int i = 0; i < 5; i++) output[index].text[i] = raw[i];
            return;
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
    Encrypted *host_input = NULL, *device_input = NULL;
    Plain *host_output = NULL, *device_output = NULL;
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
            if (next < capacity || next > SIZE_MAX / sizeof(Encrypted)) { failed = 1; goto cleanup; }
            Encrypted *larger = (Encrypted *)realloc(host_input, next * sizeof(Encrypted));
            if (!larger) { fprintf(stderr, "Host input allocation failed.\n"); failed = 1; goto cleanup; }
            host_input = larger; capacity = next;
        }
        memcpy(host_input[count].text, line, 10); host_input[count++].text[10] = '\0';
    }
    if (ferror(input)) { perror("Reading passwords"); failed = 1; goto cleanup; }
    if (fclose(input)) { input = NULL; failed = 1; goto cleanup; } input = NULL;
    if (!count || count > SIZE_MAX / sizeof(Plain)) { fprintf(stderr, "Password file must be nonempty and fit in memory.\n"); failed = 1; goto cleanup; }
    host_output = (Plain *)malloc(count * sizeof(Plain));
    if (!host_output) { failed = 1; goto cleanup; }
    CUDA_TRY(cudaGetDeviceProperties(&properties, 0));
    threads = properties.maxThreadsPerBlock < 256 ? properties.maxThreadsPerBlock : 256;
    if ((count - 1) / threads + 1 > (size_t)properties.maxGridSize[0]) { fprintf(stderr, "Input exceeds x-grid limit.\n"); failed = 1; goto cleanup; }
    blocks = (unsigned int)((count - 1) / threads + 1);
    CUDA_TRY(cudaMalloc((void **)&device_input, count * sizeof(Encrypted)));
    CUDA_TRY(cudaMalloc((void **)&device_output, count * sizeof(Plain)));
    CUDA_TRY(cudaEventCreate(&start_event)); CUDA_TRY(cudaEventCreate(&finish_event));
    transfer_start = wall_time();
    CUDA_TRY(cudaMemcpy(device_input, host_input, count * sizeof(Encrypted), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaEventRecord(start_event));
    crackPasswords<<<blocks, threads>>>(device_input, device_output, count);
    CUDA_TRY(cudaGetLastError()); CUDA_TRY(cudaEventRecord(finish_event));
    CUDA_TRY(cudaEventSynchronize(finish_event));
    CUDA_TRY(cudaEventElapsedTime(&kernel_ms, start_event, finish_event));
    CUDA_TRY(cudaMemcpy(host_output, device_output, count * sizeof(Plain), cudaMemcpyDeviceToHost));
    device_seconds = wall_time() - transfer_start;
    /* Re-encryption on the host detects transfer/indexing errors before writing. */
    for (size_t i = 0; i < count; i++) {
        if (!host_output[i].text[0]) { not_found++; continue; }
        host_output[i].text[4] = '\0'; cudaCrypt(host_output[i].text, verified);
        if (strcmp(verified, host_input[i].text)) { fprintf(stderr, "Verification failed at line %zu.\n", i+1); failed = 1; goto cleanup; }
    }
    output_file = fopen("decrypted.txt", "w");
    if (!output_file) { perror("decrypted.txt"); failed = 1; goto cleanup; }
    for (size_t i = 0; i < count; i++)
        if (fprintf(output_file, "%s\n", host_output[i].text[0] ? host_output[i].text : "NOT_FOUND") < 0) { failed = 1; goto cleanup; }
    if (fclose(output_file)) { output_file = NULL; failed = 1; goto cleanup; } output_file = NULL;
    printf("GPU: %s | Passwords: %zu | Blocks: %u | Threads/block: %u\n", properties.name, count, blocks, threads);
    printf("Kernel: %.3f ms | Transfers + kernel: %.6f s | Recovered: %zu | Not found: %zu\n", kernel_ms, device_seconds, count-not_found, not_found);
    printf("Output: decrypted.txt | Host re-encryption verification passed\n");
cleanup:
    if (input && fclose(input)) failed = 1;
    if (output_file && fclose(output_file)) failed = 1;
    if (device_input && cudaFree(device_input) != cudaSuccess) failed = 1;
    if (device_output && cudaFree(device_output) != cudaSuccess) failed = 1;
    if (start_event && cudaEventDestroy(start_event) != cudaSuccess) failed = 1;
    if (finish_event && cudaEventDestroy(finish_event) != cudaSuccess) failed = 1;
    free(host_input); free(host_output);
    return failed ? 1 : not_found ? 2 : 0;
}
