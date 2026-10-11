/* 6CS005 Task 2. Usage: ./matrix_operations MatData.txt 4 */
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int rows, cols; double **value; } Matrix;
static size_t line_number;

/* Read a complete physical line with the required standard fscanf file API.
 * This avoids silently accepting a missing row by scanning into the next one. */
static int read_line(FILE *file, char **line) {
    size_t used = 0, capacity = 128;
    char *buffer = malloc(capacity), c;
    if (!buffer) return -1;
    int got;
    while ((got = fscanf(file, "%c", &c)) == 1 && c != '\n') {
        if (c == '\0') { free(buffer); return -1; }
        if (used + 1 == capacity) {
            if (capacity > SIZE_MAX / 2) { free(buffer); return -1; }
            char *larger = realloc(buffer, capacity * 2);
            if (!larger) { free(buffer); return -1; }
            buffer = larger; capacity *= 2;
        }
        buffer[used++] = c;
    }
    if (ferror(file)) { free(buffer); return -1; }
    if (got != 1 && !used) { free(buffer); return 0; }
    buffer[used] = '\0'; *line = buffer; line_number++;
    return 1;
}
static char *spaces(char *p) { while (isspace((unsigned char)*p)) p++; return p; }
static int dimension(char **cursor, int *result) {
    char *p = spaces(*cursor), *end;
    errno = 0; long n = strtol(p, &end, 10);
    if (errno || p == end || n < 1 || n > INT_MAX) return 0;
    *result = (int)n; *cursor = spaces(end); return 1;
}
static void release_matrix(Matrix *matrix) {
    if (matrix->value) {
        for (int row = 0; row < matrix->rows; row++) free(matrix->value[row]);
        free(matrix->value);
    }
    matrix->value = NULL;
}
static int allocate(Matrix *matrix, int rows, int cols) {
    if (rows < 1 || cols < 1 || (size_t)rows > SIZE_MAX / sizeof(double *) ||
        (size_t)cols > SIZE_MAX / sizeof(double) ||
        (size_t)rows > SIZE_MAX / (size_t)cols / sizeof(double)) return 0;
    matrix->rows = rows; matrix->cols = cols;
    matrix->value = malloc((size_t)rows * sizeof(double *));
    if (!matrix->value) return 0;
    for (int row = 0; row < rows; row++) matrix->value[row] = NULL;
    for (int row = 0; row < rows; row++) {
        matrix->value[row] = malloc((size_t)cols * sizeof(double));
        if (!matrix->value[row]) { release_matrix(matrix); return 0; }
    }
    return 1;
}
static int read_matrix(FILE *file, Matrix *matrix) {
    char *line = NULL, *p;
    int status, rows, cols;
    do {
        status = read_line(file, &line);
        if (status != 1) return status;
        if (!*spaces(line)) { free(line); line = NULL; }
    } while (!line);
    p = line;
    int valid = dimension(&p, &rows);
    if (valid && *p == ',') { p++; valid = dimension(&p, &cols) && !*p; }
    else valid = 0;
    free(line);
    if (!valid) { fprintf(stderr, "Line %zu: expected positive rows,cols header.\n", line_number); return -1; }
    if (!allocate(matrix, rows, cols)) { fprintf(stderr, "Matrix allocation failed.\n"); return -1; }
    for (int row = 0; row < rows; row++) {
        line = NULL;
        if (read_line(file, &line) != 1) {
            fprintf(stderr, "Matrix needs %d rows; row %d is missing.\n", rows, row+1); goto invalid;
        }
        p = spaces(line);
        for (int col = 0; col < cols; col++) {
            char *end; errno = 0; double number = strtod(p, &end);
            if (p == end || errno || !isfinite(number)) {
                fprintf(stderr, "Line %zu, column %d: invalid finite number.\n", line_number, col+1); goto invalid;
            }
            matrix->value[row][col] = number;
            p = spaces(end);
            if (col + 1 < cols) {
                if (*p != ',') { fprintf(stderr, "Line %zu: expected %d comma-separated values.\n", line_number, cols); goto invalid; }
                p = spaces(p+1);
            } else if (*p) { fprintf(stderr, "Line %zu: extra values or trailing characters.\n", line_number); goto invalid; }
        }
        free(line); line = NULL;
    }
    return 1;
invalid:
    free(line); release_matrix(matrix); return -1;
}
static int write_matrix(FILE *file, const char *name, const Matrix *matrix) {
    if (fprintf(file, "%s - %d,%d\n", name, matrix->rows, matrix->cols) < 0) return 0;
    for (int row = 0; row < matrix->rows; row++) {
        for (int col = 0; col < matrix->cols; col++) {
            double x = matrix->value[row][col];
            if (col && fprintf(file, ",") < 0) return 0;
            if (isnan(x)) { if (fprintf(file, "NaN") < 0) return 0; }
            else if (fprintf(file, "%.17g", x) < 0) return 0;
        }
        if (fprintf(file, "\n") < 0) return 0;
    }
    return fprintf(file, "\n") >= 0;
}
static int operation(FILE *output, const Matrix *a, const Matrix *b, int kind, int requested, double *compute) {
    const char *names[] = {"Addition", "Subtraction", "Element-wise multiplication", "Element-wise division", "Transpose A", "Transpose B", "Matrix multiplication"};
    if (kind < 4 && (a->rows != b->rows || a->cols != b->cols))
        return fprintf(output, "%s cannot be done (shapes differ).\n", names[kind]) >= 0;
    if (kind == 6 && a->cols != b->rows)
        return fprintf(output, "%s cannot be done (A.cols != B.rows).\n", names[kind]) >= 0;
    const Matrix *source = kind == 5 ? b : a;
    int rows = kind == 4 || kind == 5 ? source->cols : a->rows;
    int cols = kind == 4 || kind == 5 ? source->rows : kind == 6 ? b->cols : a->cols;
    Matrix result = {0};
    if (!allocate(&result, rows, cols)) { fprintf(stderr, "Result allocation failed.\n"); return 0; }
    int threads = requested < rows ? requested : rows;
    int actual = 1;
    double start = omp_get_wtime();
    #pragma omp parallel num_threads(threads) default(none) shared(a,b,source,result,rows,cols,kind,actual)
    {
        #pragma omp single
        actual = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                double value = 0.0;
                if (kind == 4 || kind == 5) value = source->value[col][row];
                else if (kind == 6) {
                    for (int k = 0; k < a->cols; k++)
                        value += a->value[row][k] * b->value[k][col];
                } else {
                    double av = a->value[row][col], bv = b->value[row][col];
                    if (kind == 0) value = av + bv;
                    if (kind == 1) value = av - bv;
                    if (kind == 2) value = av * bv;
                    if (kind == 3) value = bv == 0.0 ? NAN : av / bv;
                }
                result.value[row][col] = value;
            }
        }
    }
    double elapsed = omp_get_wtime() - start; *compute += elapsed;
    printf("  %s: %d x %d | requested %d capped %d actual %d | %.6f s\n",
           names[kind], rows, cols, requested, threads, actual, elapsed);
    int ok = write_matrix(output, names[kind], &result);
    release_matrix(&result); return ok;
}
int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "Usage: %s input.txt threads\n", argv[0]); return 1; }
    char *end; errno = 0; long n = strtol(argv[2], &end, 10);
    if (errno || end == argv[2] || *end || n < 1 || n > 1024) { fprintf(stderr, "Threads must be 1..1024.\n"); return 1; }
    FILE *input = fopen(argv[1], "r");
    if (!input) { perror(argv[1]); return 1; }
    Matrix *matrices = NULL; size_t count = 0; int failed = 0, status;
    for (;;) {
        Matrix matrix = {0}; status = read_matrix(input, &matrix);
        if (status == 0) break;
        if (status < 0) { failed = 1; break; }
        if (count == SIZE_MAX / sizeof(Matrix)) { release_matrix(&matrix); failed = 1; break; }
        Matrix *larger = realloc(matrices, (count+1) * sizeof(Matrix));
        if (!larger) { release_matrix(&matrix); failed = 1; break; }
        matrices = larger; matrices[count++] = matrix;
    }
    if (fclose(input)) failed = 1;
    if (!count || count % 2) { fprintf(stderr, "Expected a nonempty even number of matrices; received %zu.\n", count); failed = 1; }
    FILE *output = NULL; double compute = 0;
    omp_set_dynamic(0);
    if (!failed) { output = fopen("results.txt", "w"); if (!output) { perror("results.txt"); failed = 1; } }
    if (!failed) for (size_t i = 0; i < count; i += 2) {
        if (fprintf(output, "Pair %zu | A=%d,%d | B=%d,%d\n\n", i/2+1, matrices[i].rows, matrices[i].cols, matrices[i+1].rows, matrices[i+1].cols) < 0) { failed = 1; break; }
        printf("Pair %zu\n", i/2+1);
        for (int kind = 0; kind < 7; kind++) if (!operation(output, &matrices[i], &matrices[i+1], kind, (int)n, &compute)) { failed = 1; break; }
        if (failed) break;
    }
    if (output && fclose(output)) failed = 1;
    for (size_t i = 0; i < count; i++) release_matrix(&matrices[i]);
    free(matrices);
    if (!failed) printf("Pairs: %zu | Compute: %.6f s | Output: results.txt\n", count/2, compute);
    else fprintf(stderr, "Matrix processing failed; no verified output.\n");
    return failed ? 1 : 0;
}
