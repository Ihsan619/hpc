/* 6CS005 Task 1: dynamic byte slices, private sorted arrays, and a joined merge.
 * Usage: ./word_count WordOccurrenceDataset.txt 4
 * Words are maximal ASCII alphanumeric runs, compared case-insensitively.
 */
#define _POSIX_C_SOURCE 200809L
#include <ctype.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { char *text; size_t count; } Word;
typedef struct { Word *items; size_t used, capacity, tokens; } Counts;
typedef struct {
    const unsigned char *data;
    size_t length, next, slice;
    pthread_mutex_t lock;
} Work;
typedef struct { Work *work; Counts counts; size_t slices; int failed; } Worker;

static int word_char(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9');
}
static int alphabetical(const void *a, const void *b) {
    return strcmp(((const Word *)a)->text, ((const Word *)b)->text);
}
static int append_word(Counts *counts, char *word, size_t number) {
    if (counts->used == counts->capacity) {
        size_t next = counts->capacity ? counts->capacity * 2 : 256;
        if (next < counts->capacity || next > SIZE_MAX / sizeof(Word)) return 0;
        Word *larger = realloc(counts->items, next * sizeof(Word));
        if (!larger) return 0;
        counts->items = larger; counts->capacity = next;
    }
    counts->items[counts->used++] = (Word){word, number};
    counts->tokens += number;
    return 1;
}
static void combine_duplicates(Counts *counts) {
    if (!counts->used) return;
    qsort(counts->items, counts->used, sizeof(Word), alphabetical);
    size_t unique = 0;
    for (size_t i = 0; i < counts->used; i++) {
        if (unique && !strcmp(counts->items[unique-1].text, counts->items[i].text)) {
            counts->items[unique-1].count += counts->items[i].count;
            free(counts->items[i].text);
        } else counts->items[unique++] = counts->items[i];
    }
    counts->used = unique;
}
static void release_counts(Counts *counts) {
    for (size_t i = 0; i < counts->used; i++) free(counts->items[i].text);
    free(counts->items);
}
static void *count_slice(void *argument) {
    Worker *worker = argument;
    Work *work = worker->work;
    for (;;) {
        /* Only the queue position is shared. Counting needs no shared lock. */
        pthread_mutex_lock(&work->lock);
        size_t begin = work->next, end = begin;
        if (begin < work->length) {
            end += work->slice < work->length - begin ? work->slice : work->length - begin;
            /* Extend the slice to finish a word; the next worker starts at end. */
            while (end < work->length && word_char(work->data[end])) end++;
            work->next = end;
        }
        pthread_mutex_unlock(&work->lock);
        if (begin == work->length) break;
        worker->slices++;
        size_t i = begin;
        while (i < end) {
            if (!word_char(work->data[i])) { i++; continue; }
            size_t start = i;
            while (i < end && word_char(work->data[i])) i++;
            size_t length = i - start;
            char *word = malloc(length + 1);
            if (!word) { worker->failed = 1; return NULL; }
            for (size_t j = 0; j < length; j++) word[j] = (char)tolower(work->data[start+j]);
            word[length] = '\0';
            if (!append_word(&worker->counts, word, 1)) {
                free(word); worker->failed = 1; return NULL;
            }
        }
    }
    combine_duplicates(&worker->counts);
    return NULL;
}
static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}
int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "Usage: %s input.txt threads\n", argv[0]); return 1; }
    char *tail; errno = 0; long requested = strtol(argv[2], &tail, 10);
    if (errno || *tail || tail == argv[2] || requested < 1 || requested > 1024) {
        fprintf(stderr, "Thread count must be an integer from 1 to 1024.\n"); return 1;
    }
    FILE *input = fopen(argv[1], "rb");
    if (!input) { perror(argv[1]); return 1; }
    size_t capacity = 8192, length = 0;
    unsigned char *data = malloc(capacity);
    if (!data) { fclose(input); fprintf(stderr, "Input allocation failed.\n"); return 1; }
    int c, failed = 0;
    while ((c = fgetc(input)) != EOF) {
        if (length == capacity) {
            if (capacity > SIZE_MAX / 2) { failed = 1; break; }
            unsigned char *larger = realloc(data, capacity * 2);
            if (!larger) { failed = 1; break; }
            data = larger; capacity *= 2;
        }
        data[length++] = (unsigned char)c;
    }
    if (ferror(input)) failed = 1;
    if (fclose(input)) failed = 1;
    if (failed) { free(data); fprintf(stderr, "Failed reading input.\n"); return 1; }
    size_t thread_count = (size_t)requested;
    Work work = { .data = data, .length = length, .next = 0 };
    /* More slices than workers allow faster workers to claim additional work. */
    work.slice = length / (thread_count * 8);
    if (work.slice < 1) work.slice = 1;
    Worker *workers = calloc(thread_count, sizeof(*workers));
    pthread_t *threads = malloc(thread_count * sizeof(*threads));
    if (!workers || !threads) { free(workers); free(threads); free(data); return 1; }
    int error = pthread_mutex_init(&work.lock, NULL);
    if (error) { fprintf(stderr, "Mutex: %s\n", strerror(error)); free(workers); free(threads); free(data); return 1; }
    size_t created = 0;
    double start = now();
    for (; created < thread_count; created++) {
        workers[created].work = &work;
        error = pthread_create(&threads[created], NULL, count_slice, &workers[created]);
        if (error) { fprintf(stderr, "Thread creation: %s\n", strerror(error)); failed = 1; break; }
    }
    for (size_t i = 0; i < created; i++) {
        error = pthread_join(threads[i], NULL);
        if (error) { fprintf(stderr, "Thread join: %s\n", strerror(error)); exit(1); }
        if (workers[i].failed) failed = 1;
    }
    Counts merged = {0};
    if (!failed) for (size_t i = 0; i < created && !failed; i++) {
        for (size_t j = 0; j < workers[i].counts.used; j++) {
            Word *word = &workers[i].counts.items[j];
            char *copy = strdup(word->text);
            if (!copy || !append_word(&merged, copy, word->count)) {
                free(copy); failed = 1; break;
            }
        }
    }
    if (!failed) combine_duplicates(&merged);
    double elapsed = now() - start;
    if (!failed) {
        FILE *output = fopen("result.txt", "w");
        if (!output) { perror("result.txt"); failed = 1; }
        else {
            for (size_t i = 0; i < merged.used; i++)
                if (fprintf(output, "%s %zu\n", merged.items[i].text, merged.items[i].count) < 0) failed = 1;
            if (fclose(output)) failed = 1;
        }
    }
    if (!failed) {
        printf("Tokens: %zu | Unique: %zu | Threads: %zu | Count and merge: %.6f s\n", merged.tokens, merged.used, created, elapsed);
        for (size_t i = 0; i < created; i++) printf("Worker %zu: %zu slices, %zu tokens\n", i, workers[i].slices, workers[i].counts.tokens);
        printf("Output: result.txt (alphabetical)\n");
    } else fprintf(stderr, "Word counting failed; output is not a verified result.\n");
    release_counts(&merged);
    for (size_t i = 0; i < thread_count; i++) release_counts(&workers[i].counts);
    pthread_mutex_destroy(&work.lock); free(workers); free(threads); free(data);
    return failed ? 1 : 0;
}
