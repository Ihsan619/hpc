# Alignment with the supplied senior notebooks

The four notebooks were read as examples of approach. The assessed code retains our input validation, public resources and filenames, and adapts the following methods.

| Task | Previous implementation | Adapted implementation |
|---|---|---|
| 1 | Private hash tables | Per-thread dynamically grown word arrays; alphabetical qsort and duplicate compression, followed by joined merge |
| 2 | Flat matrix allocation and collapsed cell loops | Row-wise malloc with double **; explicit OpenMP parallel region and static row distribution, with actual team measurement |
| 3 | One integer candidate loop and struct buffers | Four nested letter/digit loops with six-character prefix rejection; flat 11-byte cipher and 5-byte plaintext buffers plus per-input found status |
| 4 | Mean RGB, nearest-rounded magnitude, three RGBA planes | Integer weighted grayscale (30R+59G+11B)/100, truncated magnitude, separate RGBA edge and one-byte X/Y buffers; serial C reference check |

Pthreads dynamic slicing and mutex protection remain. Every matrix operation is checked independently. CUDA uses the lecturer's x-axis linear thread index with bounds guards and explicit allocation/copies. Task 3 uses at least two x blocks even for a small file. Task 4 keeps bird.png, zero padding and the four-panel display. The combined output preserves source alpha; the X/Y images are opaque 8-bit grayscale, matching the example method.

The assessment brief permits chosen C/CUDA source names and names result.txt, results.txt and decrypted.txt as output conventions. Existing source names and task ZIP names are retained so the submission links remain consistent.
