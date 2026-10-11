# Viva preparation: explain the design in your own words

Use this alongside the source. Practise tracing a small input and explaining
why the program is correct before discussing speed.

## Task 1: Pthreads

**What is dynamic slicing here?** A mutex protects the next unclaimed byte
position. Each worker claims a slice, processes it, then claims another. The
nominal slice is calculated from input length and requested threads. Faster
workers can claim more work. It differs from assigning one fixed range per
thread, as in the week 1 prime-number example.

**What happens when a slice ends inside a word?** Its endpoint moves forward
until the end of that word. The next slice begins at that endpoint. This makes
slices nonoverlapping and prevents lost or double-counted boundary words.

**Why use a mutex?** Reading and advancing the shared queue position is a
read-modify-write operation. Without exclusion, two workers could claim the
same range. The mutex does not cover the entire counting loop.

**Why private frequency tables?** Updating one shared table for every word
would cause contention. Each worker owns its table. After pthread_join,
the main thread merges tables without concurrent access.

**What does pthread_join guarantee?** Main waits for completion before reading
the worker's results or freeing memory it uses. Joining also provides the
synchronisation required to observe those completed writes.

**Why might four threads be slower?** Serial file reading, thread creation,
allocation, hashing, merging, scheduling, and memory traffic add overhead.
Small inputs may not contain enough parallel work to cover those costs.

## Task 2: OpenMP

**Why malloc?** Dimensions come from the input file, so fixed-size matrices
would either waste memory or impose arbitrary limits. A flat array stores
element (row,col) at row*cols+col.

**What does collapse(2) do?** It distributes the combined row and column
iteration space. It does not make the inner dot-product sum shared.

**Where are races avoided?** Each iteration writes a distinct output cell.
The multiplication accumulator is declared inside that iteration, so it is
private. Input matrices are read-only while the computation runs.

**Why check operations independently?** Equal shapes permit element-wise
operations. A.cols==B.rows permits A x B. Both transposes are always valid.
One failed condition must not suppress unrelated valid operations.

**Why NaN?** The assignment explicitly requests NaN when a divisor is zero.
The program writes it at that cell and computes the remaining cells.

**How are threads capped?** The requested number is limited to the largest
relevant matrix dimension. num_threads requests that capped team. It is not
a guarantee of one physical CPU core per thread.

**Why validate physical rows?** A loose fscanf numeric scan can consume a
value from the next row and hide a malformed input. This parser reads complete
lines through fscanf, then checks exact comma-separated field counts.

## Task 3: CUDA passwords

**How many candidates are there?** 26*26*10*10=67,600. The integer candidate
index decomposes into two letter indices and two digit indices.

**What does one thread do?** It owns one encrypted input line and searches
candidates until it finds a match. Input length determines the number of
blocks, using ceiling division by the selected threads per block.

**What is the global index?** blockIdx.x*blockDim.x+threadIdx.x. threadIdx.x
alone repeats in each block and would process the same first lines repeatedly.

**Why the bounds check?** Ceiling division launches extra threads in the last
block when the input count is not a multiple of the block size.

**Why no static encryption buffer?** A shared buffer would be overwritten by
other threads. Each thread uses private raw/encrypted arrays and writes only
its own output record.

**Why preserve unusual boundary rules?** The supplied generator defines the
cipher. Replacing its arithmetic with standard modulo wrapping would produce
different ciphertext, even if the replacement looked more conventional.

**Can the algorithm be improved?** Candidate-level parallelism could reduce
the search time for a small number of passwords, but would need a different
ownership/selection scheme. The submitted design directly follows the brief's
line-count-based launch and keeps one writer per output record.

## Task 4: CUDA Sobel

**What are Gx and Gy?** Weighted differences in a 3x3 grayscale neighbourhood.
One measures changes across x, the other across y. The program combines them
using sqrt(Gx*Gx+Gy*Gy), not abs(Gx)+abs(Gy).

**What is zero padding?** A neighbour outside the image returns grayscale zero.
Every input pixel gets an output, so dimensions remain unchanged. A bright
constant image can therefore show an edge at its border.

**Why grayscale, rounding, and clamping?** The three RGB channels are averaged
using integer arithmetic. Gradient magnitude can exceed 255; it is rounded
then clamped for the 8-bit PNG channels. The original alpha is copied.

**Are images processed simultaneously?** Host code iterates over images.
Pixels within each image run in parallel on the GPU. This avoids holding every
image in GPU memory at once. Do not describe this as parallel image batching.

**How was correctness checked?** A separate NumPy reference uses zero padding
and the same documented grayscale/output conventions. Validation compares
every RGBA output pixel, including tiny and non-block-aligned images.

## Host/device lifecycle and timing

Explain this sequence without looking at the code: read/decode on host,
malloc host storage, cudaMalloc device storage, copy host to device, launch,
check launch and completion errors, copy device to host, write/encode,
cudaFree and free.

CUDA events measure the kernel interval. A host wall timer measures transfers
plus kernel. External process wall time includes process startup, context
initialisation, file I/O, and output generation. CPU compute timing also excludes
the file reading/writing stages. These measurements answer different questions.

Speedup is T1/Tn for comparable workloads and timing scopes. Efficiency is
speedup/n. A speedup below one means the parallel version was slower. Discuss
actual evidence rather than assuming that more threads always help.

## Short live exercises

1. Trace two workers claiming a slice containing `alpha beta`.
2. Calculate row-major indices for a 2x3 matrix and its 3x2 transpose.
3. Compute a 2x3 by 3x2 matrix product on paper.
4. Decompose candidate 67,599 into `zz99`.
5. Compute the global CUDA index for block 2, thread 7, block size 256.
6. Apply Sobel to a 1x1 image with zero padding and explain why Gx=Gy=0.
7. Explain which allocations must be freed after a failed CUDA copy.
8. Identify one measured limitation and one justified future improvement.
