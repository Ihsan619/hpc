# High Performance Computing Portfolio

Module: 6CS005 High Performance Computing

This portfolio implements four workloads using Pthreads, OpenMP and CUDA. The CPU programs are written in C. The GPU programs are separate CUDA source files with C-style host memory management. Google Colab supplies the Linux compiler and GPU runtime; notebook Python cells compile, execute and check the assessed programs.

## Reproducibility and public resources

Repository: https://github.com/Ihsan619/hpc

The notebooks fetch a fixed public Git commit and verify SHA-256 hashes for ten input files. They require neither a personal Google Drive mount nor a private dataset. An Internet connection and an available Colab GPU are necessary for Tasks 3 and 4. GPU availability is controlled by Colab and can delay execution.

Runnable notebooks:

- Task 1: https://colab.research.google.com/github/Ihsan619/hpc/blob/main/notebooks/Task_01.ipynb
- Task 2: https://colab.research.google.com/github/Ihsan619/hpc/blob/main/notebooks/Task_02.ipynb
- Task 3: https://colab.research.google.com/github/Ihsan619/hpc/blob/main/notebooks/Task_03.ipynb
- Task 4: https://colab.research.google.com/github/Ihsan619/hpc/blob/main/notebooks/Task_04.ipynb

## Task 1 Word occurrence counting

The program accepts an input filename and a positive thread count. A mutex protects the next unclaimed byte position. Each worker claims a slice and extends its end to a word boundary before releasing the mutex. This prevents a word crossing a nominal boundary from being split or counted twice. The counting work occurs outside the critical section.

Each worker owns a hash table. Workers therefore do not contend on every word increment. After pthread_join completes, the main thread combines these tables, sorts the unique words and writes result.txt. Tokens are maximal ASCII alphanumeric runs and letters are converted to lowercase; this rule makes punctuation and case handling explicit.

The supplied dataset produced 120,000 tokens and 94 distinct words. Independent Counter checks compared every frequency across different thread counts. Additional inputs exercised empty files, punctuation, missing final newlines and an 8,193-character word crossing a slice boundary. Invalid thread arguments and missing files were also checked.

Dynamic slicing balances work when workers finish at different times, but the mutex, table allocation, merge and file operations remain overheads. The final recorded benchmark used 1.2 million tokens. Three process-time trials followed one excluded warm-up per thread configuration. Median times were 0.138758, 0.130090, 0.132327 and 0.134460 seconds for 1, 2, 4 and 8 threads respectively. The best measured improvement was modest, approximately 1.07 times at two threads. More threads did not improve this workload on the two logical CPUs exposed by Colab.

## Task 2 Matrix operations

The program reads dynamically allocated matrices from the comma-separated file format and processes successive pairs. It validates headers, physical row lengths, numeric tokens, matrix counts and finite input values before opening results.txt. This avoids publishing a partial result file for malformed input.

Addition, subtraction and element-wise multiplication or division require equal shapes. A zero divisor produces NaN in that output position. Both transposes are always calculated. Matrix multiplication is attempted independently when the first matrix column count equals the second matrix row count. Incompatible operations receive an explanatory output message while valid operations continue.

OpenMP parallel for loops distribute output cells. Each output cell has one writer. For matrix multiplication, the accumulation variable is private to its iteration, so no reduction or mutex is needed. The requested team size is capped by the largest relevant dimension of each operation. Actual runtime scheduling remains the responsibility of OpenMP.

All 25 supplied matrix pairs were processed. An independent list-based reference checked every output value and shape using a floating-point tolerance, including compatible rectangular multiplication, incompatible element-wise shapes and zero divisors. Malformed rows, headers, unmatched final matrices and invalid command-line arguments were rejected.

The benchmark used a reproducible 384 by 384 pair. The first validation session gave medians of 1.239362, 0.627681, 0.585129 and 0.583776 seconds at 1, 2, 4 and 8 threads. The recorded repeat gave 0.580061, 0.998280, 0.586811 and 0.598136 seconds. This large variation means the first session alone does not establish a reliable speedup. Colab host load and scheduling affect process time, and the repeat shows that extra requested threads are not consistently beneficial. These process measurements include input parsing and output formatting in addition to computation.

## Task 3 CUDA password cracking

The input contains one ten-character encrypted password per line. The device helper preserves the supplied cudaCrypt arithmetic and boundary conditions. Its output buffer belongs to the calling thread, avoiding the shared static storage used in the original teaching example.

Each GPU thread handles one encrypted input and searches the 67,600 possible plaintexts consisting of two lowercase letters and two digits. The index is blockIdx.x multiplied by blockDim.x plus threadIdx.x. The block count is the ceiling of the input count divided by 256; a bounds check handles the last partial block. Output positions correspond directly to input positions, so no atomics or ordering pass is required.

The host checks allocations, memory copies, kernel launch and completion. Recovered passwords are copied back and re-encrypted before decrypted.txt is written. Unmappable inputs are reported explicitly. Memory and CUDA events are released on normal and error paths.

The seeded dataset contains 10,000 entries with known plaintexts and deliberate boundary examples. Every recovered entry matched its expected plaintext and passed host re-encryption. Invalid, empty and unmappable inputs were tested separately. The recorded T4 run used 40 blocks of 256 threads and recovered all 10,000 entries. Its kernel took 14.393 milliseconds; transfers plus kernel took 0.014548 seconds. The recorded repeated process-time median was 0.383559 seconds, illustrating how CUDA process startup and file handling exceed kernel time. No CPU-to-GPU speedup is claimed because a comparable serial C baseline was not measured.

## Task 4 CUDA Sobel filtering

LodePNG decodes each input to RGBA. A GPU thread computes one output pixel using the horizontal and vertical 3 by 3 Sobel masks. Neighbours outside the image are zero. Grayscale is the integer average of the three colour channels. The magnitude is sqrt(Gx squared plus Gy squared), rounded to the nearest integer and clamped to 255. The original alpha channel is preserved. These choices are explicit so the reference calculation and visual result are reproducible.

Images are processed sequentially on the host while their pixels are processed in parallel on the GPU. Each image receives a dynamically sized allocation and launch. Output names are sobel_ followed by the original basename. Duplicate basenames are rejected to avoid overwriting another image's output. Decoded buffers, output buffers, device allocations and CUDA events are released after each image.

The test set includes the lecturer's 4 by 4 and 6 by 4 images, a procedural 512 by 384 building image, a 17 by 19 image exercising a partial block, a 1 by 7 image and a 1 by 1 image. Independent NumPy calculations compared every RGBA pixel, including all borders. All six images passed, and missing or corrupt input cases were checked.

An initial T4 verification of the larger image used 768 blocks of 256 threads. In three separate-process trials the kernel took 0.074, 0.076 and 0.063 milliseconds; process times were 0.284711, 0.285426 and 0.273414 seconds. For this image, context creation, PNG conversion and transfers dominate overall execution. A faster kernel alone therefore would not produce a similar improvement in total time. Shared-memory tiling is a possible extension, but the submitted implementation follows the lecturer's global-memory and linear-indexing approach.

## Validation and evidence

The reference scripts are separate from the assessed C and CUDA implementations. They compare full outputs instead of checking only whether a program exits successfully. Tests cover boundary conditions, input validation, output dimensions and thread-count repeatability. Screen recordings capture genuine Colab execution on the student's PC and are stored under evidence. Compiler and hardware information are printed by each notebook and included in its downloadable evidence bundle.

CUDA validation used a Tesla T4, CUDA compiler 13.0.88 and GCC 13.3.0. CPU notebooks exposed two logical CPUs. Build flags use C11, optimisation and compiler warnings; CUDA uses native GPU architecture selection. Reported measurements belong to these sessions and should not be treated as hardware-independent performance guarantees.

## Learning and limitations

The four tasks use different ownership models: private worker dictionaries with a protected work queue, independent OpenMP output iterations, one password per CUDA thread, and one Sobel output pixel per CUDA thread. In each case, deciding who writes each result removes unnecessary synchronisation. Correct boundary handling is as important as parallel execution.

The evidence supports functional correctness for the tested inputs. It does not establish universal scalability. The CPU measurements vary on a shared cloud host; GPU process overhead dominates the demonstrated small workloads. Larger workloads, controlled hardware and repeated batches would be required for stronger performance conclusions.

## Sources and acknowledgements

- University of Wolverhampton, 6CS005 assessment brief for 2025 to 2026, supplied module document.
- Supplied weeks 1 and 2 Pthreads lectures and examples; weeks 3 and 4 OpenMP and matrix material; weeks 6 to 9 CUDA, memory, indexing and PNG examples.
- Supplied CryptForCuda.c and PasswordGeneratorToText.c, used to preserve and validate the teaching encryption transformation.
- LodePNG by Lode Vandevenne, public source https://github.com/lvandeve/lodepng, pinned revision e584a8490db1b76b81bf298c11c09d3b55d9c270. Its licence is included with the source.

Before submission, add the student's required cover-page details, apply the supplied report template and check the institution's required declaration of assistance.
