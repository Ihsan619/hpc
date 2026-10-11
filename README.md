# 6CS005 High Performance Computing portfolio

Four standalone implementations based on the supplied module materials:

| Task | Source | Technology | Output |
|---|---|---|---|
| Word occurrence counting | task1/word_count.c | C, POSIX threads | result.txt |
| Matrix operations | task2/matrix_operations.c | C, OpenMP | results.txt |
| Password cracking | task3/password_cracking.cu | CUDA | decrypted.txt |
| Sobel edge detection | task4/sobel.cu | CUDA, LodePNG | sobel_<input>.png, gradient_x_<input>.png, gradient_y_<input>.png |

The brief provides example output names and does not prescribe source filenames.
The supplied datasets keep their original names. Each task has a separate Colab
notebook, with public GitHub downloads and no personal Drive mount. CUDA source
files remain separate deliverables. Notebook links will be added when published.

## Build and run in Linux / Colab

```sh
make cpu
cd task1
./word_count data/WordOccurrenceDataset.txt 4
cd ../task2
./matrix_operations data/MatData.txt 4
cd ..
make cuda
cd task3
./password_cracking data/passwords.txt
cd ../task4
./sobel edges data/4x4.png data/6x4.png data/bird.png
```

CUDA needs an NVIDIA GPU and nvcc. The default architecture is the GPU visible
to nvcc (`-arch=native`); override with `make CUDA_ARCH=sm_75 cuda` if necessary.
All implementations use C-style allocation and file I/O. nvcc compiles CUDA
host code through its C++ frontend; the code does not use STL or C++ classes.

## Validation

```sh
python3 validation/validate.py task1 --benchmark
python3 validation/validate.py task2 --benchmark
python3 validation/validate.py task3 --benchmark
python3 validation/validate.py task4 --benchmark
```

Python is used only for notebook orchestration, independent test references,
fixture generation, and displaying results. The assessed algorithms run in
the C/CUDA executables. Pillow and NumPy are needed for PNG validation. CPU
validation also runs in GitHub Actions; CUDA validation requires a GPU runtime.
Measured results belong in evidence/ after real execution.

## Decisions to explain in the viva

- Task 1 counts ASCII alphanumeric runs case-insensitively. Punctuation,
  apostrophes, and hyphens separate words. Mutex protection covers the shared
  slice queue; frequency tables remain private until threads are joined.
- Task 2 uses row-major malloc arrays and independent output cells. Each
  operation gets its own dimension check and capped OpenMP team request.
  A missing partner matrix is rejected. Zero divisors produce NaN.
- Task 3 keeps the lecturer's encryption arithmetic exactly, searches all
  67,600 possible candidates per encrypted line, and re-encrypts recovered
  plaintext on the CPU before writing. This educational cipher is not a
  real password-storage scheme. No static device output buffer is shared.
- Task 4 uses integer mean RGB grayscale, zero padding, Euclidean Sobel
  magnitude rounded and clamped to 255, and original alpha. Images are handled
  sequentially on the host with pixels processed in parallel on the GPU.

## Input provenance

WordOccurrenceDataset.txt and MatData.txt were supplied with the assignment.
The 4x4.png and 6x4.png inputs were supplied in week 9. Lecturer password
examples are preserved in task3/lecturer. Additional inputs are reproducible
seed-619 fixtures. SHA-256 values are listed in docs/input_manifest.json.
LodePNG's exact upstream revision and license are preserved in task4/vendor.

## Submission

Package each task separately with its source and resources, as required by
the brief. Include Colab links in the report, separate .cu files, actual
outputs, and four real screen recordings in evidence/. Review all source
and design decisions before the viva and follow the university's rules for
acknowledging assistance and external code.

Task 4 uses the supplied bird photograph. Each CUDA thread writes the combined magnitude and clipped absolute X/Y gradients into three independent RGBA planes. The notebook displays the original and all three outputs in a 2 by 2 comparison.
