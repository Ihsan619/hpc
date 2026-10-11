CC = gcc
NVCC = nvcc
CFLAGS = -std=c11 -O2 -Wall -Wextra -Wpedantic
CUDA_ARCH ?= native
NVFLAGS = -O2 -lineinfo -arch=$(CUDA_ARCH) -Xcompiler=-Wall,-Wextra

.PHONY: cpu cuda all validate-cpu
all: cpu cuda
cpu: task1/word_count task2/matrix_operations
cuda: task3/password_cracking task4/sobel

task1/word_count: task1/word_count.c
	$(CC) $(CFLAGS) -pthread $< -o $@
task2/matrix_operations: task2/matrix_operations.c
	$(CC) $(CFLAGS) -fopenmp $< -lm -o $@
task3/password_cracking: task3/password_cracking.cu
	$(NVCC) $(NVFLAGS) $< -o $@
task4/vendor/lodepng.o: task4/vendor/lodepng.c task4/vendor/lodepng.h
	$(CC) $(CFLAGS) -c $< -o $@
task4/sobel: task4/sobel.cu task4/vendor/lodepng.o
	$(NVCC) $(NVFLAGS) -Itask4/vendor $^ -o $@
validate-cpu: cpu
	python3 validation/validate.py task1
	python3 validation/validate.py task2
