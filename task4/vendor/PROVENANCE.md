# LodePNG dependency

Upstream: https://github.com/lvandeve/lodepng

Revision: e584a8490db1b76b81bf298c11c09d3b55d9c270

lodepng.cpp was copied unchanged as lodepng.c and compiled with gcc, using
upstream's C-compatible implementation. lodepng.h and LICENSE are unchanged.
The library is vendored so a fresh Colab run does not need unpinned downloads.
LodePNG handles PNG decoding/encoding; the Sobel algorithm is in ../sobel.cu.
