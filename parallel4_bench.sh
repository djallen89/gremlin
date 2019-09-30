#!/bin/sh
RAYON_NUM_THREADS=4 \
                 RUSTFLAGS='-C target-feature=+avx2,+fma -C target-cpu=native' \
                 cargo bench --bench parallel04
