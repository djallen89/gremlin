#!/bin/sh
RAYON_NUM_THREADS=16 \
                 RUSTFLAGS='-C target-feature=+avx2,+fma -C target-cpu=native' \
                 cargo bench --bench parallel
