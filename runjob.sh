#!/bin/sh
set -x
RAYON_NUM_THREADS="$1" \
                 RUSTFLAGS='-C target-feature=+avx2,+fma -C target-cpu=native' \
                 cargo bench --bench "$2"
