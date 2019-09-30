#!/bin/sh
RUSTFLAGS='-C target-feature=+avx2,+fma -C target-cpu=native' cargo build --release
