# gremlin
[![Build Status](https://travis-ci.org/djallen89/gremlin.svg?branch=master)](https://travis-ci.org/djallen89/gremlin)

Matrix multiplication algorithm using tiling and SIMD implemented in Rust. 
%RAYON_NUM_THREADS=8 RUSTFLAGS='-C target-feature=+avx,+fma' cargo bench --bench parallel
