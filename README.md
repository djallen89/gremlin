# gremlin
[![Build Status](https://travis-ci.org/djallen89/gremlin.svg?branch=master)](https://travis-ci.org/djallen89/gremlin)

Matrix multiplication algorithm using tiling and SIMD implemented in
Rust. There is only one current target, 64 bit x86 architectures which
have AVX2 instructions.

## Building
1. clone gremlin
```sh
git clone https://github.com/djallen89/gremlin.git
```
2. Install Rust, Cargo, and other Dependencies
  - **Ubuntu**
```sh
sudo apt install cargo gfortran libopenblas-dev
```
  - **Fedora**
  - **CentOS**
  - **Other Linux**
  - **OSX**
  - **Windows**

3. Building the release version
   You can either run build.sh, or use the following:
```sh
RUSTFLAGS='-C target-feature=+avx2,+fma -C target-cpu=native' cargo build --release
```
## Testing

Simply run
```sh
cargo test
```

## Running Benchmarks

There are four scripts provided to run the benchmarks: 
+ parallel2\_bench
+ parallel4\_bench.sh
+ parallel16\_bench.sh
+ parallel\_bench.sh

If your cpu has some other number of cores, you can call runjob.sh to
run your parallel benchmark for you, specifying the number of threads
as the first argument, and the benchmark as the second argument.
    

