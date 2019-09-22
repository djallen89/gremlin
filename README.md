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

3. Building release version
   ```sh
   RUSTFLAGS='-C target-feature=+avx2' cargo build --release
   ```

## Running
    To be written.
    
## Documentation
    ```sh
    cargo doc
    ```
