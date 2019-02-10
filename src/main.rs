#![feature(stdsimd, repr_simd, simd_ffi)]
#![feature(core_intrinsics, align_offset)]
extern crate libc;

use core::arch::x86_64::__m256d;
use core::arch::x86_64::_mm256_broadcast_sd;
use core::arch::x86_64::{_mm256_load_pd,  _mm256_store_pd};
use core::arch::x86_64::{_mm256_loadu_pd,  _mm256_storeu_pd};
use core::arch::x86_64::_mm256_fmadd_pd;

/*
pub trait FMADD {
    fn fmadd(&mut self, a: Self, b: Self);
}

impl FMADD for f32 {
    fn fmadd(&mut self, a: f32, b: f32) {
        *self = a.mul_add(b, *self);
    }
}

impl FMADD for f64 {
    fn fmadd(&mut self, a: f64, b: f64) {
        *self = a.mul_add(b, *self);
    }
}

fn madd<T: FMADD>(a: T, b: T, c: &mut T) {
    c.fmadd(a, b);
}
*/

#[target_feature(enable = "avx2")]
#[cfg(any(target_arch = "x86_64"))]
unsafe fn scalar_vec_fmadd_f64u(a_elt: &f64, b_row: &[f64], c_row: &mut [f64]) {
    let a: __m256d = _mm256_broadcast_sd(a_elt);
    let b: __m256d = _mm256_loadu_pd(&b_row[0] as *const f64);
    let mut c: __m256d = _mm256_loadu_pd(&c_row[0] as *const f64);
    c = _mm256_fmadd_pd(a, b, c);
    _mm256_storeu_pd(&mut c_row[0] as *mut f64, c);
}

#[target_feature(enable = "avx2")]
#[cfg(any(target_arch = "x86_64"))]
unsafe fn scalar_vec_fmadd_f64a(a_elt: &f64, b_row: &[f64], c_row: &mut [f64]) {
    let a: __m256d = _mm256_broadcast_sd(a_elt);
    let b: __m256d = _mm256_load_pd(&b_row[0] as *const f64);
    let mut c: __m256d = _mm256_load_pd(&c_row[0] as *const f64);
    c = _mm256_fmadd_pd(a, b, c);
    _mm256_store_pd(&mut c_row[0] as *mut f64, c);
}

fn minimatrix_fmadd64(a_arr: &[f64], b_arr: &[f64], c_arr: &mut [f64]) {
    /* For 3x3 matrices, AB + C = C can be represented as:
     * | A11 A12 A13 | | B11 B12 B13 | |C11 C12 C13| |C11 C12 C13| 
     * | A21 A22 A23 |*| B21 B22 B23 |+|C21 C22 C23|=|C21 C22 C23|
     * | A31 A32 A33 | | B31 B32 B33 | |C31 C32 C33| |C31 C32 C33| 
     *
     * A11B11+A12B21+A13B31+C11, A11B12+A12B22+A13B32+C12, A11B13+A12B23+A13B33+C13
     * ...
     * 
     * A11B11 + C11 = C11, A11B12 + C12 = C12, A11B13 + C13 = C13
     * A12B21 + C11 = C11, A12B22 + C12 = C12, A12B23 + C13 = C13
     * A13B31 + C11 = C11, A13B32 + C12 = C12, A13B33 + C13 = C13
     * ... and so on for the following rows.  Generalizing this, each
     * row of C can be calculated iterating over the columns of A,
     * calculating the product of A[i][j]*B[i]+C[i].
     * 
     */
    unsafe {
        let bptr_ofst = (&b_arr[0] as *const f64).align_offset(32);
        let cptr_ofst = (&c_arr[0] as *const f64).align_offset(32);
        if bptr_ofst == 0 && cptr_ofst == 0 {
            println!("Using aligned load");
            for row in 0 .. 4 {
                let ridx = row * 4;
                let mut c_row = &mut c_arr[ridx .. ridx + 4];
                for col in 0 .. 4 {
                    let bidx = col * 4;
                    let b_row = &b_arr[bidx .. bidx + 4];

                    let a_elt = &a_arr[ridx + col];
                    scalar_vec_fmadd_f64a(a_elt, b_row, &mut c_row);
                }
            }
        } else {
            println!("Using unaligned load");
            for row in 0 .. 4 {
                let ridx = row * 4;
                let mut c_row = &mut c_arr[ridx .. ridx + 4];
                for col in 0 .. 4 {
                    let bidx = col * 4;
                    let b_row = &b_arr[bidx .. bidx + 4];

                    let a_elt = &a_arr[ridx + col];
                    scalar_vec_fmadd_f64u(a_elt, b_row, &mut c_row);
                }
            }
        }
    } 
}

fn main() {
    let a_arr = vec!(1.0, 2.0, 3.0, 4.0,
                     7.0, 6.0, 5.0, 4.0,
                     0.5, 1.0, 2.0, 4.0,
                     8.0, 2.0, 0.5, 0.125);

    let align1 = vec!(0.0, 0.0, 0.0, 0.0);

    let b_arr = vec!(12.0,  13.0, 15.0, 17.0,
                     01.0,   2.0,  3.0,  4.0,
                     42.0, 404.0, 13.0,  9.0,
                     01.0,   2.0,  3.0,  4.0);

    let align2 = vec!(0.0, 0.0, 0.0, 0.0);

    let mut c_arr = vec!(10.0, 20.0,  30.0,  40.0,
                     02.0,  6.0,  24.0, 120.0,
                     01.0,  1.0,   2.0,   3.0,
                     05.0, 25.0, 125.0, 625.0);

    let res_arr = vec!(154.000, 1257.00, 102.000, 108.0,
                       306.000, 2137.00, 224.000, 324.0,
                       096.000,  825.50,  50.500,  49.5,
                       124.125,  335.25, 257.875, 447.0);

    
    let bptr_ofst = (&b_arr[0] as *const f64).align_offset(32);
    let cptr_ofst = (&c_arr[0] as *const f64).align_offset(32);
    println!("b offset = {}, c offset = {}", bptr_ofst, cptr_ofst);

    minimatrix_fmadd64(&a_arr, &b_arr, &mut c_arr);

    for row in 0 .. 4 {
        let ridx = row * 4;
        print!("[");
        for col in 0 .. 3 {
            print!(" {} == {}", res_arr[ridx + col], c_arr[ridx + col]);
            assert!(res_arr[ridx + col] == c_arr[ridx + col]);
        }
        println!("]");
    }
}
