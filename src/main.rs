#![feature(align_offset)]

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

/// Calculates C = AB + C for a 4x4 submatrix with AVX2 instructions.
fn minimatrix_fmadd64(a_arr: &[f64], b_arr: &[f64], c_arr: &mut [f64]) {
    /* For 3x3 matrices, AB + C = C can be represented as:
     * | A11 A12 A13 | | B11 B12 B13 | |C11 C12 C13| |C11 C12 C13| 
     * | A21 A22 A23 |*| B21 B22 B23 |+|C21 C22 C23|=|C21 C22 C23|
     * | A31 A32 A33 | | B31 B32 B33 | |C31 C32 C33| |C31 C32 C33| 
     *
     * The elements of a resulting matrix multiplication are the dot
     * products of a row of A by a column of B. Addition normally
     * comes after finding the product, like so for the first row:
     *
     * A11B11+A12B21+A13B31+C11, A11B12+A12B22+A13B32+C12, A11B13+A12B23+A13B33+C13
     * ...
     * 
     * However, you can rethink process a bit, and reorder the
     * addition and multiplication. Looking at the first row of the
     * resulting matrix we can start by reordering the addition:
     *
     * A11B11 + C11 = C11, A11B12 + C12 = C12, A11B13 + C13 = C13
     * A12B21 + C11 = C11, A12B22 + C12 = C12, A12B23 + C13 = C13
     * A13B31 + C11 = C11, A13B32 + C12 = C12, A13B33 + C13 = C13
     * 
     * Generalizing this, each row of C can be calculated by iterating
     * over the columns of A, calculating the product of
     * A[i][j]*row(B, j)+row(C, i). So by iterating over the rows of C
     * and applying this method, C can be calculated in this way. This
     * is more efficient than the naive approach because we can
     * minimize cache misses and maximize the use of the cache as we
     * go through the matrices.
     * 
     */
    unsafe {
        let bptr_ofst = (&b_arr[0] as *const f64).align_offset(32);
        let cptr_ofst = (&c_arr[0] as *const f64).align_offset(32);
        if bptr_ofst == 0 && cptr_ofst == 0 {
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

fn floateq(a: f64, b: f64) -> bool {
    use std::f64;
    
    let abs_a = a.abs();
    let abs_b = b.abs();
    let diff = (a - b).abs();

    if a == b { // Handle infinities.
	true
    } else if a == 0.0 || b == 0.0 || diff < f64::MIN_POSITIVE {
	// One of a or b is zero (or both are extremely close to it,) use absolute error.
	diff < (f64::EPSILON * f64::MIN_POSITIVE)
    } else { // Use relative error.
	(diff / f64::min(abs_a + abs_b, f64::MAX)) < f64::EPSILON
    }
}

fn main() {
    let a_arr = vec!(1.0, 2.0, 3.0, 4.0,
                     7.0, 6.0, 5.0, 4.0,
                     0.5, 1.0, 2.0, 4.0,
                     8.0, 2.0, 0.5, 0.125);

    let _align1 = vec!(0.0, 0.0, 0.0, 0.0);

    let b_arr = vec!(12.0,  13.0, 15.0, 17.0,
                     01.0,   2.0,  3.0,  4.0,
                     42.0, 404.0, 13.0,  9.0,
                     01.0,   2.0,  3.0,  4.0);

    let _align2 = vec!(0.0, 0.0, 0.0, 0.0);

    let mut c_arr = vec!(10.0, 20.0,  30.0,  40.0,
                     02.0,  6.0,  24.0, 120.0,
                     01.0,  1.0,   2.0,   3.0,
                     05.0, 25.0, 125.0, 625.0);

    let res_arr = vec!(154.000, 1257.00, 102.000, 108.0,
                       306.000, 2137.00, 224.000, 324.0,
                       096.000,  825.50,  50.500,  49.5,
                       124.125,  335.25, 257.875, 447.0);

    /*
    let bptr_ofst = (&b_arr[0] as *const f64).align_offset(32);
    let cptr_ofst = (&c_arr[0] as *const f64).align_offset(32);
    println!("b offset = {}, c offset = {}", bptr_ofst, cptr_ofst);
     */
    
    minimatrix_fmadd64(&a_arr, &b_arr, &mut c_arr);

    for row in 0 .. 4 {
        let ridx = row * 4;
        for col in 0 .. 3 {
            //assert!(res_arr[ridx + col] == c_arr[ridx + col]);
            assert!(floateq(res_arr[ridx + col], c_arr[ridx + col]));
        }
    }
}
