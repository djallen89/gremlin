#![feature(align_offset)]

extern crate rand;
extern crate ndarray;
extern crate matrixmultiply;

use rand::prelude::*;
use core::arch::x86_64::__m256d;
use core::arch::x86_64::_mm256_broadcast_sd;
use core::arch::x86_64::{_mm256_load_pd,  _mm256_store_pd};
use core::arch::x86_64::{_mm256_loadu_pd,  _mm256_storeu_pd};
use core::arch::x86_64::_mm256_fmadd_pd;

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

pub fn madd<T: FMADD>(a: T, b: T, c: &mut T) {
    c.fmadd(a, b);
}

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
pub fn minimatrix_fmadd64(cols: usize, nrows: usize, a_arr: &[f64], b_arr: &[f64], c_arr: &mut [f64]) {
    /* For 3x3 matrices, AB + C = C can be represented as:
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
     * go through the matrices. */
    for row in 0 .. 4 {
        let ridx = row * 4;
        let mut c_row = &mut c_arr[ridx .. ridx + 4];
        
        for col in 0 .. 4 {
            let bidx = col * 4;
            let b_row = &b_arr[bidx .. bidx + 4];
            let a_elt = &a_arr[ridx + col];
            unsafe { 
                scalar_vec_fmadd_f64u(a_elt, b_row, &mut c_row);
            }
        }
    }
}

pub fn matrix_madd(n_cols: usize, m_rows: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    if n_cols == 1 && m_rows == 1 {
        c[0] = a[0]*b[0] + c[0];
    }
    /* 4col x 4row block of C += (n_cols x 4row of A)(4col * m_rows of B) */
    let row_blocks = (m_rows - m_rows % 4) / 4;
    let col_blocks = (n_cols - n_cols % 4) / 4;

    /* Zig zag in 4 column chunks of B through 4 row stripes of A */
    for b_col_group in 0 .. col_blocks {
        let bidx = b_col_group * 4;
        let b_block = &b[bidx .. bidx + 4 * n_cols];

        /* Dispatch the chunks of A and C in 4x4 blocks*/
        for a_row_group in 0 .. row_blocks {
            let idx0 = a_row_group * n_cols + b_col_group * 4;
            let idxf = idx0 + 4 * n_cols;
            let a_block = &a[idx0 .. idxf];
            let c_block = &mut c[idx0 .. idxf];
            minimatrix_fmadd64(n_cols, 4, a_block, b_block, c_block);
        }
/*
        if row_blocks * 4 < m_rows {
            let rem_rows = m_rows - 4 * row_blocks;
            let idx0 = row_blocks * n_cols + b_col_group * 4;
            let a_block = &a[idx0 .. ];
            let c_block = &mut c[idx0 ..];
            minimatrix_fmadd64(n_cols, rem_rows, a_block, b_block, c_block);
        }
*/
    }
/*
    for b_col in col_blocks * 4 .. n_cols {
        for a_row in 0 .. m_rows {
            let mut sum = 0.0;
            for k in 0 .. m_rows {
                sum += a[a_row * n_cols + k]*b[k * n_cols + b_col];
            }
            c[a_row * n_cols + b_col] += sum;
        }
    }
*/
/*
    for a_row_group in in 0 .. row_blocks * 4 {
        let idx0 = a_row_group * n_cols
        let idxf = 

        for b_col_group in col_blocks .. n_cols {
            
        }

        
    }
*/
}

pub fn floateq(a: f64, b: f64) -> bool {
    use std::f64;
    
    let abs_a = a.abs();
    let abs_b = b.abs();
    let diff = (a - b).abs();

    if a == b {
	true
    } else if a == 0.0 || b == 0.0 || diff < f64::MIN_POSITIVE {
	diff < (f64::EPSILON * f64::MIN_POSITIVE)
    } else { 
	(diff / f64::min(abs_a + abs_b, f64::MAX)) < f64::EPSILON
    }

}

pub fn random_array<T>(cols: usize, rows: usize, low: T, high: T) -> Vec<T>
    where T: rand::distributions::uniform::SampleUniform
{
    use rand::distributions::Uniform;
    use std::usize;
    
    assert!(usize::MAX / rows > cols);
    
    let interval = Uniform::from(low .. high);
    let mut rng = rand::thread_rng();
    let mut arr = Vec::with_capacity(rows * cols);
    
    for _ in 0 .. rows * cols {
        arr.push(interval.sample(&mut rng))
    }

    return arr;
}
