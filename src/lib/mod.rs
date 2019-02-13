#![feature(align_offset)]

extern crate rand;
extern crate ndarray;
extern crate matrixmultiply;

mod utilities;

use rand::prelude::*;
use core::arch::x86_64::__m256d;
use core::arch::x86_64::_mm256_broadcast_sd;
use core::arch::x86_64::{_mm256_load_pd,  _mm256_store_pd};
use core::arch::x86_64::{_mm256_loadu_pd,  _mm256_storeu_pd};
use core::arch::x86_64::_mm256_fmadd_pd;
use utilities::*;

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
pub fn minimatrix_fmadd64(n_cols: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    /* For 4x4 matrices, the first row of AB + C can be
     * represented as:
     *
     * A11B11 + A12B21 + A13B31 + A14B41 + C11,
     * A11B12 + A12B22 + A13B32 + A14B42 + C12, 
     * A11B13 + A12B23 + A13B33 + A14B43 + C13,
     * A11B14 + A12B24 + A13B34 + A14B44 + C14
     * 
     * However, the products and summation can be reordered:
     *
     * A11B11 + C11 = C11, A11B12 + C12 = C12, A11B13 + C13 = C13, A11B14 + C14 = C14,
     * A12B21 + C11 = C11, A12B22 + C12 = C12, A12B23 + C13 = C13, A12B24 + C14 = C14,
     * A13B31 + C11 = C11, A13B32 + C12 = C12, A13B33 + C13 = C13, A13B34 + C14 = C14,
     * A14B41 + C11 = C11, A14B42 + C12 = C12, A14B43 + C13 = C13, A14B44 + C14 = C14,
     * 
     * Generalizing this, one row (or 4 columns of one row) of C can
     * be calculated in 4 iterations by the successive product of an
     * element from the corresponding row of A and a row of B; row(C,
     * i) = A[i][j]*row(B, j) + row(C, i)
     *
     * Iterating over the rows of C and applying this method, 4
     * columns of 4 rows C can be calculated in this way. This is more
     * efficient than the naive approach because we can minimize cache
     * misses and maximize the use of the cache as we go through the
     * matrices. */
    for row in 0 .. 4 {
        let row_range = get_row(row, 4, n_cols);
        let mut c_row = &mut c[row_range];

        for col in 0 .. 4 {
            let b_row_range = get_row(col, 4, n_cols);
            let b_row = &b[b_row_range];
            let aidx = get_elt(row, col, n_cols);
            if aidx == a.len() {
                panic!("{}*{}+{}={}", row, n_cols, col, row * n_cols + col);
            }
            let a_elt = &a[aidx];
            unsafe { 
                scalar_vec_fmadd_f64u(a_elt, b_row, &mut c_row);
            }
        }
    }
}

pub fn matrix_madd(n_cols: usize, m_rows: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    if n_cols == 1 && m_rows == 1 {
        c[0] = a[0]*b[0] + c[0];
        return;
    }
    /* 4col x 4row block of C += (n_cols x 4row of A)(4col * m_rows of B) */
    let row_stripes = m_rows - m_rows % 4;
    let col_pillars = n_cols - n_cols % 4;
    let blocks = col_pillars;

    if n_cols >= 4 && n_cols % 4 == 0 {
        for stripe in (0 .. row_stripes).step_by(4) {
            for pillar in (0 .. col_pillars).step_by(4) {
                let c_chunk_range = get_chunk(stripe, pillar, 4, 4, n_cols);
                let mut c_chunk = &mut c[c_chunk_range];
                
                for block in (0 .. blocks).step_by(4) {
                    let a_chunk_range = get_chunk(stripe, block, 4, 4, n_cols);
                    let b_chunk_range = get_chunk(block, pillar, 4, 4, n_cols);

                    let a_chunk = &a[a_chunk_range];
                    let b_chunk = &b[b_chunk_range];
                    minimatrix_fmadd64(n_cols, a_chunk, b_chunk, &mut c_chunk);
                }
            }
        }
    } else {
        for i in 0 .. m_rows {
            for j in 0 .. n_cols {
                for k in 0 .. n_cols {
                    c[i * n_cols + j] += a[i * n_cols + k] * b[k * n_cols + j];
                }
            }
        }
    }
}

pub fn floateq(a: f64, b: f64) -> bool {
    use std::f64;
    
    let abs_a = a.abs();
    let abs_b = b.abs();
    let diff = (a - b).abs();
    let epsilon = 0.00001;

    if a == b {
	true
    } else if a == 0.0 || b == 0.0 || diff < f64::MIN_POSITIVE {
	diff < (epsilon * f64::MIN_POSITIVE)
    } else { 
	(diff / f64::min(abs_a + abs_b, f64::MAX)) < epsilon
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
