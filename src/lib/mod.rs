#![feature(align_offset)]

extern crate rand;
extern crate ndarray;
extern crate matrixmultiply;

mod utilities;

use rand::prelude::*;
use core::arch::x86_64::__m256d;
use core::arch::x86_64::_mm256_broadcast_sd;
use core::arch::x86_64::_mm256_setzero_pd;
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

fn madd<T: FMADD>(a: T, b: T, c: &mut T) {
    c.fmadd(a, b);
}

fn dot_product_add<T: FMADD + Copy>(a: &[T], b: &[T], c: &mut T) {
    let mut i = 0;
    while i < a.len() {
        c.fmadd(a[i], b[i]);
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
#[cfg(any(target_arch = "x86_64"))]
unsafe fn folding_dot_prod(row: &[f64], col: &[f64]) -> f64 {
    let final_max = row.len();
    let first_max = final_max - final_max % 4;

    let mut sum = 0.0;
    let mut store: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
    let mut a: __m256d;
    let mut b: __m256d;
    let mut c: __m256d = _mm256_setzero_pd();
    
    for i in (0 .. first_max).step_by(4) {
        a = _mm256_loadu_pd(&row[i] as *const f64);
        b = _mm256_loadu_pd(&col[i] as *const f64);
        c = _mm256_fmadd_pd(a, b, c);
    }

    _mm256_storeu_pd(&mut store[0] as *mut f64, c);
    sum += store[0] + store[1] + store[2] + store[3];
    for i in first_max .. final_max {
        sum.fmadd(row[i], col[i]);
    }
    sum
}

#[target_feature(enable = "avx2")]
#[cfg(any(target_arch = "x86_64"))]
unsafe fn scalar_vec_fmadd_f64u(a_elt: &f64, b_row: *const f64, c_row: *mut f64) {
    let a: __m256d = _mm256_broadcast_sd(a_elt);
    let b: __m256d = _mm256_loadu_pd(b_row);
    let mut c: __m256d = _mm256_loadu_pd(c_row as *const f64);
    
    c = _mm256_fmadd_pd(a, b, c);
    _mm256_storeu_pd(c_row, c);
}

/// Calculates C = AB + C for a 4x4 submatrix with AVX2 instructions.
pub fn minimatrix_fmadd64(n_cols: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    /* For 4x4 matrices, the first row of AB + C can be represented as:
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
     * Generalizing this, one row (or 4 columns of one row) of C can be
     * calculated in 4 iterations 
     * row(C, i) = A[i][j]*row(B, j) + row(C, i)
     *
     * Iterating over the rows of C and applying this method, 4
     * columns of 4 rows C can be calculated in a tight loop. */
    
    for row in 0 .. 4 {
        let c_elt = get_elt(row, 0, n_cols);
        let mut c_row = &mut c[c_elt] as *mut f64;

        for col in 0 .. 4 {
            //let b_row_range = get_row(col, 4, n_cols);
            let b_elt = get_elt(col, 0, n_cols);
            let b_row = &b[b_elt] as *const f64;
            let aidx = get_elt(row, col, n_cols);
            let a_elt = &a[aidx];
            unsafe { 
                scalar_vec_fmadd_f64u(a_elt, b_row, c_row);
            }
        }
    }
}

pub fn matrix_madd(a_rows: usize, b_cols: usize, m_dim: usize,
                   a: &[f64], b: &[f64], c: &mut [f64]) {
    /* Check dimensionality */
    let a_len = a.len();
    let b_len = b.len();
    let c_len = c.len();
    /* Check for zeros before checking dimensions to prevent division by zero */
    match (a_rows, b_cols, m_dim, a_len, b_len, c_len) {
        (0, _, _, _, _, _) |  (_, 0, _, _, _, _) | (_, _, 0, _, _, _) |
        (_, _, _, 0, _, _) | (_, _, _, _, 0, _) => {
            panic!("Cannot do matrix multiplication where A or B have 0 elements")
        }
        (_, _, _, _, _, 0) => panic!("Cannot do matrix multiplication where C has 0 elements"),
        _ => {}
    }

    if a_len / a_rows != m_dim {
        panic!("{}\n{}*{} == {} != {}",
               "Dimensionality of A does not match parameters.",
               a_rows, m_dim, a_rows * m_dim, a_len);
    }

    if b_len / b_cols != m_dim {
        panic!("{}\n{}*{} == {} != {}",
               "Dimensionality of B does not match parameters.",
               b_cols, m_dim, b_cols * m_dim, b_len);
    }

    if c_len / a_rows != b_cols {
        panic!("{}\n{}*{} == {} != {}",
               "Dimensionality of C does not match parameters.",
               a_rows, b_cols, a_rows * b_cols, c_len);
    }
    
    if b_cols == 1 && a_rows == 1 {
        return dot_product_add(a, b, &mut c[0]);
    }
    /* 4col x 4row block of C += (b_cols x 4row of A)(4col * a_rows of B) */
    let row_stripes = a_rows - a_rows % 4;
    let col_pillars = b_cols - b_cols % 4;
    let blocks = col_pillars;

    if b_cols >= 4 && b_cols % 4 == 0 {
        for stripe in (0 .. row_stripes).step_by(4) {
            for pillar in (0 .. col_pillars).step_by(4) {
                let c_chunk_range = get_chunk(stripe, pillar, 4, 4, b_cols);
                let mut c_chunk = &mut c[c_chunk_range];
                
                for block in (0 .. blocks).step_by(4) {
                    let a_chunk_range = get_chunk(stripe, block, 4, 4, b_cols);
                    let b_chunk_range = get_chunk(block, pillar, 4, 4, b_cols);

                    let a_chunk = &a[a_chunk_range];
                    let b_chunk = &b[b_chunk_range];
                    minimatrix_fmadd64(b_cols, a_chunk, b_chunk, &mut c_chunk);
                }
            }
        }
    } else {
        for i in 0 .. a_rows {
            for j in 0 .. b_cols {
                for k in 0 .. b_cols {
                    c[i * b_cols + j].fmadd(a[i * b_cols + k], b[k * b_cols + j]);
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
