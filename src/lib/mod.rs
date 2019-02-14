#![feature(align_offset)]

extern crate ndarray;

mod utilities;

use std::cmp::min;
use core::arch::x86_64::__m256d;
use core::arch::x86_64::_mm256_broadcast_sd;
use core::arch::x86_64::_mm256_setzero_pd;
use core::arch::x86_64::{_mm256_loadu_pd,  _mm256_storeu_pd};
use core::arch::x86_64::_mm256_fmadd_pd;
pub use utilities::{get_elt, random_array, floateq};

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

#[inline(always)]
unsafe fn madd(a: *const f64, b: *const f64, c: *mut f64) {
    let res = (*a).mul_add(*b, *c);
    *c = res;
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

    unsafe {
        if b_cols == 1 && a_rows == 1 && m_dim >= 4 {
            return folding_dot_prod(a, b, &mut c[0]);
        } else {
            /*
            let a_ptr = &a[0] as *const f64;
            let b_ptr = &b[0] as *const f64;
            let c_ptr = &mut c[0] as *mut f64;
            const BLOCK: usize = 128;

            let row_stripes = a_rows - a_rows % BLOCK;
            let col_pillars = b_cols - b_cols % BLOCK;
            let blocks = col_pillars;
            
            for stripe in (0 .. row_stripes).step_by(BLOCK) {
                let subrows = min(BLOCK, a_rows - stripe);

                for pillar in (0 .. col_pillars).step_by(BLOCK) {
                    let subcols = min(BLOCK, b_cols - pillar);
                    let c_idx = get_elt(stripe, pillar, b_cols);
                    let c_chunk = &mut c[c_idx] as *mut f64;
                    
                    for block in (0 .. blocks).step_by(BLOCK) {
                        //let subblock = min(BLOCK, m_dim - block);
                        let a_idx = get_elt(stripe, block, m_dim);
                        let b_idx = get_elt(block, pillar, m_dim);

                        let a_chunk = &a[a_idx] as *const f64;
                        let b_chunk = &b[b_idx] as *const f64;
                        matrix_madd_block(m_dim, subrows, subcols,
                                          a_chunk, b_chunk, c_chunk);
                    }
                }
            }
             */
            matrix_madd_block(m_dim, a_rows, b_cols,
                              &a[0] as *const f64,
                              &b[0] as *const f64,
                              &mut c[0] as *mut f64);
        }
    }
}

unsafe fn matrix_madd_block(m_dim: usize, a_rows: usize, b_cols: usize, 
                                a: *const f64, b: *const f64, c: *mut f64) {
    /* 4col x 4row block of C += (b_cols x 4row of A)(4col * a_rows of B) */
    const MINIBLOCK: usize = 4;
    let row_stripes = a_rows - a_rows % MINIBLOCK;
    let col_pillars = b_cols - b_cols % MINIBLOCK;
    let blocks = col_pillars;

    for stripe in (0 .. row_stripes).step_by(MINIBLOCK) {
        for pillar in (0 .. col_pillars).step_by(MINIBLOCK) {
            let c_idx = get_elt(stripe, pillar, m_dim);
            let c_chunk = c.offset(c_idx as isize);
            
            for block in (0 .. blocks).step_by(MINIBLOCK) {
                let a_idx = get_elt(stripe, block, m_dim);
                let b_idx = get_elt(block, pillar, m_dim);

                let a_chunk = a.offset(a_idx as isize);
                let b_chunk = b.offset(b_idx as isize);
                minimatrix_fmadd64(b_cols, a_chunk, b_chunk, c_chunk);
            }
        }
    }

    for i in row_stripes .. a_rows {
        for j in 0 .. b_cols {
            for k in 0 .. b_cols {
                madd(a.offset((i * b_cols + k) as isize),
                     b.offset((k * b_cols + j) as isize),
                     c.offset((i * b_cols + j) as isize));
            }
        }
    }
}

#[target_feature(enable = "avx2")]
#[cfg(any(target_arch = "x86_64"))]
unsafe fn folding_dot_prod(row: &[f64], col: &[f64], c_elt: &mut f64) {
    let final_max = row.len();
    let first_max = final_max - final_max % 4;
    let mut c_tmp = *c_elt;

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
    c_tmp += store[0] + store[1] + store[2] + store[3];
    for i in first_max .. final_max {
        c_tmp.fmadd(row[i], col[i]);
    }
    *c_elt = c_tmp;
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
pub fn minimatrix_fmadd64(n_cols: usize, a: *const f64, b: *const f64, c: *mut f64) {
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
        let c_row;
        unsafe {
            c_row = c.offset(c_elt as isize);
        }

        for col in 0 .. 4 {
            let b_elt = get_elt(col, 0, n_cols);
            let aidx = get_elt(row, col, n_cols);
            unsafe {
                let a_elt = &*a.offset(aidx as isize);
                let b_row = b.offset(b_elt as isize);
                scalar_vec_fmadd_f64u(a_elt, b_row, c_row);
            }
        }
    }
}
