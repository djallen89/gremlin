#![feature(align_offset)]

extern crate ndarray;

mod utilities;

use std::cmp::{max};
use core::arch::x86_64::__m256d;
use core::arch::x86_64::_mm256_set_pd;
use core::arch::x86_64::_mm256_broadcast_sd;
use core::arch::x86_64::_mm256_setzero_pd;
use core::arch::x86_64::{_mm256_loadu_pd,  _mm256_storeu_pd};
use core::arch::x86_64::{_mm256_mul_pd, _mm256_fmadd_pd};
pub use utilities::{get_elt, random_array, floateq};
pub use utilities::{matrix_madd_n_sq, matrix_madd_nxm, matrix_madd_nmp};

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

/// Performs C <= AB + C, where A is a matrix of n rows by m columns,
/// B is a matrix of m rows by p columns, and C is a matrix of n rows
/// by p columns.
pub fn matrix_madd(n_rows: usize, m_dim: usize, p_cols: usize,
                   a: &[f64], b: &[f64], c: &mut [f64]) {
    /* Check dimensionality */
    let a_len = a.len();
    let b_len = b.len();
    let c_len = c.len();
    /* Check for zeros before checking dimensions to prevent division by zero */
    match (n_rows, p_cols, m_dim, a_len, b_len, c_len) {
        (0, _, _, _, _, _) |  (_, 0, _, _, _, _) | (_, _, 0, _, _, _) |
        (_, _, _, 0, _, _) | (_, _, _, _, 0, _) => {
            panic!("Cannot do matrix multiplication where A or B have 0 elements")
        }
        (_, _, _, _, _, 0) => panic!("Cannot do matrix multiplication where C has 0 elements"),
        _ => {}
    }

    if a_len / n_rows != m_dim {
        /* A is n * m*/
        panic!("{}\n{}*{} == {} != {}",
               "Dimensionality of A does not match parameters.",
               n_rows, m_dim, n_rows * m_dim, a_len);
    } else if b_len / p_cols != m_dim {
        /* B is m * p */
        panic!("{}\n{}*{} == {} != {}",
               "Dimensionality of B does not match parameters.",
               p_cols, m_dim, p_cols * m_dim, b_len);
    } else if c_len / n_rows != p_cols {
        /* C is n * p */
        panic!("{}\n{}*{} == {} != {}",
               "Dimensionality of C does not match parameters.",
               n_rows, p_cols, n_rows * p_cols, c_len);
    }

    unsafe {
        if p_cols < 4 || n_rows < 4 {
            return small_matrix_mul_add(n_rows, m_dim, p_cols, a, b, c)
        }

        let a_ptr = &a[0] as *const f64;
        let b_ptr = &b[0] as *const f64;
        let c_ptr = &mut c[0] as *mut f64;

        if n_rows <= 4 && p_cols <= 4 && m_dim <= 4 {
            return minimatrix_fmadd64(m_dim, a_ptr, b_ptr, c_ptr);
        }

        if n_rows <= 32 && p_cols <= 32 && m_dim <= 32 {
            return matrix_madd_inner_block(m_dim,
                                           n_rows, p_cols, m_dim,
                                           a_ptr, b_ptr, c_ptr);
        }
        // (* 32 8) 256
        if n_rows <= 256 && p_cols <= 256 && m_dim <= 256 {
            return matrix_madd_block(m_dim,
                                     n_rows, p_cols, m_dim,
                                     a_ptr, b_ptr, c_ptr);
        }
        
        recursive_matrix_mul(p_cols, m_dim,
                             n_rows, p_cols, m_dim,
                             a_ptr, b_ptr, c_ptr);
    }
}

fn small_matrix_mul_add(n_rows: usize, m_dim: usize, p_cols: usize,
                        a: &[f64], b: &[f64], c: &mut [f64]) {

    for i in 0 .. n_rows {
        for j in 0 .. p_cols {
            for k in 0 .. m_dim {
                c[i * p_cols + j].fmadd(a[i * m_dim + k], b[k * m_dim + j]);
            }
        }
    }
}

unsafe fn recursive_matrix_mul(p_cols_orig: usize, m_dim_orig: usize,
                               n_rows: usize, p_cols: usize, m_dim: usize,
                               a: *const f64, b: *const f64, c: *mut f64) {

    if n_rows == 4 && p_cols == 4 && m_dim == 4 {
        return minimatrix_fmadd64(m_dim_orig, a, b, c);
    }

    if (n_rows <= 32 && p_cols <= 32 && m_dim <= 32) || m_dim_orig % 32 != 0 {
        return matrix_madd_inner_block(m_dim_orig,
                                       n_rows, p_cols, m_dim,
                                       a, b, c);
    }

    if n_rows <= 256 && p_cols <= 256 && m_dim <= 256 {
        return matrix_madd_block(m_dim_orig,
                                 n_rows, p_cols, m_dim,
                                 a, b, c);
    }

    let maxn = max(max(n_rows, p_cols), m_dim);
    if maxn == n_rows {
        let n_rows_new = n_rows / 2;
        let aidx2 = n_rows_new * m_dim_orig;
        let a_2 = a.offset(aidx2 as isize);
        let cidx2 = n_rows_new * p_cols_orig;
        let c_2 = c.offset(cidx2 as isize);
        recursive_matrix_mul(p_cols_orig, m_dim_orig,
                             n_rows_new, p_cols, m_dim,
                             a, b, c);
        recursive_matrix_mul(p_cols_orig, m_dim_orig,
                             n_rows_new, p_cols, m_dim,
                             a_2, b, c_2);
    } else if maxn == p_cols {
        let p_cols_new = p_cols / 2;
        let b_2 = b.offset(p_cols_new as isize);
        let c_2 = c.offset(p_cols_new as isize);
        recursive_matrix_mul(p_cols_orig, m_dim_orig,
                             n_rows, p_cols_new, m_dim,
                             a, b, c);
        recursive_matrix_mul(p_cols_orig, m_dim_orig,
                             n_rows, p_cols_new, m_dim,
                             a, b_2, c_2);
    } else {
        let m_dim_new = m_dim / 2;
        let a_2 = a.offset(m_dim_new as isize);
        let bidx2 = m_dim_new * p_cols_orig;
        let b_2 = b.offset(bidx2 as isize);
        
        recursive_matrix_mul(p_cols_orig, m_dim_orig,
                             n_rows, p_cols, m_dim_new,
                             a, b, c);
        recursive_matrix_mul(p_cols_orig, m_dim_orig,
                             n_rows, p_cols, m_dim_new,
                             a_2, b_2, c);
    }

}

unsafe fn matrix_madd_block(m_dim: usize, a_rows: usize, b_cols: usize, _f_blocks: usize,
                                  a: *const f64, b: *const f64, c: *mut f64) {
    /* 4col x 4row block of C += (b_cols x 4row of A)(4col * a_rows of B) */
    let miniblock = 128;
    let row_stripes = a_rows - a_rows % miniblock;
    let col_pillars = b_cols - b_cols % miniblock;
    let blocks = col_pillars;

    for pillar in (0 .. col_pillars).step_by(miniblock) {
        for block in (0 .. blocks).step_by(miniblock) {
            for stripe in (0 .. row_stripes).step_by(miniblock) {
                let c_idx = get_elt(stripe, pillar, m_dim);
                let c_chunk = c.offset(c_idx as isize);
                    
                let a_idx = get_elt(stripe, block, m_dim);
                let b_idx = get_elt(block, pillar, m_dim);
                let a_chunk = a.offset(a_idx as isize);
                let b_chunk = b.offset(b_idx as isize);
                matrix_madd_inner_block(m_dim,
                                        miniblock, miniblock, miniblock,
                                        a_chunk, b_chunk, c_chunk);
            }
        }
    }
}

#[inline(always)]
unsafe fn matrix_madd_inner_block(m_dim: usize, a_rows: usize, b_cols: usize, c_blocks: usize,
                                  a: *const f64, b: *const f64, c: *mut f64) {
    /* 4col x 4row block of C += (b_cols x 4row of A)(4col * a_rows of B) */
    const MINIBLOCK: usize = 4;
    let row_stripes = a_rows - a_rows % MINIBLOCK;
    let col_pillars = b_cols - b_cols % MINIBLOCK;
    let sub_blocks = c_blocks - c_blocks % MINIBLOCK;

    for stripe in (0 .. row_stripes).step_by(MINIBLOCK) {
        for block in (0 .. sub_blocks).step_by(MINIBLOCK) {
            for pillar in (0 .. col_pillars).step_by(MINIBLOCK) {
                let c_idx = get_elt(stripe, pillar, m_dim);
                let c_chunk = c.offset(c_idx as isize);
            
                let a_idx = get_elt(stripe, block, m_dim);
                let b_idx = get_elt(block, pillar, m_dim);
                let a_chunk = a.offset(a_idx as isize);
                let b_chunk = b.offset(b_idx as isize);
                minimatrix_fmadd64(m_dim, a_chunk, b_chunk, c_chunk);
            }
            
            /* Say you have a 5x5 matrix. 
             * |A11 A12 A13 A14 A15||B11 B12 B13 B14 B15| |C11 C12 C13 C14 C15| 
             * |A21 A22 A23 A24 A25||B21 B22 B23 B24 B25| |C21 C22 C23 C24 C25| 
             * |A31 A32 A33 A34 A35||B31 B32 B33 B34 B35|+|C31 C32 C33 C34 C35|
             * |A41 A42 A43 A44 A45||B41 B42 B43 B44 B45| |C41 C42 C43 C44 C45|
             * |A51 A52 A53 A54 A55||B51 B52 B53 B54 B55| |C51 C52 C53 C54 C55|
             *
             * Then at this point, C has been (partially) handled from
             * C(1..4),(1..4).  C15 to C45 can be handled by
             *
             * C15 += A11*B15 + A12*B25 + A13*B35 + A14*B45 | + A15*B55  (later)
             * C25 += A21*B15 + A22*B25 + A23*B35 + A24*B45 | + A25*B55  (later)
             * C35 += A31*B15 + A32*B25 + A33*B35 + A34*B45 | + A35*B55  (later)
             * C45 += A41*B15 + A42*B25 + A43*B35 + A44*B45 | + A45*B55  (later)
             *
             * However, we can't go past this stripe. We will have to
             * handle row(B, 5+) later. Regardless, This presents
             * another opportunity for vectorization. */
            for column in col_pillars .. b_cols {
                let b1 = *b.offset(get_elt(block, column, m_dim) as isize);
                let b2 = *b.offset(get_elt(block + 1, column, m_dim) as isize);
                let b3 = *b.offset(get_elt(block + 2, column, m_dim) as isize);
                let b4 = *b.offset(get_elt(block + 3, column, m_dim) as isize);
                let b_col_vec = _mm256_set_pd(b1, b2, b3, b4);
                for row in 0 .. MINIBLOCK {
                    let c_idx = get_elt(stripe + row, column, m_dim) as isize;
                    let mut ci = c.offset(c_idx);
                    let a_idx = get_elt(stripe + row, block, m_dim);
                    let a_row = _mm256_loadu_pd(a.offset(a_idx as isize));
                    let res = _mm256_mul_pd(a_row, b_col_vec);
                    let mut res_arr: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
                    _mm256_storeu_pd(&mut res_arr[0] as *mut f64, res);
                    *ci += res_arr[0] + res_arr[1] + res_arr[2] + res_arr[3];
                }
            }
        }
    }

    /* Say you have a 5x5 matrix. 
     * |A11 A12 A13 A14 A15||B11 B12 B13 B14 B15| |C11 C12 C13 C14 C15| 
     * |A21 A22 A23 A24 A25||B21 B22 B23 B24 B25| |C21 C22 C23 C24 C25| 
     * |A31 A32 A33 A34 A35||B31 B32 B33 B34 B35|+|C31 C32 C33 C34 C35|
     * |A41 A42 A43 A44 A45||B41 B42 B43 B44 B45| |C41 C42 C43 C44 C45|
     * |A51 A52 A53 A54 A55||B51 B52 B53 B54 B55| |C51 C52 C53 C54 C55|
     *
     * Then at this point, C has been (partially) handled from
     * C(1..4),(1..5), so we need to finish up everything:
     * C11 += A15B51, C12 += A15B52, C13 += A15B53, C14 += A15B54, C15 += A15B55
     * C21 += A25B51, C22 += A25B52, C23 += A25B53, C24 += A25B54, C25 += A25B55
     * C31 += A35B51, C32 += A35B52, C33 += A35B53, C34 += A35B54, C35 += A35B55
     * C41 += A45B51, C42 += A45B52, C43 += A45B53, C44 += A45B54, C45 += A45B55
     *
     * And of course the last row of C is the fifth row of A dotted by
     * each column of B:
     * C51 += AR5.BC1, C52 += AR5.BC2, C53 += AR5.BC3, C54 += AR5.BC4, C55 += AR5.BC5
     */

    if  row_stripes != a_rows {
        for col in 0 .. 
    }
    for row in row_stripes .. a_rows {
        for col in 0 .. b_cols {
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
