//#![feature(align_offset)]
extern crate ndarray;

mod utilities;

use std::cmp::{min, max};
use core::arch::x86_64::{__m256d, _mm256_set_pd, _mm256_broadcast_sd};
use core::arch::x86_64::{_mm256_setzero_pd, _mm256_loadu_pd,  _mm256_storeu_pd};
use core::arch::x86_64::_mm256_fmadd_pd;
pub use utilities::{get_elt, random_array, float_eq, test_equality};
pub use utilities::{matrix_madd_n_sq, matrix_madd_nxm, matrix_madd_nmp};
pub use utilities::FMADD;

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
        if n_rows == 1 && p_cols == 1 && m_dim >= 4 {
            return single_dot_prod_add(a, b, &mut c[0])
        } else  if p_cols < 4 || n_rows < 4 {
            return small_matrix_mul_add(n_rows, m_dim, p_cols, a, b, c)
        }

        let a_ptr = &a[0] as *const f64;
        let b_ptr = &b[0] as *const f64;
        let c_ptr = &mut c[0] as *mut f64;

        if n_rows <= 4 && p_cols <= 4 && m_dim <= 4 {
            return minimatrix_fmadd_f64(m_dim, a_ptr, b_ptr, c_ptr);
        }

        if n_rows <= 32 && p_cols <= 32 && m_dim <= 32 {
            return matrix_madd_inner_block(m_dim, p_cols,
                                           n_rows, p_cols, m_dim,
                                           a_ptr, b_ptr, c_ptr);
        }
        if n_rows <= 256 && p_cols <= 256 && m_dim <= 256 {
            return matrix_madd_block(m_dim, p_cols,
                                     n_rows, p_cols, m_dim,
                                     a_ptr, b_ptr, c_ptr);
        }
        
        recursive_matrix_mul(p_cols, m_dim,
                             n_rows, p_cols, m_dim,
                             a, b, c)
    }
}

#[target_feature(enable = "avx2")]
#[cfg(any(target_arch = "x86_64"))]
/* This function is simply for the case of the root arrays A and B
 * being a single row and a single column. */
unsafe fn single_dot_prod_add(row: &[f64], col: &[f64], c_elt: &mut f64) {
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

fn recursive_matrix_mul(p_cols_orig: usize, m_dim_orig: usize,
                        n_rows: usize, p_cols: usize, m_dim: usize,
                        a: &[f64], b: &[f64], c: &mut [f64]) {
    let a_ptr = &a[0] as *const f64;
    let b_ptr = &b[0] as *const f64;
    let c_ptr = &mut c[0] as *mut f64;

    unsafe {
        if n_rows == 4 && p_cols == 4 && m_dim == 4 {
            return minimatrix_fmadd_f64(m_dim_orig, a_ptr, b_ptr, c_ptr);
        }

        if (n_rows <= 32 && p_cols <= 32 && m_dim <= 32) || m_dim_orig % 32 != 0 {
            return matrix_madd_inner_block(m_dim_orig, p_cols_orig,
                                           n_rows, p_cols, m_dim,
                                           a_ptr, b_ptr, c_ptr);
        }

        if n_rows <= 256 && p_cols <= 256 && m_dim <= 256 {
            return matrix_madd_block(m_dim_orig, p_cols,
                                     n_rows, p_cols, m_dim,
                                     a_ptr, b_ptr, c_ptr);
        }
    }
    
    let maxn = max(max(n_rows, p_cols), m_dim);
    if maxn == n_rows {
        let n_rows_new = n_rows / 2;
        let aidx2 = n_rows_new * m_dim_orig;
        let a_2 = &a[aidx2 ..];
        let cidx2 = n_rows_new * p_cols_orig;
        let (c_1, c_2) = c.split_at_mut(cidx2);
        recursive_matrix_mul(p_cols_orig, m_dim_orig,
                             n_rows_new, p_cols, m_dim,
                             a, b, c_1);
        recursive_matrix_mul(p_cols_orig, m_dim_orig,
                             n_rows_new, p_cols, m_dim,
                             a_2, b, c_2);
    } else if maxn == p_cols {
        let p_cols_new = p_cols / 2;
        let b_2 = &b[p_cols_new ..];
        let (c_1, c_2) = c.split_at_mut(p_cols_new);
        recursive_matrix_mul(p_cols_orig, m_dim_orig,
                             n_rows, p_cols_new, m_dim,
                             a, b, c_1);
        recursive_matrix_mul(p_cols_orig, m_dim_orig,
                             n_rows, p_cols_new, m_dim,
                             a, b_2, c_2);
    } else {
        let m_dim_new = m_dim / 2;
        let a_2 = &a[m_dim_new ..];
        let bidx2 = m_dim_new * p_cols_orig;
        let b_2 = &b[bidx2 ..];
        
        recursive_matrix_mul(p_cols_orig, m_dim_orig,
                             n_rows, p_cols, m_dim_new,
                             a, b, c);
        recursive_matrix_mul(p_cols_orig, m_dim_orig,
                             n_rows, p_cols, m_dim_new,
                             a_2, b_2, c);
    }

}

unsafe fn matrix_madd_block(m_dim: usize, p_cols: usize,
                            a_rows: usize, b_cols: usize, sub_m: usize,
                                  a: *const f64, b: *const f64, c: *mut f64) {
    /* 4col x 4row block of C += (b_cols x 4row of A)(4col * a_rows of B) */
    let miniblock = min(128, m_dim);
    let row_stripes = a_rows - a_rows % miniblock;
    let col_pillars = b_cols - b_cols % miniblock;
    let blocks = sub_m - sub_m % miniblock;
    for pillar in (0 .. col_pillars).step_by(miniblock) {
        for block in (0 .. blocks).step_by(miniblock) {
            for stripe in (0 .. row_stripes).step_by(miniblock) {
                let c_idx = get_elt(stripe, pillar, p_cols);
                let c_chunk = c.offset(c_idx as isize);
                    
                let a_idx = get_elt(stripe, block, m_dim);
                let a_chunk = a.offset(a_idx as isize);
                
                let b_idx = get_elt(block, pillar, p_cols);
                let b_chunk = b.offset(b_idx as isize);
                matrix_madd_inner_block(m_dim, p_cols,
                                        miniblock, miniblock, miniblock,
                                        a_chunk, b_chunk, c_chunk);
            }
        }
    }
}

#[inline(always)]
unsafe fn matrix_madd_inner_block(m_dim: usize, p_cols: usize,
                                  a_rows: usize, b_cols: usize, sub_m: usize,
                                  a: *const f64, b: *const f64, c: *mut f64) {
    /* 4col x 4row block of C += (b_cols x 4row of A)(4col * a_rows of B) */
    const MINIBLOCK: usize = 4;
    let row_stripes = a_rows - a_rows % MINIBLOCK;
    let col_pillars = b_cols - b_cols % MINIBLOCK;
    let sub_blocks = sub_m - sub_m % MINIBLOCK;


    for stripe in (0 .. row_stripes).step_by(MINIBLOCK) {
        for block in (0 .. sub_blocks).step_by(MINIBLOCK) {
            for pillar in (0 .. col_pillars).step_by(MINIBLOCK) {
                let c_idx = get_elt(stripe, pillar, p_cols);
                let c_chunk = c.offset(c_idx as isize);
            
                let a_idx = get_elt(stripe, block, m_dim);
                let b_idx = get_elt(block, pillar, p_cols);
                let a_chunk = a.offset(a_idx as isize);
                let b_chunk = b.offset(b_idx as isize);
                minimatrix_fmadd_f64(m_dim, a_chunk, b_chunk, c_chunk);
            }
        }
            
        /* Say A is a 7x6 matrix and B is a 6x7 matrix, then C must be a 7x7 matrix
         * |11 12 13 14 15 16||11 12 13 14 15 16 17| |11 12 13 14 15 16 17| 
         * |21 22 23 24 25 26||21 22 23 24 25 26 27| |21 22 23 24 25 26 27| 
         * |31 32 33 34 35 36||31 32 33 34 35 36 37| |31 32 33 34 35 36 37|
         * |41 42 43 44 45 46||41 42 43 44 45 46 47|+|41 42 43 44 45 46 47|
         * |51 52 53 54 55 56||51 52 53 54 55 56 57| |51 52 53 54 55 56 57|
         * |61 62 63 64 65 66||61 62 63 64 65 66 67| |61 62 63 64 65 66 67|
         * |71 72 73 74 75 76|                       |71 72 73 74 75 76 77|
         *
         * Then at this point, C has been (partially) handled from
         * C(1..4),(1..4).  C15 to C47 can be partially handled by
         *
         * C15 += A11B15 + A12B25 + A13B35 + A14B45 | + A15B55 + A16B65  (later)
         * C16 += A11B16 + A12B26 + A13B36 + A14B46 | + A15B56 + A16B66  (later)
         * C17 += A11B17 + A12B27 + A13B37 + A14B47 | + A15B57 + A16B67  (later)
         * 
         * C25 += A21B15 + A22B25 + A23B35 + A24B45 | + A25B55 + A26B65  (later)
         * C26 += A21B16 + A22B26 + A23B36 + A24B46 | + A25B56 + A26B66  (later)
         * C27 += A21B17 + A22B27 + A23B37 + A24B47 | + A25B57 + A26B67  (later)
         *
         * C35 += A31B15 + A32B25 + A33B35 + A34B45 | + A35B55 + A36B65  (later)
         * C36 += A31B16 + A32B26 + A33B36 + A34B45 | + A35B56 + A36B66  (later)
         * C37 += A31B17 + A32B27 + A33B37 + A34B45 | + A35B57 + A36B67  (later)
         *
         * C45 += A41B15 + A42B25 + A43B35 + A44B45 | + A45B55 + A46B65  (later)
         * C46 += A31B16 + A32B26 + A33B36 + A34B45 | + A35B56 + A36B66  (later)
         * C47 += A31B17 + A32B27 + A33B37 + A34B45 | + A35B57 + A36B67  (later)
         */

        for row in stripe .. stripe + MINIBLOCK {
            for k in sub_blocks .. sub_m {
                let c_idx = get_elt(row, k, p_cols) as isize;
                let c_elt = c.offset(c_idx);

                for col in 0 .. col_pillars {
                    let a_idx = get_elt(row, col, m_dim) as isize;
                    let b_idx = get_elt(col, k, p_cols) as isize;
                    let a_elt = a.offset(a_idx);
                    let b_elt = b.offset(b_idx);

                    madd(a_elt, b_elt, c_elt);                    
                }
            }
        }        
    }

    /* Returning to the (7x6)(6x7) + (7x7) case...
     * At this point, C has been partially handled from
     * C(1..4),(1..7), so we need to finish up everything:
     * C11+=A15B51, C12+=A15B52, C13+=A15B53, C14+=A15B54, C15+=A15B55, C16+=A15B56
     * C11+=A16B61, C12+=A16B62, C13+=A16B63, C14+=A16B64, C15+=A16B65, C16+=A16B66
     * C11+=A17B71, C12+=A17B72, C13+=A17B73, C14+=A17B74, C15+=A17B75, C16+=A17B76
     *
     * C21+=A25B51, C22+=A25B52, C23+=A25B53, C24+=A25B54, C25+=A25B55, C16+=A15B56
     * ...
     * C31+=A35B51, C32+=A35B52, C33+=A35B53, C34+=A35B54, C35+=A35B55, C36+=A15B56
     * ...
     * C41+=A45B51, C42+=A45B52, C43+=A45B53, C44+=A45B54, C45+=A45B55, C46+=A15B56
     */

    for row in 0 .. row_stripes {
        for col in 0 .. b_cols {
            let c_idx = get_elt(row, col, p_cols) as isize;
            let c_elt = c.offset(c_idx);
            
            for k in sub_blocks .. m_dim {
                let a_idx = get_elt(row, k, m_dim) as isize;
                let b_idx = get_elt(k, col, p_cols) as isize;

                let a_elt = a.offset(a_idx);
                let b_elt = b.offset(b_idx);
                madd(a_elt, b_elt, c_elt);
            }
        }
        /*
        for rem_row in row_stripes .. a_rows {
            let a_idx = get_elt(row, rem_row, m_dim) as isize;
            let a_elt = a.offset(a_idx);
            let a_elt_mult = _mm256_broadcast_sd(&*a_elt);

            for pillar in (0 .. col_pillars).step_by(MINIBLOCK) {
                let b_idx = get_elt(rem_row, pillar, p_cols) as isize;
                let b_row_vec = _mm256_loadu_pd(b.offset(b_idx));

                let c_idx = get_elt(row, pillar, p_cols) as isize;
                let mut c_row_vec = _mm256_loadu_pd(c.offset(c_idx));

                c_row_vec = _mm256_fmadd_pd(a_elt_mult, b_row_vec, c_row_vec);
                _mm256_storeu_pd(c.offset(c_idx), c_row_vec);
            }
            
            for col in col_pillars .. b_cols {
                let b_idx = get_elt(rem_row, col, p_cols) as isize;
                let b_elt = b.offset(b_idx);
                let c_idx = get_elt(row, col, p_cols) as isize;
                let c_elt = c.offset(c_idx);
                madd(a_elt, b_elt, c_elt);
            }
        }
         */
    }


    /* And of course the last 3 rows of C are the corresponding row of A dotted by
     * each column of B:
     * C51+=AR5.BC1, C52+=AR5.BC2, C53+=AR5.BC3, C54+=AR5.BC4, C55+=AR5.BC5, C56+=AR5.BC6
     * C61+=AR6.BC1, C62+=AR6.BC2, C63+=AR6.BC3, C64+=AR6.BC4, C65+=AR6.BC6, C66+=AR6.BC6
     * C71+=AR7.BC1, C72+=AR7.BC2, C73+=AR7.BC3, C74+=AR7.BC4, C75+=AR7.BC7, C76+=AR7.BC6
     *
     * In a bit more detail...
     * C51+=A51B11, C52+=A51B12, C53+=A51B13, C54+=A51B14, C55+=A51B15, C56+=A51B16
     * C51+=A52B21, C52+=A52B22, C53+=A52B23, C54+=A52B24, C55+=A52B25, C56+=A52B26
     * C51+=A53B31, C52+=A53B32, C53+=A53B33, C54+=A53B34, C55+=A53B35, C56+=A53B36 
     * C51+=A54B41, C52+=A54B41, C53+=A54B41, C54+=A54B41, C55+=A54B41, C56+=A54B41
     * C51+=A55B51, C52+=A55B52, C53+=A55B53, C54+=A55B54, C55+=A55B55, C56+=A55B56
     * C51+=A56B61, C52+=A56B62, C53+=A56B63, C54+=A56B64, C55+=A56B65, C56+=A56B63

     */

    for rem_row in row_stripes .. a_rows {
        for k in 0 .. sub_m {
            let a_idx = get_elt(rem_row, k, m_dim) as isize;
            let a_elt = a.offset(a_idx);
            let a_elt_mult = _mm256_broadcast_sd(&*a_elt);

            for pillar in (0 .. col_pillars).step_by(MINIBLOCK) {
                let b_idx = get_elt(k, pillar, p_cols) as isize;
                let b_row_vec = _mm256_loadu_pd(b.offset(b_idx));

                let c_idx = get_elt(rem_row, pillar, p_cols) as isize;
                let mut c_row_vec = _mm256_loadu_pd(c.offset(c_idx));
                c_row_vec = _mm256_fmadd_pd(a_elt_mult, b_row_vec, c_row_vec);
                _mm256_storeu_pd(c.offset(c_idx), c_row_vec);
            }

            for col in col_pillars .. b_cols {
                let b_idx = get_elt(k, col, p_cols) as isize;
                let b_elt = b.offset(b_idx);

                let c_idx = get_elt(rem_row, col, p_cols) as isize;
                let c_elt = c.offset(c_idx);
                madd(a_elt, b_elt, c_elt);
            }
        }
    }

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
pub fn minimatrix_fmadd_f64(n_cols: usize, a: *const f64, b: *const f64, c: *mut f64) {
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
