use core::arch::x86_64::{__m256d, _mm256_broadcast_sd};
use core::arch::x86_64::{_mm256_setzero_pd, _mm256_loadu_pd,  _mm256_storeu_pd};
use core::arch::x86_64::_mm256_fmadd_pd;
use super::utilities::get_idx;

/* Microblocks should fit into the register space of amd64. */
const MICROBLOCK: usize = 4;
/* Miniblocks should fit into L1D cache (optimized for amd64) */
//(- (/ (* 32 1024) 3 8 64) (% (/ (* 32 1024) 3 8 64) 4)) 20
const MINIBLOCKROW: usize = 32;
const MINIBLOCKCOL: usize = 40;
const MINIBLOCKM: usize = MINIBLOCKCOL;
/* Blocks should fit into L2 cache (optimized for amd64) */
//(/ (* 512 1024) 3 8 240) 91
//(- (/ (* 512 1024) 3 8 240) (% (/ (* 512 1024) 3 8 240) 4)) 88
//old was 80 and 256
const BLOCKROW: usize = 88;
const BLOCKCOL: usize = 240;
const BLOCKM: usize = 80;
/* Megablocks should fit into L3 cache. 
This should probably be parameterized since it varies much by architecture. */
//(/ (* 8 1024 1024) 3 8 640) 546
//(- (/ (* 8 1024 1024) 3 8 640) (% (/ (* 8 1024 1024) 3 8 640) 32)) 544
const MEGABLOCKROW: usize = 480;
const MEGABLOCKCOL: usize = 720;
const MEGABLOCKM: usize = 480;

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

#[target_feature(enable = "avx2")]
#[cfg(any(target_arch = "x86_64"))]
unsafe fn scalar_vec_fmadd_f64u(a_elt: &f64, b_row: *const f64, c_row: *mut f64) {
    let a: __m256d = _mm256_broadcast_sd(a_elt);
    let b: __m256d = _mm256_loadu_pd(b_row);
    let mut c: __m256d = _mm256_loadu_pd(c_row as *const f64);
    
    c = _mm256_fmadd_pd(a, b, c);
    _mm256_storeu_pd(c_row, c);
}

#[target_feature(enable = "avx2")]
#[cfg(any(target_arch = "x86_64"))]
unsafe fn vectorized_range_dot_prod_add(row: *const f64, col: *const f64,
                                        c_elt: *mut f64, end: usize) {
    let mut store: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
    let mut a: __m256d;
    let mut b: __m256d;
    let mut c: __m256d = _mm256_setzero_pd();
    
    for i in (0 .. end).step_by(4) {
        a = _mm256_loadu_pd(row.offset(i as isize));
        b = _mm256_loadu_pd(col.offset(i as isize));
        c = _mm256_fmadd_pd(a, b, c);
    }

    _mm256_storeu_pd(store.as_mut_ptr(), c);
    *c_elt += store[0] + store[1] + store[2] + store[3];
}

pub fn scalar_vector_fmadd(length: usize, alpha: f64, b: *const f64, c: *mut f64) {
    let end = length - length % 4;
    for i in (0 .. end).step_by(4) {
        unsafe {
            scalar_vec_fmadd_f64u(&alpha, b.offset(i as isize), c.offset(i as isize))
        }
    }

    for i in end .. length {
        unsafe {
            madd(&alpha as *const f64, b.offset(i as isize), c.offset(i as isize))
        }
    }
}

/// Calculates the dot product and addition of the vectors A, B, and
/// C; only for use on slices whose dimensions are correct, i.e., 1xm,
/// mx1, and 1xm.
pub fn single_dot_prod_add(m_dim: usize, row: *const f64, col: *const f64, c_elt: *mut f64) {
    let final_max = m_dim;
    let first_max = final_max - final_max % 4;
    unsafe {
        vectorized_range_dot_prod_add(row, col, c_elt, first_max);

        for i in first_max .. final_max {
            madd(row.offset(i as isize), col.offset(i as isize), c_elt);
        }
    }
}

/// Calculates C = AB + C, where A is a 1xm vector, B is a mxp matrix,
/// and C is a 1xp vector. The stride is the original number of
/// columns per row of B.
pub fn vector_matrix_mul_add(stride: usize, m_dim: usize, p_cols: usize,
                             a: *const f64, b: *const f64, c: *mut f64) {
    unsafe {
        for m in 0 .. m_dim {
            let a_elt = a.offset(m as isize);
            let mut p = 0;
            while p < p_cols {
                if p + 4 < p_cols {

                    let b_idx = get_idx(m, p, stride);
                    scalar_vec_fmadd_f64u(&*a_elt,
                                          b.offset(b_idx as isize),
                                          c.offset(p as isize));
                    p += 4;
                    
                } else {
                    let b_idx = get_idx(m, p, stride);
                    madd(&*a_elt, b.offset(b_idx as isize), c.offset(p as isize));
                    p += 1;
                }
            }
        }
    }
}

/// Calculates C = AB + C, where A is a nxm matrix, B is a mx1 vector,
/// and C is a nx1 vector. m_stride is the original number of columns
/// per row of A, and p_stride is the original number of columns per
/// row of B.
pub fn matrix_vector_mul_add(m_stride: usize, p_stride: usize,
                             n_rows: usize, m_dim: usize,
                             a: *const f64, b: *const f64, c: *mut f64) {
    let mut b_tmp = Vec::with_capacity(m_dim);
    
    for row in 0 .. m_dim {
        let idx = get_idx(row, 0, p_stride);
        unsafe {
            let b_elt = b.offset(idx as isize);
            madd(a.offset(row as isize), b_elt, c.offset(idx as isize));

            if n_rows > 1 {
                b_tmp.push(b_elt);
            }
        }
    }

    for row in 1 .. n_rows {
        let a_idx = get_idx(row, 0, m_stride);
        let c_idx = get_idx(row, 0, p_stride);
        unsafe {
            let a_row = a.offset(a_idx as isize);
            let c_elt = c.offset(c_idx as isize);
            single_dot_prod_add(m_dim, a_row, b_tmp[0] as *const f64, c_elt);
        }
    }
}

/// Calculates C <= AB + C for large matrices.
pub fn matrix_mul_add(m_stride: usize,
                      p_stride: usize, 
                      n_rows: usize, m_dim: usize, p_cols: usize, 
                      a: *const f64, b: *const f64, c: *mut f64) {
    if n_rows == 1 && p_cols <= 512 {
        unsafe {
            vector_matrix_mul_add(p_stride, m_dim, p_cols, &*a, b, c)
        }
    } else if p_cols == 1 && m_dim <= 512 {
        unsafe {
            matrix_vector_mul_add(m_stride, p_stride,
                                  n_rows, m_dim, &*a, &*b, c)
        }
    } else if n_rows == MICROBLOCK && p_cols == MICROBLOCK && m_dim == MICROBLOCK {
        unsafe {
            minimatrix_fmadd_f64(m_stride, p_stride, a, b, c);
        }
    } else if n_rows <= MINIBLOCKROW && m_dim <= MINIBLOCKM && p_cols <= MINIBLOCKCOL {
        inner_matrix_mul_add(m_stride, p_stride,
                             n_rows, m_dim, p_cols,
                             a, b, c);
    } else {
        big_matrix_mul_add(m_stride, p_stride,
                           n_rows, m_dim, p_cols,
                           a, b, c)
    } 
}

fn big_matrix_mul_add(m_stride: usize, p_stride: usize,
                      n_rows: usize, m_dim: usize, p_cols: usize,
                      a: *const f64, b: *const f64, c: *mut f64) {
    let row_stripes = n_rows - n_rows % MEGABLOCKROW;
    let col_pillars = p_cols - p_cols % MEGABLOCKCOL;
    let sub_blocks = m_dim - m_dim % MEGABLOCKM;
    
    let n_rows_rem = n_rows - row_stripes;
    let m_dim_rem = m_dim - sub_blocks;
    let p_cols_rem =  p_cols - col_pillars;

    for stripe in (0 .. row_stripes).step_by(MEGABLOCKROW) {
        upper_left_subroutine(m_stride, p_stride,
                              MEGABLOCKROW, MEGABLOCKM, MEGABLOCKCOL,
                              stripe, sub_blocks, col_pillars,
                              a, b, c, &outer_matrix_mul_add);

        /* Finish adding remaining the products of A's columns by b's rows */
        block_routine(m_stride, p_stride, MEGABLOCKROW, m_dim_rem, MEGABLOCKCOL,
                      stripe, sub_blocks, col_pillars, a, b, c, &middle_matrix_mul_add);

        if p_cols_rem > 0 {
            unsafe {
                let c_idx = get_idx(stripe, col_pillars, p_stride);
                let c_chunk = c.offset(c_idx as isize);

                for block in (0 .. sub_blocks).step_by(MEGABLOCKM) {
                    col_rem_subroutine(m_stride, p_stride,
                                       MEGABLOCKROW, MEGABLOCKM, p_cols_rem,
                                       stripe, block, col_pillars,
                                       a, b, c_chunk, &middle_matrix_mul_add);
                }

                col_rem_subroutine(m_stride, p_stride,
                                   MEGABLOCKROW, m_dim_rem, p_cols_rem,
                                   stripe, sub_blocks, col_pillars,
                                   a, b, c_chunk, &middle_matrix_mul_add);
            }
        }
    }

    if p_cols_rem > 0 {
        unsafe {
            let c_idx = get_idx(row_stripes, col_pillars, p_stride);
            let c_chunk = c.offset(c_idx as isize);

            col_rem_subroutine(m_stride, p_stride,
                               n_rows_rem, m_dim, p_cols_rem,
                               row_stripes, 0, col_pillars,
                               a, b, c_chunk, &middle_matrix_mul_add);
        }
    }

    if n_rows_rem > 0  {
        block_routine(m_stride, p_stride, n_rows_rem, m_dim, MEGABLOCKCOL,
                      row_stripes, 0, col_pillars,
                      a, b, c, &middle_matrix_mul_add);
    }

}

fn outer_matrix_mul_add(m_stride: usize, p_stride: usize,
                        n_rows: usize, m_dim: usize, p_cols: usize,
                        a: *const f64, b: *const f64, c: *mut f64) {

    let row_stripes = n_rows - n_rows % BLOCKROW;
    let col_pillars = p_cols - p_cols % BLOCKCOL;
    let sub_blocks = m_dim - m_dim % BLOCKM;
    
    let n_rows_rem = n_rows - row_stripes;
    let m_dim_rem = m_dim - sub_blocks;
    let p_cols_rem =  p_cols - col_pillars;

    for stripe in (0 .. row_stripes).step_by(BLOCKROW) {
        upper_left_subroutine(m_stride, p_stride,
                              BLOCKROW, BLOCKM, BLOCKCOL,
                              stripe, sub_blocks, col_pillars,
                              a, b, c, &middle_matrix_mul_add);

        /* Finish adding remaining the products of A's columns by b's rows */
        block_routine(m_stride, p_stride, BLOCKROW, m_dim_rem, BLOCKCOL,
                      stripe, sub_blocks, col_pillars,
                      a, b, c, &inner_matrix_mul_add);

        if p_cols_rem > 0 {
            unsafe {
                let c_idx = get_idx(stripe, col_pillars, p_stride);
                let c_chunk = c.offset(c_idx as isize);

                for block in (0 .. sub_blocks).step_by(BLOCKM) {
                    col_rem_subroutine(m_stride, p_stride,
                                       BLOCKROW, BLOCKM, p_cols_rem,
                                       stripe, block, col_pillars,
                                       a, b, c_chunk, &inner_matrix_mul_add);
                }
                
                col_rem_subroutine(m_stride, p_stride,
                                   BLOCKROW, m_dim_rem, p_cols_rem,
                                   stripe, sub_blocks, col_pillars,
                                   a, b, c_chunk, &inner_matrix_mul_add);
            }
        }
    }

    if p_cols_rem > 0 {
        unsafe {
            let c_idx = get_idx(row_stripes, col_pillars, p_stride);
            let c_chunk = c.offset(c_idx as isize);

            col_rem_subroutine(m_stride, p_stride, n_rows_rem, m_dim, p_cols_rem,
                               row_stripes, 0, col_pillars, a, b, c_chunk,
                               &inner_matrix_mul_add);
        }
    }

    if n_rows_rem > 0  {
        block_routine(m_stride, p_stride, n_rows_rem, m_dim, BLOCKCOL,
                      row_stripes, 0, col_pillars,
                      a, b, c, &inner_matrix_mul_add);
    }

}

fn middle_matrix_mul_add(m_stride: usize, p_stride: usize,
                         n_rows: usize, m_dim: usize, p_cols: usize,
                         a: *const f64, b: *const f64, c: *mut f64) {
    
    let row_stripes = n_rows - n_rows % MINIBLOCKROW;
    let col_pillars = p_cols - p_cols % MINIBLOCKCOL;
    let sub_blocks = m_dim - m_dim % MINIBLOCKM;
    
    let n_rows_rem = n_rows - row_stripes;
    let m_dim_rem = m_dim - sub_blocks;
    let p_cols_rem =  p_cols - col_pillars;

    /* Calculate upper left corner of C */
    for stripe in (0 .. row_stripes).step_by(MINIBLOCKROW) {
        upper_left_subroutine(m_stride, p_stride, MINIBLOCKROW, MINIBLOCKM, MINIBLOCKCOL,
                              stripe, sub_blocks, col_pillars, a, b, c,
                              &inner_matrix_mul_add);

        /* Finish adding remaining the products of A's columns by b's rows */
        block_routine(m_stride, p_stride, MINIBLOCKROW, m_dim_rem, MINIBLOCKCOL,
                      stripe, sub_blocks, col_pillars,
                      a, b, c, &inner_matrix_mul_add);

    }

    if p_cols_rem > 0 {
        unsafe {
            for stripe in (0 .. row_stripes).step_by(MINIBLOCKROW) {
                let c_idx = get_idx(stripe, col_pillars, p_stride);
                let c_chunk = c.offset(c_idx as isize);

                for block in (0 .. sub_blocks).step_by(MINIBLOCKM) {
                    col_rem_subroutine(m_stride, p_stride, MINIBLOCKROW, MINIBLOCKM, p_cols_rem,
                                       stripe, block, col_pillars, a, b, c_chunk,
                                       &inner_matrix_mul_add);
                }

                col_rem_subroutine(m_stride, p_stride, MINIBLOCKROW, m_dim_rem, p_cols_rem,
                                   stripe, sub_blocks, col_pillars, a, b, c_chunk,
                                   &inner_matrix_mul_add);
            }

            let c_idx = get_idx(row_stripes, col_pillars, p_stride);
            let c_chunk = c.offset(c_idx as isize);

            col_rem_subroutine(m_stride, p_stride, n_rows_rem, m_dim, p_cols_rem,
                               row_stripes, 0, col_pillars,
                               a, b, c_chunk, &inner_matrix_mul_add);
        }
    }

    if n_rows_rem > 0  {
        block_routine(m_stride, p_stride, n_rows_rem, m_dim, MINIBLOCKCOL,
                      row_stripes, 0, col_pillars,
                      a, b, c, &inner_matrix_mul_add);
    }

}

fn inner_matrix_mul_add(m_stride: usize, p_stride: usize,
                        n_rows: usize, m_dim: usize, p_cols: usize,
                        a: *const f64, b: *const f64, c: *mut f64) {

    let row_stripes = n_rows - n_rows % MICROBLOCK;
    let col_pillars = p_cols - p_cols % MICROBLOCK;
    let sub_blocks = m_dim - m_dim % MICROBLOCK;
    let n_rows_rem = n_rows - row_stripes;
    let p_cols_rem = p_cols - col_pillars;
    let m_dim_rem = m_dim - sub_blocks;

    /* Take on the big left corner*/
    for stripe in (0 .. row_stripes).step_by(MICROBLOCK) {
        for block in (0 .. sub_blocks).step_by(MICROBLOCK) {
            for pillar in (0 .. col_pillars).step_by(MICROBLOCK) {
                unsafe {
                    let c_idx = get_idx(stripe, pillar, p_stride);
                    let c_chunk = c.offset(c_idx as isize);

                    let a_idx = get_idx(stripe, block, m_stride);
                    let a_chunk = a.offset(a_idx as isize);

                    let b_idx = get_idx(block, pillar, p_stride);
                    let b_chunk = b.offset(b_idx as isize);

                    minimatrix_fmadd_f64(m_stride, p_stride, a_chunk, b_chunk, c_chunk);
                }
            }
        }

        /* Finish adding remaining the products of A's columns by b's rows */
        // Pillar/k/row is a good arrangement
        for pillar in (0 .. col_pillars).step_by(MICROBLOCK) {
            for k in sub_blocks .. m_dim {
                for row in 0 .. MICROBLOCK {
                    unsafe {
                        let c_idx = get_idx(stripe + row, pillar, p_stride);
                        let c_row = c.offset(c_idx as isize);

                        let a_idx = get_idx(stripe + row, k, m_stride);
                        let b_idx = get_idx(k, pillar, p_stride);

                        let a_elt = a.offset(a_idx as isize);
                        let b_row = b.offset(b_idx as isize);
                        scalar_vec_fmadd_f64u(&*a_elt, b_row, c_row);
                    }
                }
            }
        }

        if col_pillars != p_cols {
            for block in (0 .. sub_blocks).step_by(MICROBLOCK) {
                unsafe {
                    let a_idx = get_idx(stripe, block, m_stride);
                    let a_chunk = a.offset(a_idx as isize);

                    let b_idx = get_idx(block, col_pillars, p_stride);
                    let b_chunk = b.offset(b_idx as isize);
                    
                    let c_idx = get_idx(stripe, col_pillars, p_stride);
                    let c_chunk = c.offset(c_idx as isize);

                    inner_small_matrix_mul_add(m_stride, p_stride,
                                               MICROBLOCK, MICROBLOCK, p_cols_rem,
                                               a_chunk, b_chunk, c_chunk);
                }
            }

            unsafe {
                let a_idx = get_idx(stripe, sub_blocks, m_stride);
                let a_chunk = a.offset(a_idx as isize);

                let b_idx = get_idx(sub_blocks, col_pillars, p_stride);
                let b_chunk = b.offset(b_idx as isize);

                let c_idx = get_idx(stripe, col_pillars, p_stride);
                let c_chunk = c.offset(c_idx as isize);

                inner_small_matrix_mul_add(m_stride, p_stride,
                                           MICROBLOCK, m_dim_rem, p_cols_rem,
                                           a_chunk, b_chunk, c_chunk);
            }
        }
    }

    for pillar in (0 .. col_pillars).step_by(4) {
        for row in row_stripes .. n_rows {
            for k in (0 .. sub_blocks).step_by(4) {
                unsafe {
                    let a_idx = get_idx(row, k, m_stride);
                    
                    let a1_elt = a.offset(a_idx as isize);
                    let a2_elt = a.offset((a_idx + 1) as isize);
                    let a3_elt = a.offset((a_idx + 2) as isize);
                    let a4_elt = a.offset((a_idx + 3) as isize);

                    let b11_idx = get_idx(k, pillar + 0, p_stride);
                    let b12_idx = get_idx(k, pillar + 1, p_stride);
                    let b13_idx = get_idx(k, pillar + 2, p_stride);
                    let b14_idx = get_idx(k, pillar + 3, p_stride);
                    let b21_idx = get_idx(k + 1, pillar + 0, p_stride);
                    let b22_idx = get_idx(k + 1, pillar + 1, p_stride);
                    let b23_idx = get_idx(k + 1, pillar + 2, p_stride);
                    let b24_idx = get_idx(k + 1, pillar + 3, p_stride);

                    let b11_elt = b.offset(b11_idx as isize);
                    let b12_elt = b.offset(b12_idx as isize);
                    let b13_elt = b.offset(b13_idx as isize);
                    let b14_elt = b.offset(b14_idx as isize);
                    let b21_elt = b.offset(b21_idx as isize);
                    let b22_elt = b.offset(b22_idx as isize);
                    let b23_elt = b.offset(b23_idx as isize);
                    let b24_elt = b.offset(b24_idx as isize);

                    let c11_idx = get_idx(row, pillar + 0, p_stride);
                    let c11_elt = c.offset(c11_idx as isize);
                    madd(a1_elt, b11_elt, c11_elt);
                    madd(a2_elt, b21_elt, c11_elt);

                    let c12_idx = get_idx(row, pillar + 1, p_stride);
                    let c12_elt = c.offset(c12_idx as isize);
                    madd(a1_elt, b12_elt, c12_elt);
                    madd(a2_elt, b22_elt, c12_elt);
                    
                    let c13_idx = get_idx(row, pillar + 2, p_stride);
                    let c13_elt = c.offset(c13_idx as isize);
                    madd(a1_elt, b13_elt, c13_elt);
                    madd(a2_elt, b23_elt, c13_elt);
                    
                    let c14_idx = get_idx(row, pillar + 3, p_stride);                    
                    let c14_elt = c.offset(c14_idx as isize);
                    madd(a1_elt, b14_elt, c14_elt);
                    madd(a2_elt, b24_elt, c14_elt);

                    let b31_idx = get_idx(k + 2, pillar + 0, p_stride);
                    let b32_idx = get_idx(k + 2, pillar + 1, p_stride);
                    let b33_idx = get_idx(k + 2, pillar + 2, p_stride);
                    let b34_idx = get_idx(k + 2, pillar + 3, p_stride);
                    let b41_idx = get_idx(k + 3, pillar + 0, p_stride);
                    let b42_idx = get_idx(k + 3, pillar + 1, p_stride);
                    let b43_idx = get_idx(k + 3, pillar + 2, p_stride);
                    let b44_idx = get_idx(k + 3, pillar + 3, p_stride);
                    
                    let b31_elt = b.offset(b31_idx as isize);
                    let b32_elt = b.offset(b32_idx as isize);                    
                    let b33_elt = b.offset(b33_idx as isize);
                    let b34_elt = b.offset(b34_idx as isize);
                    let b41_elt = b.offset(b41_idx as isize);
                    let b42_elt = b.offset(b42_idx as isize);
                    let b43_elt = b.offset(b43_idx as isize);
                    let b44_elt = b.offset(b44_idx as isize);

                    madd(a3_elt, b31_elt, c11_elt);
                    madd(a4_elt, b41_elt, c11_elt);
                    
                    madd(a3_elt, b32_elt, c12_elt);
                    madd(a4_elt, b42_elt, c12_elt);
                    
                    madd(a3_elt, b33_elt, c13_elt);
                    madd(a4_elt, b43_elt, c13_elt);
                    
                    madd(a3_elt, b34_elt, c14_elt);
                    madd(a4_elt, b44_elt, c14_elt);
                }
            }
        }
    }

    for pillar in (0 .. col_pillars).step_by(MICROBLOCK) {
        for k in sub_blocks .. m_dim {
            unsafe {
                let b1_idx = get_idx(k, pillar, p_stride);
                let b2_idx = get_idx(k, pillar + 1, p_stride);
                let b3_idx = get_idx(k, pillar + 2, p_stride);
                let b4_idx = get_idx(k, pillar + 3, p_stride);

                let b1_elt = b.offset(b1_idx as isize);
                let b2_elt = b.offset(b2_idx as isize);
                let b3_elt = b.offset(b3_idx as isize);
                let b4_elt = b.offset(b4_idx as isize);

                for row in row_stripes .. n_rows {
                    let c1_idx = get_idx(row, pillar, p_stride);
                    let c2_idx = get_idx(row, pillar + 1, p_stride);
                    let c3_idx = get_idx(row, pillar + 2, p_stride);
                    let c4_idx = get_idx(row, pillar + 3, p_stride);

                    let a_idx = get_idx(row, k, m_dim);
                    let a_elt = a.offset(a_idx as isize);

                    madd(a_elt, b1_elt, c.offset(c1_idx as isize));
                    madd(a_elt, b2_elt, c.offset(c2_idx as isize));
                    madd(a_elt, b3_elt, c.offset(c3_idx as isize));
                    madd(a_elt, b4_elt, c.offset(c4_idx as isize));
                }
            }
        }
    }

    /*
    for pillar in (0 .. col_pillars).step_by(MICROBLOCK) {
        for k in sub_blocks .. m_dim {
            unsafe {
                let b_idx = get_idx(k, pillar, p_stride);
                let b_row_vec = _mm256_loadu_pd(b.offset(b_idx as isize));

                for row in row_stripes .. n_rows {
                    let c_idx = get_idx(row, pillar, p_stride);
                    let mut c_row_vec = _mm256_loadu_pd(c.offset(c_idx as isize));

                    let a_idx = get_idx(row, k, m_dim);
                    let a_elt = a.offset(a_idx as isize);
                    let a_elt_mult = _mm256_broadcast_sd(&*a_elt);

                    c_row_vec = _mm256_fmadd_pd(a_elt_mult, b_row_vec, c_row_vec);
                    _mm256_storeu_pd(c.offset(c_idx as isize), c_row_vec);
                }
            }
        }
    }
     */

    /* Calculate the last rows of C up to the right columns */
    /* If n_rows_rem == 0, nothing will be done other than a function call */
    unsafe {
        let a_idx = get_idx(row_stripes, 0, m_stride);
        let a_stripe = a.offset(a_idx as isize);

        let b_idx = get_idx(0, col_pillars, p_stride);
        let b_cols = b.offset(b_idx as isize);
        
        let c_idx = get_idx(row_stripes, col_pillars, p_stride);
        let c_chunk = c.offset(c_idx as isize);

        inner_small_matrix_mul_add(m_stride, p_stride,
                                   n_rows_rem, m_dim, p_cols_rem,
                                   a_stripe, b_cols, c_chunk);
    }
}

pub fn small_matrix_mul_add(n_rows: usize, m_dim: usize, p_cols: usize,
                            a: &[f64], b: &[f64], c: &mut [f64]) {
    for k in 0 .. m_dim {
        for column in 0 .. p_cols {
            for row in 0 .. n_rows {
                let a_idx = get_idx(row, k, m_dim);
                let b_idx = get_idx(k, column, p_cols);
                let c_idx = get_idx(row, column, p_cols);
                c[c_idx].fmadd(a[a_idx], b[b_idx]);
            }
        }
    }
}

unsafe fn inner_small_matrix_mul_add(m_stride: usize, p_stride: usize,
                                     n_rows: usize, m_dim: usize, p_cols: usize,
                                     a: *const f64, b: *const f64, c: *mut f64) {
    for row in 0 .. n_rows {
        for k in 0 .. m_dim {
            let a_idx = get_idx(row, k, m_stride);
            let a_elt = a.offset(a_idx as isize);
            
            for column in 0 .. p_cols {
                let b_idx = get_idx(k, column, p_stride);
                let b_elt = b.offset(b_idx as isize);
                
                let c_idx = get_idx(row, column, p_stride);
                madd(a_elt, b_elt, c.offset(c_idx as isize));
            }
        }

    }
}

fn upper_left_subroutine(m_stride: usize, p_stride: usize,
                         subrow_step: usize, block_step: usize, subcol_step: usize,
                         stripe: usize, blocks: usize, pillars: usize,
                         a: *const f64, b: *const f64, c: *mut f64,
                         block_fn: &Fn(usize, usize, usize, usize, usize,
                                       *const f64, *const f64, *mut f64)) {
    for block in (0 .. blocks).step_by(block_step) {
        block_routine(m_stride, p_stride, subrow_step, block_step, subcol_step,
                      stripe, block, pillars, a, b, c, block_fn);
    }
}

fn block_routine(m_stride: usize, p_stride: usize,
                 subrow_step: usize, block_rem: usize, subcol_step: usize,
                 stripe: usize, block: usize, pillars: usize,
                 a: *const f64, b: *const f64, c: *mut f64,
                 block_fn: &Fn(usize, usize, usize, usize, usize,
                               *const f64, *const f64, *mut f64)) {
    unsafe {
        let a_idx = get_idx(stripe, block, m_stride);
        let a_chunk = a.offset(a_idx as isize);
        
        for pillar in (0 .. pillars).step_by(subcol_step) {
            let b_idx = get_idx(block, pillar, p_stride);
            let b_rows = b.offset(b_idx as isize);

            let c_idx = get_idx(stripe, pillar, p_stride);
            let c_chunk = c.offset(c_idx as isize);
            block_fn(m_stride, p_stride,
                     subrow_step, block_rem, subcol_step,
                     a_chunk, b_rows, c_chunk);
        }
    }
}

fn col_rem_subroutine(m_stride: usize, p_stride: usize,
                      subrow_step: usize, block_step: usize, col_rem: usize,
                      stripe: usize, block: usize, pillars: usize,
                      a: *const f64, b: *const f64, c_chunk: *mut f64,
                      block_fn: &Fn(usize, usize, usize, usize, usize,
                                    *const f64, *const f64, *mut f64)) {
    unsafe {
        let a_idx = get_idx(stripe, block, m_stride);
        let a_rows = a.offset(a_idx as isize);

        let b_idx = get_idx(block, pillars, p_stride);
        let b_cols = b.offset(b_idx as isize);

        block_fn(m_stride, p_stride,
                 subrow_step, block_step, col_rem,
                 a_rows, b_cols, c_chunk);
    }
}


/// Calculates C = AB + C for a 4x4 submatrix with AVX2 instructions.
#[target_feature(enable = "avx2")]
#[cfg(any(target_arch = "x86_64"))]
unsafe fn minimatrix_fmadd_f64(m_stride: usize, p_stride: usize,
                               a: *const f64, b: *const f64, c: *mut f64) {
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
     */

    let mut a_arr: [f64; 16] = [0.0; 16];
    let mut b_arr: [f64; 16] = [0.0; 16];

    for row in 0 .. 4 {
        for column in 0 .. 4 {
            let a_idx = get_idx(row, column, m_stride);
            a_arr[row * 4 + column] = *a.offset(a_idx as isize);
        }
    }

    for row in 0 .. 4 {
        for column in 0 .. 4 {
            let b_idx = get_idx(row, column, p_stride);
            b_arr[row * 4 + column] = *b.offset(b_idx as isize);
        }
    }

    for row in 0 .. 4 {
        let c_idx = get_idx(row, 0, p_stride);
        let c_elt = c.offset(c_idx as isize);
        let mut c_row: __m256d = _mm256_loadu_pd(c_elt as *const f64);
        for column in 0 .. 4 {
            let a_idx = get_idx(row, column, 4);
            let a_elt = &a_arr[a_idx];
            let a_mult: __m256d = _mm256_broadcast_sd(a_elt);
            //let b_elt = b_arr.as_ptr().offset((column * 4) as isize);
            let b_elt = &(b_arr[column * 4]) as *const f64;
            let b_row: __m256d = _mm256_loadu_pd(b_elt);
            c_row = _mm256_fmadd_pd(a_mult, b_row, c_row);
        }
        _mm256_storeu_pd(c_elt, c_row);
    }
}

/*
unsafe fn minimatrix_fmadd_f64(m_stride: usize, p_stride: usize,
                               a: *const f64, b: *const f64, c: *mut f64) {
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
     */

    let mut a_arr: [f64; 16] = [0.0; 16];
    let mut b_arr: [f64; 16] = [0.0; 16];
    for row in 0 .. 4 {
        for column in 0 .. 4 {
            let a_idx = get_idx(row, column, m_stride);
            a_arr[row * 4 + column] = *a.offset(a_idx as isize);
        }
    }

    for row in 0 .. 4 {
        for column in 0 .. 4 {
            let b_idx = get_idx(row, column, p_stride);
            b_arr[row * 4 + column] = *b.offset(b_idx as isize);
        }
    }

    for row in 0 .. 4 {
        let c_idx = get_idx(row, 0, p_stride);
        let mut c1 = *c.offset(c_idx as isize);
        let mut c2 = *c.offset((c_idx + 1) as isize);
        let mut c3 = *c.offset((c_idx + 2) as isize);
        let mut c4 = *c.offset((c_idx + 3) as isize);
        
        for column in 0 .. 4 {
            let b1 = b_arr[column * 4];
            let b2 = b_arr[column * 4 + 1];
            let b3 = b_arr[column * 4 + 2];
            let b4 = b_arr[column * 4 + 3];

            let a_idx = get_idx(row, column, 4);
            let a_elt = a_arr[a_idx];

            c1.fmadd(a_elt, b1);
            c2.fmadd(a_elt, b2);
            c3.fmadd(a_elt, b3);
            c4.fmadd(a_elt, b4);
        }
        
        *c.offset(c_idx as isize) = c1;
        *c.offset((c_idx + 1) as isize) = c2;
        *c.offset((c_idx + 2) as isize) = c3;
        *c.offset((c_idx + 3) as isize) = c4;
    }
}
*/
