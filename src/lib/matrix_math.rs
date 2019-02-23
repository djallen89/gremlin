use core::arch::x86_64::{__m256d, _mm256_broadcast_sd};
use core::arch::x86_64::{_mm256_setzero_pd, _mm256_loadu_pd,  _mm256_storeu_pd};
use core::arch::x86_64::_mm256_fmadd_pd;
use super::utilities::get_idx;
use std::slice;

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

    let funloop = if n_rows % 8 == 0 {
        8
    } else {
        4
    };
    let row_stop = n_rows - n_rows % funloop;
    /* fewer than 4 columns */
    for column in 0 .. p_cols {
        for k in 0 .. m_dim {
            let b_idx = get_idx(k, column, p_stride);
            let b_elt = b.offset(b_idx as isize);

            for row in (0 .. row_stop).step_by(funloop) {
                let mut a_row;
                let mut c_row;
                let a_idx1 = get_idx(row + 0, k, m_stride);
                let a_elt1 = a.offset(a_idx1 as isize);
                let c_idx1 = get_idx(row + 0, column, p_stride);
                let c_elt1 = c.offset(c_idx1 as isize);
                //madd(a_elt1, b_elt, c_elt1);

                let a_idx2 = get_idx(row + 1, k, m_stride);
                let a_elt2 = a.offset(a_idx2 as isize);
                let c_idx2 = get_idx(row + 1, column, p_stride);
                let c_elt2 = c.offset(c_idx2 as isize);
                //madd(a_elt2, b_elt, c_elt2);
                
                let a_idx3 = get_idx(row + 2, k, m_stride);
                let a_elt3 = a.offset(a_idx3 as isize);
                let c_idx3 = get_idx(row + 2, column, p_stride);
                let c_elt3 = c.offset(c_idx3 as isize);
                //madd(a_elt3, b_elt, c_elt3);

                let a_idx4 = get_idx(row + 3, k, m_stride);
                let a_elt4 = a.offset(a_idx4 as isize);
                let c_idx4 = get_idx(row + 3, column, p_stride);
                let c_elt4 = c.offset(c_idx4 as isize);
                //madd(a_elt4, b_elt, c_elt4);

                a_row = [*a_elt1, *a_elt2, *a_elt3, *a_elt4];
                c_row = [*c_elt1, *c_elt2, *c_elt3, *c_elt4];
                scalar_vec_fmadd_f64u(&*b_elt, a_row.as_ptr(), c_row.as_mut_ptr());
                *c_elt1 = c_row[0];
                *c_elt2 = c_row[1];
                *c_elt3 = c_row[2];
                *c_elt4 = c_row[3];
                if funloop == 8 {
                    let a_idx5 = get_idx(row + 4, k, m_stride);
                    let a_elt5 = a.offset(a_idx5 as isize);
                    let c_idx5 = get_idx(row + 4, column, p_stride);
                    let c_elt5 = c.offset(c_idx5 as isize);
                    //madd(a_elt5, b_elt, c_elt5);

                    let a_idx6 = get_idx(row + 5, k, m_stride);
                    let a_elt6 = a.offset(a_idx6 as isize);
                    let c_idx6 = get_idx(row + 5, column, p_stride);
                    let c_elt6 = c.offset(c_idx6 as isize);
                    //madd(a_elt6, b_elt, c_elt6);

                    let a_idx7 = get_idx(row + 6, k, m_stride);
                    let a_elt7 = a.offset(a_idx7 as isize);
                    let c_idx7 = get_idx(row + 6, column, p_stride);
                    let c_elt7 = c.offset(c_idx7 as isize);
                    //madd(a_elt7, b_elt, c_elt7);
                    
                    let a_idx8 = get_idx(row + 7, k, m_stride);
                    let a_elt8 = a.offset(a_idx8 as isize);
                    let c_idx8 = get_idx(row + 7, column, p_stride);
                    let c_elt8 = c.offset(c_idx8 as isize);
                    //madd(a_elt8, b_elt, c_elt8);

                    a_row = [*a_elt5, *a_elt6, *a_elt7, *a_elt8];
                    c_row = [*c_elt5, *c_elt6, *c_elt7, *c_elt8];
                    scalar_vec_fmadd_f64u(&*b_elt, a_row.as_ptr(), c_row.as_mut_ptr());
                    *c_elt5 = c_row[0];
                    *c_elt6 = c_row[1];
                    *c_elt7 = c_row[2];
                    *c_elt8 = c_row[3];
                }
            }

            for row in row_stop .. n_rows {
                let a_idx = get_idx(row, k, m_stride);
                let a_elt = a.offset(a_idx as isize);
                let c_idx = get_idx(row, column, p_stride);
                let c_elt = c.offset(c_idx as isize);
                madd(a_elt, b_elt, c_elt);
            }

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
            return vector_matrix_mul_add(p_stride, m_dim, p_cols, &*a, b, c)
        }
    } else if p_cols == 1 && m_dim <= 512 {
        unsafe {
            return matrix_vector_mul_add(m_stride, p_stride,
                                         n_rows, m_dim, &*a, &*b, c)
        }
    } else if n_rows <= 160 && m_dim <= 160 && p_cols <= 160 {
        unsafe {
            return inner_matrix_mul_add(m_stride, p_stride,
                                        n_rows, m_dim, p_cols,
                                        a, b, c);
        }
    }

    const MEGABLOCK: usize = 480;
    let row_stripes = n_rows - n_rows % MEGABLOCK;
    let col_pillars = p_cols - p_cols % MEGABLOCK;
    let sub_blocks = m_dim - m_dim % MEGABLOCK;
    let n_rows_rem = n_rows - row_stripes;
    let m_dim_rem = m_dim - sub_blocks;
    let p_cols_rem =  p_cols - col_pillars;
    unsafe {
        for stripe in (0 .. row_stripes).step_by(MEGABLOCK) {
            for block in (0 .. sub_blocks).step_by(MEGABLOCK) {
                for pillar in (0 .. col_pillars).step_by(MEGABLOCK) {
                    let c_idx = get_idx(stripe, pillar, p_stride);
                    let c_block = c.offset(c_idx as isize);

                    let a_idx = get_idx(stripe, block, m_stride);
                    let a_block = a.offset(a_idx as isize);

                    let b_idx = get_idx(block, pillar, p_stride);
                    let b_block = b.offset(b_idx as isize);
                    outer_matrix_mul_add(m_stride, p_stride,
                                         MEGABLOCK, MEGABLOCK, MEGABLOCK,
                                         a_block, b_block, c_block);
                }
            }
        }

        if col_pillars != p_cols {
            let idx = get_idx(0, col_pillars, p_stride);
            let b_cols = b.offset(idx as isize);
            let c_cols = c.offset(idx as isize);

            outer_matrix_mul_add(m_stride, p_stride,
                                 n_rows, m_dim, p_cols_rem,
                                 a, b_cols, c_cols);
        }

        /* Finish adding remaining the products of A's columns by b's rows */
        if sub_blocks != m_dim {
            let a_idx = get_idx(0, sub_blocks, m_stride);
            let b_idx = get_idx(sub_blocks, 0, p_stride);

            let a_cols = a.offset(a_idx as isize);
            let b_rows = b.offset(b_idx as isize);
            outer_matrix_mul_add(m_stride, p_stride,
                                 row_stripes, m_dim_rem, col_pillars,
                                 a_cols, b_rows, c);
        }

        if row_stripes != n_rows {
            let a_idx = get_idx(row_stripes, 0, m_stride);
            let c_idx = get_idx(row_stripes, 0, p_stride);

            let a_rows = a.offset(a_idx as isize);
            let c_rows = c.offset(c_idx as isize);
            outer_matrix_mul_add(m_stride, p_stride,
                                 n_rows_rem, m_dim, col_pillars,
                                 a_rows, b, c_rows);
        }
    }
}
/*
// Calculates C = AB + C for a 4x4 submatrix with AVX2 instructions.
#[inline(always)]
pub fn minimatrix_fmadd_f64(m_stride: usize, p_stride: usize,
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
    
    for row in 0 .. 4 {
        let c_elt = get_idx(row, 0, p_stride);
        let c_row;
        unsafe {
            c_row = c.offset(c_elt as isize);
        }

        for column in 0 .. 4 {
            let b_elt = get_idx(column, 0, p_stride);
            let aidx = get_idx(row, column, m_stride);
            unsafe {
                let a_elt = &*a.offset(aidx as isize);
                let b_row = b.offset(b_elt as isize);
                scalar_vec_fmadd_f64u(a_elt, b_row, c_row);
            }
        }
    }
}
 */

pub fn minimatrix_fmadd_f64(m: usize, k: usize,
                            a: &[f64], b: &[f64], c: &mut [f64]) {
    use std::arch::x86_64::{__m256d, _mm256_broadcast_sd, _mm256_fmadd_pd,
                            _mm256_loadu_pd, _mm256_storeu_pd};
    use std::mem::size_of;

    const BRICK_LENGTH: usize = size_of::<__m256d>() / size_of::<f64>();
    const FUNROLL_LOOPS: usize = BRICK_LENGTH * 4;
    const BRICK_STEP: isize = size_of::<__m256d>() as isize;

    // A is m x n
    // B is n x k
    // C is m x k
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

    // row(C, i)
    for (i, c_row) in c.chunks_exact_mut(k).enumerate() {
        // row(B, j)
        for (j, b_row) in b.chunks_exact(k).enumerate() {
            let mut c_row = c_row.chunks_exact_mut(FUNROLL_LOOPS);
            let mut b_row = b_row.chunks_exact(FUNROLL_LOOPS);
            // A[i][j]
            //let a_ele = a[(i * m) + j];
            let a_ele = a[get_idx(i, j, m)];
            let a_brick = unsafe { _mm256_broadcast_sd(&a_ele) };
            // row(C, i) = A[i][j] * row(B, j) + row(C, i)
            while let (Some(c_piece), Some(b_piece)) = (c_row.next(), b_row.next()) {
                unsafe {
                    let c_bricks: *mut f64 = c_piece.as_mut_ptr();
                    let b_bricks: *const f64 = b_piece.as_ptr();

                    let b_0 = _mm256_loadu_pd(b_bricks.offset(BRICK_STEP * 0));
                    let b_1 = _mm256_loadu_pd(b_bricks.offset(BRICK_STEP * 1));
                    let b_2 = _mm256_loadu_pd(b_bricks.offset(BRICK_STEP * 2));
                    let b_3 = _mm256_loadu_pd(b_bricks.offset(BRICK_STEP * 3));

                    let c_0 = _mm256_loadu_pd(c_bricks.offset(BRICK_STEP * 0));
                    let c_1 = _mm256_loadu_pd(c_bricks.offset(BRICK_STEP * 1));
                    let c_2 = _mm256_loadu_pd(c_bricks.offset(BRICK_STEP * 2));
                    let c_3 = _mm256_loadu_pd(c_bricks.offset(BRICK_STEP * 3));          

                    let c_0 = _mm256_fmadd_pd(a_brick, b_0, c_0);
                    let c_1 = _mm256_fmadd_pd(a_brick, b_1, c_1);
                    let c_2 = _mm256_fmadd_pd(a_brick, b_2, c_2);
                    let c_3 = _mm256_fmadd_pd(a_brick, b_3, c_3);

                    _mm256_storeu_pd(c_bricks.offset(BRICK_STEP * 0), c_0);
                    _mm256_storeu_pd(c_bricks.offset(BRICK_STEP * 1), c_1);
                    _mm256_storeu_pd(c_bricks.offset(BRICK_STEP * 2), c_2);
                    _mm256_storeu_pd(c_bricks.offset(BRICK_STEP * 3), c_3);
                }
            }
            // Handle remaining ragged edges
            for (c_ele, b_ele) in c_row.into_remainder().iter_mut().zip(b_row.remainder()) {
                *c_ele = a_ele.mul_add(*b_ele, *c_ele)
            }
        }
    }
}

unsafe fn outer_matrix_mul_add(m_stride: usize, p_stride: usize,
                               n_rows: usize, m_dim: usize, p_cols: usize,
                               a: *const f64, b: *const f64, c: *mut f64) {
    const MINIBLOCK: usize = 120;
    
    let row_stripes = n_rows - n_rows % MINIBLOCK;
    let col_pillars = p_cols - p_cols % MINIBLOCK;
    let sub_blocks = m_dim - m_dim % MINIBLOCK;
    
    let n_rows_rem = n_rows - row_stripes;
    let m_dim_rem = m_dim - sub_blocks;
    let p_cols_rem =  p_cols - col_pillars;

    for stripe in (0 .. row_stripes).step_by(MINIBLOCK) {
        for block in (0 .. sub_blocks).step_by(MINIBLOCK) {
            for pillar in (0 .. col_pillars).step_by(MINIBLOCK) {
                let c_idx = get_idx(stripe, pillar, p_stride);
                let c_block = c.offset(c_idx as isize);

                let a_idx = get_idx(stripe, block, m_stride);
                let a_block = a.offset(a_idx as isize);

                let b_idx = get_idx(block, pillar, p_stride);
                let b_block = b.offset(b_idx as isize);
                inner_matrix_mul_add(m_stride, p_stride,
                                     MINIBLOCK, MINIBLOCK, MINIBLOCK,
                                     a_block, b_block, c_block);
            }
        }

        /* Finish adding remaining the products of A's columns by b's rows */
        if sub_blocks != m_dim {        
            let b_idx = get_idx(sub_blocks, 0, p_stride);
            let b_rows = b.offset(b_idx as isize);

            let a_idx = get_idx(stripe, sub_blocks, m_stride);
            let a_chunk = a.offset(a_idx as isize);

            let c_idx = get_idx(stripe, 0, p_stride);
            let c_chunk = c.offset(c_idx as isize);

            inner_matrix_mul_add(m_stride, p_stride,
                                 MINIBLOCK, m_dim_rem, col_pillars,
                                 a_chunk, b_rows, c_chunk);
        }

        if col_pillars != p_cols {
            for block in (0 .. sub_blocks).step_by(MINIBLOCK) {
                let a_idx = get_idx(stripe, block, m_stride);
                let a_chunk = a.offset(a_idx as isize);

                let b_idx = get_idx(block, col_pillars, p_stride);
                let b_chunk = b.offset(b_idx as isize);
                
                let c_idx = get_idx(stripe, col_pillars, p_stride);
                let c_chunk = c.offset(c_idx as isize);

                inner_matrix_mul_add(m_stride, p_stride,
                                     MINIBLOCK, MINIBLOCK, p_cols_rem,
                                     a_chunk, b_chunk, c_chunk);
            }

            let a_idx = get_idx(stripe, sub_blocks, m_stride);
            let a_chunk = a.offset(a_idx as isize);

            let b_idx = get_idx(sub_blocks, col_pillars, p_stride);
            let b_chunk = b.offset(b_idx as isize);

            let c_idx = get_idx(stripe, col_pillars, p_stride);
            let c_chunk = c.offset(c_idx as isize);

            inner_matrix_mul_add(m_stride, p_stride,
                                 MINIBLOCK, m_dim_rem, p_cols_rem,
                                 a_chunk, b_chunk, c_chunk);
            
        }
    }

    /* Handle the last few rows */
    if row_stripes != n_rows {
        for pillar in (0 .. col_pillars).step_by(MINIBLOCK) {
            let a_idx = get_idx(row_stripes, 0, m_stride);
            let a_rows = a.offset(a_idx as isize);

            let b_idx = get_idx(0, pillar, p_stride);
            let b_cols = b.offset(b_idx as isize);
            
            let c_idx = get_idx(row_stripes, pillar, p_stride);
            let c_rows = c.offset(c_idx as isize);
            inner_matrix_mul_add(m_stride, p_stride,
                                 n_rows_rem, m_dim, MINIBLOCK,
                                 a_rows, b_cols, c_rows);
        }
    }

    if col_pillars != p_cols {
        let a_idx = get_idx(row_stripes, 0, m_stride);
        let a_stripe = a.offset(a_idx as isize);

        let b_idx = get_idx(0, col_pillars, p_stride);
        let b_cols = b.offset(b_idx as isize);
        
        let c_idx = get_idx(row_stripes, col_pillars, p_stride);
        let c_chunk = c.offset(c_idx as isize);

        inner_matrix_mul_add(m_stride, p_stride,
                             n_rows_rem, m_dim, p_cols_rem,
                             a_stripe, b_cols, c_chunk);
    }
}

unsafe fn inner_matrix_mul_add(m_stride: usize, p_stride: usize,
                               n_rows: usize, m_dim: usize, p_cols: usize,
                               a: *const f64, b: *const f64, c: *mut f64) {
    const MINIBLOCK: usize = 4;
    let row_stripes = n_rows - n_rows % MINIBLOCK;
    let col_pillars = p_cols - p_cols % MINIBLOCK;
    let sub_blocks = m_dim - m_dim % MINIBLOCK;
    let n_rows_rem = n_rows - row_stripes;
    let p_cols_rem = p_cols - col_pillars;
    let m_dim_rem = m_dim - sub_blocks;

    /* Take on the big left corner*/
    for stripe in (0 .. row_stripes).step_by(MINIBLOCK) {
        for block in (0 .. sub_blocks).step_by(MINIBLOCK) {
            for pillar in (0 .. col_pillars).step_by(MINIBLOCK) {
                
                let a_idx = get_idx(stripe, block, m_stride);
                let b_idx = get_idx(block, pillar, p_stride);
                let c_idx = get_idx(stripe, pillar, p_stride);
                
                let a_chunk = a.offset(a_idx as isize);
                let b_chunk = b.offset(b_idx as isize);
                let c_chunk = c.offset(c_idx as isize);
                //minimatrix_fmadd_f64(m_stride, p_stride, a_chunk, b_chunk, c_chunk); 

                let a_s = slice::from_raw_parts(a_chunk, MINIBLOCK * m_stride);
                let b_s = slice::from_raw_parts(b_chunk, MINIBLOCK * p_stride);
                let c_s = slice::from_raw_parts_mut(c_chunk, MINIBLOCK * p_stride);
                minimatrix_fmadd_f64(m_stride, p_stride, a_s, b_s, c_s); 
            }
        }

        if col_pillars != p_cols {
            for block in (0 .. sub_blocks).step_by(MINIBLOCK) {
                let a_idx = get_idx(stripe, block, m_stride);
                let a_chunk = a.offset(a_idx as isize);

                let b_idx = get_idx(block, col_pillars, p_stride);
                let b_chunk = b.offset(b_idx as isize);
                
                let c_idx = get_idx(stripe, col_pillars, p_stride);
                let c_chunk = c.offset(c_idx as isize);

                inner_small_matrix_mul_add(m_stride, p_stride,
                                           MINIBLOCK, MINIBLOCK, p_cols_rem,
                                           a_chunk, b_chunk, c_chunk);
            }

            let a_idx = get_idx(stripe, sub_blocks, m_stride);
            let a_chunk = a.offset(a_idx as isize);

            let b_idx = get_idx(sub_blocks, col_pillars, p_stride);
            let b_chunk = b.offset(b_idx as isize);

            let c_idx = get_idx(stripe, col_pillars, p_stride);
            let c_chunk = c.offset(c_idx as isize);

            inner_small_matrix_mul_add(m_stride, p_stride,
                                       MINIBLOCK, m_dim_rem, p_cols_rem,
                                       a_chunk, b_chunk, c_chunk);
        }
        
        /* Finish adding remaining the products of A's columns by b's rows */
        for row in 0 .. MINIBLOCK {
            for k in sub_blocks .. m_dim {
                for pillar in (0 .. col_pillars).step_by(MINIBLOCK) {
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

    /* Calculate the right columns of C*/
    if col_pillars != p_cols {
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

    for row in row_stripes .. n_rows {
        for k in 0 .. m_dim {
            let a_idx = get_idx(row, k, m_dim);
            let a_elt = a.offset(a_idx as isize);
            let a_elt_mult = _mm256_broadcast_sd(&*a_elt);
            
            for pillar in (0 .. col_pillars).step_by(MINIBLOCK) {
                let b_idx = get_idx(k, pillar, p_stride);
                let b_row_vec = _mm256_loadu_pd(b.offset(b_idx as isize));

                let c_idx = get_idx(row, pillar, p_stride);
                let mut c_row_vec = _mm256_loadu_pd(c.offset(c_idx as isize));
                
                c_row_vec = _mm256_fmadd_pd(a_elt_mult, b_row_vec, c_row_vec);
                _mm256_storeu_pd(c.offset(c_idx as isize), c_row_vec);
            }
        }
    }
}

