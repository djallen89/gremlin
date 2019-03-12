use core::arch::x86_64::{__m256d, _mm256_broadcast_sd};
use core::arch::x86_64::{_mm256_setzero_pd, _mm256_loadu_pd, _mm256_storeu_pd};
use core::arch::x86_64::_mm256_fmadd_pd;
use super::utilities::{FMADD, get_idx, delimiters, remainders};

const CACHELINE: usize = 8;
/* Microblocks should fit into the register space of amd64. */
const MICROBLOCKROW: usize = 4;
const MICROBLOCKCOL: usize = 4;
const MICROBLOCKM: usize = 4;
/* Miniblocks should fit into L1D cache (optimized for amd64) */
//(/ (* 32 1024) 3 8 32) 42
//(- (/ (* 32 1024) 3 8 36) (% (/ (* 32 1024) 3 8 36) 4)) 36
//(- (* 32 1024.0) (* 8 (+ (* 32 36) (* 36 40) (* 32 40)))) 1792.0
const MINIBLOCKROW: usize = 32;
const MINIBLOCKCOL: usize = 40;
const MINIBLOCKM: usize = 36;
/* Blocks should fit into L2 cache (optimized for amd64) */
//(* 32 2) 64
//(/ (* 512 1024) 3 8 64) 341
//(sqrt (* 64 340)) 147.5127113166862
//(- (* 512 1024) (* 8 (+ (* 64 108) (* 108 340) (* 64 340)))) 1152
const BLOCKROW: usize = 64;
const BLOCKCOL: usize = 340;
const BLOCKM: usize = 108;
/* Megablocks should fit into L3 cache.
This should probably be parameterized since it varies much by architecture. */
//(* 64 6) 384
//(/ (* 8 1024 1024) 3 8 384) 910
//(- (/ (* 8 1024 1024) 3 8 384) (% (/ (* 8 1024 1024) 3 8 384) 4)) 908
//(* 32 17) 544
//(- (* 8 1024 1024.0) (* 8 (+ (* 384 540) (* 540 908) (* 384 908)))) 17792.0
const MEGABLOCKROW: usize = 384;
const MEGABLOCKCOL: usize = 908;
const MEGABLOCKM: usize = 540;

type Stride = usize;
type Dim = usize;
type ConstPtr = *const f64;
type MutPtr = *mut f64;
type BlockFn<'a> = &'a Fn(Stride, Stride, Dim, Dim, Dim, ConstPtr, ConstPtr, MutPtr);

macro_rules! row_grabber4 {
    ($arr:ident, $row:expr, $col:expr, $stride:expr) => {
        (($arr.offset(get_idx($row, $col + 0, $stride) as isize)),
         ($arr.offset(get_idx($row, $col + 1, $stride) as isize)),
         ($arr.offset(get_idx($row, $col + 2, $stride) as isize)),
         ($arr.offset(get_idx($row, $col + 3, $stride) as isize)))
    };
}

#[inline(always)]
unsafe fn madd(a: ConstPtr, b: ConstPtr, c: MutPtr) {
    let res = (*a).mul_add(*b, *c);
    *c = res;
}

#[target_feature(enable = "avx2")]
#[cfg(any(target_arch = "x86_64"))]
unsafe fn scalar_vec_fmadd_f64u(a_elt: &f64, b_row: ConstPtr, c_row: MutPtr) {
    let a: __m256d = _mm256_broadcast_sd(a_elt);
    let b: __m256d = _mm256_loadu_pd(b_row);
    let mut c: __m256d = _mm256_loadu_pd(c_row as *const f64);

    c = _mm256_fmadd_pd(a, b, c);
    _mm256_storeu_pd(c_row, c);
}

#[target_feature(enable = "avx2")]
#[cfg(any(target_arch = "x86_64"))]
unsafe fn vectorized_range_dot_prod_add(row: ConstPtr, col: ConstPtr, c_elt: MutPtr, end: usize) {
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

pub fn scalar_vector_fmadd(length: usize, alpha: f64, b: ConstPtr, c: MutPtr) {
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
pub fn single_dot_prod_add(m_dim: Dim, row: ConstPtr, col: ConstPtr, c_elt: MutPtr) {
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
pub fn vector_matrix_mul_add(stride: Stride, m_dim: Dim, p_cols: Dim,
                             a: ConstPtr, b: ConstPtr, c: MutPtr) {
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
pub fn matrix_vector_mul_add(m_stride: Stride, p_stride: Stride,
                             n_rows: Dim, m_dim: Dim,
                             a: ConstPtr, b: ConstPtr, c: MutPtr) {
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

/// Calculates C = AB + C where matrices where all dimensions are less than 4.
pub fn small_matrix_mul_add(n_rows: Dim, m_dim: Dim, p_cols: Dim,
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

/// Calculates C <= AB + C for large matrices.
pub fn matrix_mul_add(m_stride: Stride,
                      p_stride: Stride,
                      n_rows: Dim, m_dim: Dim, p_cols: Dim,
                      a: ConstPtr, b: ConstPtr, c: MutPtr) {
    if n_rows == 1 && p_cols <= 512 {
        unsafe {
            vector_matrix_mul_add(p_stride, m_dim, p_cols, &*a, b, c)
        }
    } else if p_cols == 1 && m_dim <= 512 {
        unsafe {
            matrix_vector_mul_add(m_stride, p_stride,
                                  n_rows, m_dim, &*a, &*b, c)
        }
    } else if n_rows == MICROBLOCKROW && p_cols == MICROBLOCKCOL && m_dim == MICROBLOCKM {
        unsafe {
            dgemm_microkernel(m_stride, p_stride, a, b, c);
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

fn big_matrix_mul_add(m_stride: Stride, p_stride: Stride,
                      n_rows: Dim, m_dim: Dim, p_cols: Dim,
                      a: ConstPtr, b: ConstPtr, c: MutPtr) {

    let (row_rem, col_rem, m_rem) = remainders(n_rows, p_cols, m_dim,
                                               MEGABLOCKROW, MEGABLOCKCOL, MEGABLOCKM);
    let (stripes, pillars, blocks) = delimiters(n_rows, p_cols, m_dim,
                                                row_rem, col_rem, m_rem);

    upper_kernel(m_stride, p_stride,
                 MEGABLOCKROW, MEGABLOCKM, MEGABLOCKCOL,
                 stripes, blocks, pillars,
                 m_rem, col_rem,
                 a, b, c, &outer_matrix_mul_add);


    if row_rem == 0 {
        return
    }

    unsafe {
        for block in (0 .. blocks).step_by(MEGABLOCKM) {
            let a_idx = get_idx(stripes, block, m_stride);
            let a_chunk = a.offset(a_idx as isize);

            for pillar in (0 .. pillars).step_by(MEGABLOCKCOL) {
                let c_idx = get_idx(stripes, pillar, p_stride);
                let c_chunk = c.offset(c_idx as isize);


                let b_idx = get_idx(block, pillar, p_stride);
                let b_rows = b.offset(b_idx as isize);

                outer_matrix_mul_add(m_stride, p_stride,
                                     row_rem, MEGABLOCKM, MEGABLOCKCOL,
                                     a_chunk, b_rows, c_chunk);
            }
        }

        if m_rem > 0 {
            let a_idx = get_idx(stripes, blocks, m_stride);
            let a_chunk = a.offset(a_idx as isize);

            for pillar in (0 .. pillars).step_by(MEGABLOCKCOL) {
                let c_idx = get_idx(stripes, pillar, p_stride);
                let c_chunk = c.offset(c_idx as isize);

                let b_idx = get_idx(blocks, pillar, p_stride);
                let b_rows = b.offset(b_idx as isize);

                outer_matrix_mul_add(m_stride, p_stride,
                                     row_rem, m_rem, MEGABLOCKCOL,
                                     a_chunk, b_rows, c_chunk);
            }
        }
    }

    if col_rem == 0 {
        return
    }

    unsafe {
        let c_idx = get_idx(stripes, pillars, p_stride);
        let c_chunk = c.offset(c_idx as isize);

        for block in (0 .. blocks).step_by(MEGABLOCKM) {
            let a_idx = get_idx(stripes, block, m_stride);
            let a_rows = a.offset(a_idx as isize);

            let b_idx = get_idx(block, pillars, p_stride);
            let b_cols = b.offset(b_idx as isize);

            outer_matrix_mul_add(m_stride, p_stride,
                                 row_rem, MEGABLOCKM, col_rem,
                                 a_rows, b_cols, c_chunk);

        }

        if m_rem == 0 {
            return
        }

        let c_idx = get_idx(stripes, pillars, p_stride);
        let c_chunk = c.offset(c_idx as isize);

        let a_idx = get_idx(stripes, blocks, m_stride);
        let a_rows = a.offset(a_idx as isize);

        let b_idx = get_idx(blocks, pillars, p_stride);
        let b_cols = b.offset(b_idx as isize);

        outer_matrix_mul_add(m_stride, p_stride,
                             row_rem, m_rem, col_rem,
                             a_rows, b_cols, c_chunk);
    }
}

fn outer_matrix_mul_add(m_stride: Stride, p_stride: Stride,
                        n_rows: Dim, m_dim: Dim, p_cols: Dim,
                        a: ConstPtr, b: ConstPtr, c: MutPtr) {

    let (row_rem, col_rem, m_rem) = remainders(n_rows, p_cols, m_dim,
                                               BLOCKROW, BLOCKCOL, BLOCKM);
    let (stripes, pillars, blocks) = delimiters(n_rows, p_cols, m_dim,
                                                row_rem, col_rem, m_rem);
    upper_kernel(m_stride, p_stride,
                 BLOCKROW, BLOCKM, BLOCKCOL,
                 stripes, blocks, pillars,
                 m_rem, col_rem,
                 a, b, c, &middle_matrix_mul_add);

    if row_rem == 0 {
        return
    }

    unsafe {
        for block in (0 .. blocks).step_by(BLOCKM) {
            let a_idx = get_idx(stripes, block, m_stride);
            let a_chunk = a.offset(a_idx as isize);

            for pillar in (0 .. pillars).step_by(BLOCKCOL) {
                let c_idx = get_idx(stripes, pillar, p_stride);
                let c_chunk = c.offset(c_idx as isize);


                let b_idx = get_idx(block, pillar, p_stride);
                let b_rows = b.offset(b_idx as isize);

                middle_matrix_mul_add(m_stride, p_stride,
                                      row_rem, BLOCKM, BLOCKCOL,
                                      a_chunk, b_rows, c_chunk);
            }
        }

        if m_rem > 0 {
            let a_idx = get_idx(stripes, blocks, m_stride);
            let a_chunk = a.offset(a_idx as isize);

            for pillar in (0 .. pillars).step_by(BLOCKCOL) {
                let c_idx = get_idx(stripes, pillar, p_stride);
                let c_chunk = c.offset(c_idx as isize);

                let b_idx = get_idx(blocks, pillar, p_stride);
                let b_rows = b.offset(b_idx as isize);

                middle_matrix_mul_add(m_stride, p_stride,
                                      row_rem, m_rem, BLOCKCOL,
                                      a_chunk, b_rows, c_chunk);
            }
        }
    }

    if col_rem == 0 {
        return
    }

    unsafe {
        let c_idx = get_idx(stripes, pillars, p_stride);
        let c_chunk = c.offset(c_idx as isize);

        for block in (0 .. blocks).step_by(BLOCKM) {
            let a_idx = get_idx(stripes, block, m_stride);
            let a_rows = a.offset(a_idx as isize);

            let b_idx = get_idx(block, pillars, p_stride);
            let b_cols = b.offset(b_idx as isize);

            middle_matrix_mul_add(m_stride, p_stride,
                                  row_rem, BLOCKM, col_rem,
                                  a_rows, b_cols, c_chunk);
        }

        if m_rem == 0 {
            return
        }

        let c_idx = get_idx(stripes, pillars, p_stride);
        let c_chunk = c.offset(c_idx as isize);

        let a_idx = get_idx(stripes, blocks, m_stride);
        let a_rows = a.offset(a_idx as isize);

        let b_idx = get_idx(blocks, pillars, p_stride);
        let b_cols = b.offset(b_idx as isize);

        middle_matrix_mul_add(m_stride, p_stride,
                              row_rem, m_rem, col_rem,
                              a_rows, b_cols, c_chunk);
    }
}

fn middle_matrix_mul_add(m_stride: Stride, p_stride: Stride,
                         n_rows: Dim, m_dim: Dim, p_cols: Dim,
                         a: ConstPtr, b: ConstPtr, c: MutPtr) {

    let (row_rem, col_rem, m_rem) = remainders(n_rows, p_cols, m_dim,
                                               MINIBLOCKROW, MINIBLOCKCOL, MINIBLOCKM);
    let (stripes, pillars, blocks) = delimiters(n_rows, p_cols, m_dim,
                                                row_rem, col_rem, m_rem);

    upper_kernel(m_stride, p_stride,
                 MINIBLOCKROW, MINIBLOCKM, MINIBLOCKCOL,
                 stripes, blocks, pillars,
                 m_rem, col_rem,
                 a, b, c, &inner_matrix_mul_add);

    if row_rem == 0 {
        return
    }

    unsafe {
        for block in (0 .. blocks).step_by(MINIBLOCKM) {
            let a_idx = get_idx(stripes, block, m_stride);
            let a_chunk = a.offset(a_idx as isize);

            for pillar in (0 .. pillars).step_by(MINIBLOCKCOL) {
                let c_idx = get_idx(stripes, pillar, p_stride);
                let c_chunk = c.offset(c_idx as isize);

                let b_idx = get_idx(block, pillar, p_stride);
                let b_rows = b.offset(b_idx as isize);

                inner_matrix_mul_add(m_stride, p_stride,
                                     row_rem, MINIBLOCKM, MINIBLOCKCOL,
                                     a_chunk, b_rows, c_chunk);
            }
        }

        if m_rem > 0 {
            let a_idx = get_idx(stripes, blocks, m_stride);
            let a_chunk = a.offset(a_idx as isize);

            for pillar in (0 .. pillars).step_by(MINIBLOCKCOL) {
                let c_idx = get_idx(stripes, pillar, p_stride);
                let c_chunk = c.offset(c_idx as isize);

                let b_idx = get_idx(blocks, pillar, p_stride);
                let b_rows = b.offset(b_idx as isize);

                inner_middle_matrix_mul_add(m_stride, p_stride,
                                            row_rem, m_rem, MINIBLOCKCOL,
                                            a_chunk, b_rows, c_chunk);
            }
        }

        if col_rem == 0 {
            return
        }

        let c_idx = get_idx(stripes, pillars, p_stride);
        let c_chunk = c.offset(c_idx as isize);

        for block in (0 .. blocks).step_by(MINIBLOCKM) {
            let a_idx = get_idx(stripes, block, m_stride);
            let a_rows = a.offset(a_idx as isize);

            let b_idx = get_idx(block, pillars, p_stride);
            let b_cols = b.offset(b_idx as isize);

            inner_matrix_mul_add(m_stride, p_stride,
                                 row_rem, MINIBLOCKM, col_rem,
                                 a_rows, b_cols, c_chunk);

        }

        if m_rem == 0 {
            return
        }

        let c_idx = get_idx(stripes, pillars, p_stride);
        let c_chunk = c.offset(c_idx as isize);

        let a_idx = get_idx(stripes, blocks, m_stride);
        let a_rows = a.offset(a_idx as isize);

        let b_idx = get_idx(blocks, pillars, p_stride);
        let b_cols = b.offset(b_idx as isize);

        inner_middle_matrix_mul_add(m_stride, p_stride,
                                    row_rem, m_rem, col_rem,
                                    a_rows, b_cols, c_chunk);
    }
}

fn inner_matrix_mul_add(m_stride: Stride, p_stride: Stride,
                        n_rows: Dim, m_dim: Dim, p_cols: Dim,
                        a: ConstPtr, b: ConstPtr, c: MutPtr) {

    let (row_rem, col_rem, m_rem) = remainders(n_rows, p_cols, m_dim,
                                               MICROBLOCKROW, MICROBLOCKCOL, MICROBLOCKM);
    let (stripes, pillars, blocks) = delimiters(n_rows, p_cols, m_dim,
                                                row_rem, col_rem, m_rem);
    /* Take on the big left corner*/
    for stripe in (0 .. stripes).step_by(MICROBLOCKROW) {
        let mut a_arr = [0.0; MICROBLOCKROW * MICROBLOCKM];

        for block in (0 .. blocks).step_by(MICROBLOCKM) {
            for row in 0 .. MICROBLOCKROW {
                for column in 0 .. MICROBLOCKM {
                    let a_idx = get_idx(row + stripe, column + block, m_stride);
                    unsafe {
                        a_arr[row * MICROBLOCKM + column] = *a.offset(a_idx as isize);
                    }
                }
            }

            unsafe {
                for pillar in (0 .. pillars).step_by(MICROBLOCKCOL) {
                    let b_idx = get_idx(block, pillar, p_stride);
                    let b_chunk = b.offset(b_idx as isize);

                    let c_idx = get_idx(stripe, pillar, p_stride);
                    let c_chunk = c.offset(c_idx as isize);
                    dgemm_microkernel(MICROBLOCKM, p_stride, a_arr.as_ptr(), b_chunk, c_chunk);
                }
            }
        }

        /* Finish adding remaining the products of A's columns by b's rows */
        // Pillar/k/row is a good arrangement
        for pillar in (0 .. pillars).step_by(MICROBLOCKCOL) {
            for k in blocks .. m_dim {
                for row in 0 .. MICROBLOCKROW {
                    unsafe {
                        let a_idx = get_idx(stripe + row, k, m_stride);
                        let a_elt = *a.offset(a_idx as isize);

                        let b_idx = get_idx(k, pillar, p_stride);
                        let b_row = b.offset(b_idx as isize);

                        let c_idx = get_idx(stripe + row, pillar, p_stride);
                        let c_row = c.offset(c_idx as isize);
                        scalar_vec_fmadd_f64u(&a_elt, b_row, c_row);
                    }
                }
            }
        }

        if pillars != p_cols {
            for block in (0 .. blocks).step_by(MICROBLOCKM) {
                unsafe {
                    let a_idx = get_idx(stripe, block, m_stride);
                    let a_chunk = a.offset(a_idx as isize);

                    let b_idx = get_idx(block, pillars, p_stride);
                    let b_chunk = b.offset(b_idx as isize);

                    let c_idx = get_idx(stripe, pillars, p_stride);
                    let c_chunk = c.offset(c_idx as isize);

                    inner_small_matrix_mul_add(m_stride, p_stride,
                                               MICROBLOCKROW, MICROBLOCKM, col_rem,
                                               a_chunk, b_chunk, c_chunk);
                }
            }

            unsafe {
                let a_idx = get_idx(stripe, blocks, m_stride);
                let a_chunk = a.offset(a_idx as isize);

                let b_idx = get_idx(blocks, pillars, p_stride);
                let b_chunk = b.offset(b_idx as isize);

                let c_idx = get_idx(stripe, pillars, p_stride);
                let c_chunk = c.offset(c_idx as isize);

                inner_small_matrix_mul_add(m_stride, p_stride,
                                           MICROBLOCKROW, m_rem, col_rem,
                                           a_chunk, b_chunk, c_chunk);
            }
        }
    }

    if row_rem == 0 {
        return;
    }

    let cleanup_col_rem = p_cols % CACHELINE;
    let cleanup_pillars = p_cols - cleanup_col_rem;
    for pillar in (0 .. cleanup_pillars).step_by(CACHELINE) {
        for k in (0 .. blocks).step_by(MICROBLOCKM) {
            let mut b_arr = [0.0; 32];
            for kb in 0 .. 4 {
                for col in 0 .. 8 {
                    let b_idx = get_idx(kb + k, pillar + col, p_stride);
                    unsafe {
                        b_arr[kb * 8 + col] = *b.offset(b_idx as isize);
                    }
                }
            }
            let b_ptr = b_arr.as_ptr();

            for row in stripes .. n_rows {
                unsafe {
                    let a_arr = match row_grabber4!(a, row, k, m_stride) {
                        (n, m, p, q) => [*n, *m, *p, *q]
                    };
                    column_cache_kernel(p_stride, row, pillar,
                                        &a_arr, b_ptr, c);
                }
            }
        }
    }

    for row in stripes .. n_rows {
        for k in blocks .. m_dim {
            for pillar in (0 .. cleanup_pillars).step_by(CACHELINE) {
                unsafe {
                    let b1_idx = get_idx(k, pillar + 0, p_stride);
                    let b5_idx = get_idx(k, pillar + 4, p_stride);
                    let b1 = b.offset(b1_idx as isize);
                    let b5 = b.offset(b5_idx as isize);
                    let b1_row_vec = _mm256_loadu_pd(b1);
                    let b2_row_vec = _mm256_loadu_pd(b5);

                    let c1_idx = get_idx(row, pillar + 0, p_stride);
                    let c5_idx = get_idx(row, pillar + 4, p_stride);
                    let c11 = c.offset(c1_idx as isize);
                    let c15 = c.offset(c5_idx as isize);

                    let a_idx = get_idx(row, k, m_dim);
                    let a_elt = *a.offset(a_idx as isize);
                    let a_mult = _mm256_broadcast_sd(&a_elt);

                    let mut c_row_vec = _mm256_loadu_pd(c11);
                    c_row_vec = _mm256_fmadd_pd(a_mult, b1_row_vec, c_row_vec);
                    _mm256_storeu_pd(c11, c_row_vec);

                    c_row_vec = _mm256_loadu_pd(c15);
                    c_row_vec = _mm256_fmadd_pd(a_mult, b2_row_vec, c_row_vec);
                    _mm256_storeu_pd(c15, c_row_vec);
                }
            }
        }

        for pillar in (cleanup_pillars .. pillars).step_by(MICROBLOCKCOL) {
            for k in (0 .. blocks).step_by(MICROBLOCKM) {
                unsafe {
                    let (a1, a2, a3, a4) = match row_grabber4!(a, row, k, m_stride) {
                        (n, m, p, q) => (*n, *m, *p, *q)
                    };

                    let c11_idx = get_idx(row, pillar + 0, p_stride);
                    let c11 = c.offset(c11_idx as isize);

                    let mut c_row_vec = _mm256_loadu_pd(c11);
                    let b1_idx = get_idx(k, pillar, p_stride);
                    let mut b_row_vec = _mm256_loadu_pd(b.offset(b1_idx as isize));
                    let mut a_mult = _mm256_broadcast_sd(&a1);
                    c_row_vec = _mm256_fmadd_pd(a_mult, b_row_vec, c_row_vec);

                    let b2_idx = get_idx(k + 1, pillar, p_stride);
                    a_mult = _mm256_broadcast_sd(&a2);
                    b_row_vec = _mm256_loadu_pd(b.offset(b2_idx as isize));
                    c_row_vec = _mm256_fmadd_pd(a_mult, b_row_vec, c_row_vec);

                    let b3_idx = get_idx(k + 2, pillar, p_stride);
                    a_mult = _mm256_broadcast_sd(&a3);
                    b_row_vec = _mm256_loadu_pd(b.offset(b3_idx as isize));
                    c_row_vec = _mm256_fmadd_pd(a_mult, b_row_vec, c_row_vec);

                    let b4_idx = get_idx(k + 3, pillar, p_stride);
                    a_mult = _mm256_broadcast_sd(&a4);
                    b_row_vec = _mm256_loadu_pd(b.offset(b4_idx as isize));
                    c_row_vec = _mm256_fmadd_pd(a_mult, b_row_vec, c_row_vec);
                    _mm256_storeu_pd(c11, c_row_vec);
                }
            }

            for k in blocks .. m_dim {
                unsafe {
                    let a_idx = get_idx(row, k, m_dim);
                    let a_elt = *a.offset(a_idx as isize);

                    let b1_idx = get_idx(k, pillar, p_stride);
                    let b2_idx = get_idx(k, pillar + 1, p_stride);
                    let b3_idx = get_idx(k, pillar + 2, p_stride);
                    let b4_idx = get_idx(k, pillar + 3, p_stride);

                    let b1 = *b.offset(b1_idx as isize);
                    let b2 = *b.offset(b2_idx as isize);
                    let b3 = *b.offset(b3_idx as isize);
                    let b4 = *b.offset(b4_idx as isize);

                    let c1_idx = get_idx(row, pillar + 0, p_stride);
                    let c2_idx = get_idx(row, pillar + 1, p_stride);
                    let c3_idx = get_idx(row, pillar + 2, p_stride);
                    let c4_idx = get_idx(row, pillar + 3, p_stride);

                    (*c.offset(c1_idx as isize)).fmadd(a_elt, b1);
                    (*c.offset(c2_idx as isize)).fmadd(a_elt, b2);
                    (*c.offset(c3_idx as isize)).fmadd(a_elt, b3);
                    (*c.offset(c4_idx as isize)).fmadd(a_elt, b4);
                }
            }
        }
    }

    /* Calculate the last columns of C */
    unsafe {
        let a_idx = get_idx(stripes, 0, m_stride);
        let a_stripe = a.offset(a_idx as isize);

        let b_idx = get_idx(0, pillars, p_stride);
        let b_cols = b.offset(b_idx as isize);

        let c_idx = get_idx(stripes, pillars, p_stride);
        let c_chunk = c.offset(c_idx as isize);

        inner_small_matrix_mul_add(m_stride, p_stride,
                                   row_rem, m_dim, col_rem,
                                   a_stripe, b_cols, c_chunk);

    }
}

unsafe fn inner_small_matrix_mul_add(m_stride: Stride, p_stride: Stride,
                                     n_rows: Dim, m_dim: Dim, p_cols: Dim,
                                     a: ConstPtr, b: ConstPtr, c: MutPtr) {
    let m_dim_rem = m_dim % MICROBLOCKM;
    let sub_blocks = m_dim - m_dim_rem;

    for column in 0 .. p_cols {
        for k in (0 .. sub_blocks).step_by(MICROBLOCKM) {
            let b1_idx = get_idx(k, column, p_stride);
            let b2_idx = get_idx(k + 1, column, p_stride);
            let b3_idx = get_idx(k + 2, column, p_stride);
            let b4_idx = get_idx(k + 3, column, p_stride);
            let b1 = *b.offset(b1_idx as isize);
            let b2 = *b.offset(b2_idx as isize);
            let b3 = *b.offset(b3_idx as isize);
            let b4 = *b.offset(b4_idx as isize);

            for row in 0 .. n_rows {
                let a_idx = get_idx(row, k, m_stride);
                let a1 = *a.offset(a_idx as isize);
                let a2 = *a.offset((a_idx + 1) as isize);
                let a3 = *a.offset((a_idx + 2) as isize);
                let a4 = *a.offset((a_idx + 3) as isize);

                let c_idx = get_idx(row, column, p_stride);
                let mut c_elt = *c.offset(c_idx as isize);

                c_elt.fmadd(a1, b1);
                c_elt.fmadd(a2, b2);
                c_elt.fmadd(a3, b3);
                c_elt.fmadd(a4, b4);
                *c.offset(get_idx(row, column, p_stride) as isize) = c_elt;
            }
        }

        for k in sub_blocks .. m_dim {
            let b_idx = get_idx(k, column, p_stride);
            let b_elt = b.offset(b_idx as isize);

            for row in 0 .. n_rows {
                let a_idx = get_idx(row, k, m_stride);
                let a_elt = a.offset(a_idx as isize);

                let c_idx = get_idx(row, column, p_stride);
                madd(a_elt, b_elt, c.offset(c_idx as isize));
            }
        }
    }
}

#[target_feature(enable = "avx2")]
#[cfg(any(target_arch = "x86_64"))]
unsafe fn inner_middle_matrix_mul_add(m_stride: Stride, p_stride: Stride,
                                      n_rows: Dim, m_dim: Dim, p_cols: Dim,
                                      a: ConstPtr, b: ConstPtr, c: MutPtr) {
    let m_dim_rem = m_dim % MICROBLOCKM;
    let blocks = m_dim - m_dim_rem;
    let p_cols_rem = p_cols % CACHELINE;
    let pillars = p_cols - p_cols_rem;

    for row in 0 .. n_rows {
        for k in (0 .. blocks).step_by(MICROBLOCKM) {
            let a_arr = [*a.offset(get_idx(row, k, m_stride) as isize),
                         *a.offset(get_idx(row, k + 1, m_stride) as isize),
                         *a.offset(get_idx(row, k + 2, m_stride) as isize),
                         *a.offset(get_idx(row, k + 3, m_stride) as isize)];

            for pillar in (0 .. pillars).step_by(CACHELINE) {
                let mut b_arr = [0.0; MICROBLOCKM * CACHELINE];
                for kb in 0 .. MICROBLOCKM {
                    for col in 0 .. CACHELINE {
                        let b_idx = get_idx(kb + k, pillar + col, p_stride);
                        b_arr[kb * CACHELINE + col] = *b.offset(b_idx as isize);
                    }
                }
                let b_ptr = b_arr.as_ptr();
                column_cache_kernel(p_stride, row, pillar, &a_arr, b_ptr, c);
            }

            for column in pillars .. p_cols {
                let b_idx = get_idx(k + 0, column, p_stride);
                let b_elt = *b.offset(b_idx as isize);
                let c_idx = get_idx(row, column, p_stride);
                let mut c_elt = 0.0;
                c_elt.fmadd(a_arr[0], b_elt);

                let b_idx = get_idx(k + 1, column, p_stride);
                let b_elt = *b.offset(b_idx as isize);
                c_elt.fmadd(a_arr[1], b_elt);

                let b_idx = get_idx(k + 2, column, p_stride);
                let b_elt = *b.offset(b_idx as isize);
                c_elt.fmadd(a_arr[2], b_elt);

                let b_idx = get_idx(k + 3, column, p_stride);
                let b_elt = *b.offset(b_idx as isize);
                c_elt.fmadd(a_arr[3], b_elt);

                *c.offset(c_idx as isize) += c_elt;
            }
        }

        for k in blocks .. m_dim {
            let a_elt = *a.offset(get_idx(row, k, m_stride) as isize);
            let a_mult = _mm256_broadcast_sd(&a_elt);

            for column in (0 .. pillars).step_by(MICROBLOCKCOL) {
                let b_idx = get_idx(k, column, p_stride);
                let b_elt = b.offset(b_idx as isize);
                let b_row = _mm256_loadu_pd(b_elt);

                let c_idx = get_idx(row, column, p_stride);
                let c_elt = c.offset(c_idx as isize);
                let mut c_row = _mm256_loadu_pd(c_elt);
                c_row = _mm256_fmadd_pd(a_mult, b_row, c_row);
                _mm256_storeu_pd(c_elt, c_row);
            }

            for column in pillars .. p_cols {
                let b_idx = get_idx(k, column, p_stride);
                let b_elt = b.offset(b_idx as isize);

                let c_idx = get_idx(row, column, p_stride);
                madd(&a_elt, b_elt, c.offset(c_idx as isize));
            }
        }
    }
}

#[inline(always)]
fn upper_kernel(m_stride: Stride, p_stride: Stride,
                row_block: Dim, m_block: Dim, col_block: Dim,
                stripes: Dim, blocks: Dim, pillars: Dim,
                m_rem: Dim, col_rem: Dim,
                a: ConstPtr, b: ConstPtr, c: MutPtr,
                block_fn: BlockFn) {

    for stripe in (0 .. stripes).step_by(row_block) {
        for block in (0 .. blocks).step_by(m_block) {
            block_kernel(m_stride, p_stride, row_block, m_block, col_block,
                          stripe, block, pillars, a, b, c, block_fn);
        }
        /* Finish adding remaining the products of A's columns by b's rows */
        block_kernel(m_stride, p_stride, row_block, m_rem, col_block,
                      stripe, blocks, pillars, a, b, c, &block_fn);

        if col_rem > 0 {
            unsafe {
                let c_idx = get_idx(stripe, pillars, p_stride);
                let c_chunk = c.offset(c_idx as isize);

                for block in (0 .. blocks).step_by(m_block) {
                    col_rem_kernel(m_stride, p_stride,
                                   row_block, m_block, col_rem,
                                   stripe, block, pillars,
                                   a, b, c_chunk, &block_fn);
                }

                col_rem_kernel(m_stride, p_stride,
                               row_block, m_rem, col_rem,
                               stripe, blocks, pillars,
                               a, b, c_chunk, &block_fn);
            }
        }
    }
}

fn block_kernel(m_stride: Stride, p_stride: Stride,
                subrow_step: Dim, block_rem: Dim, subcol_step: Dim,
                stripe: Dim, block: Dim, pillars: Dim,
                a: ConstPtr, b: ConstPtr, c: MutPtr,
                block_fn: BlockFn) {
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

fn col_rem_kernel(m_stride: Stride, p_stride: Stride,
                  subrow_step: usize, block_step: usize, col_rem: usize,
                  stripe: usize, block: usize, pillars: usize,
                  a: ConstPtr, b: ConstPtr, c_chunk: MutPtr,
                  block_fn: BlockFn) {
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

#[target_feature(enable = "avx2")]
#[cfg(any(target_arch = "x86_64"))]
unsafe fn column_cache_kernel(p_stride: Stride,
                              row: Dim, pillar: Dim,
                              a_arr: &[f64], b_ptr: ConstPtr, c: MutPtr) {
    let mut a_mult = _mm256_broadcast_sd(&a_arr[0]);

    let c1_idx = get_idx(row, pillar, p_stride);
    let c1_elt = c.offset(c1_idx as isize);
    let mut c_row1 = _mm256_loadu_pd(c1_elt);

    let c2_idx = get_idx(row, pillar + 4, p_stride);
    let c2_elt = c.offset(c2_idx as isize);
    let mut c_row2 = _mm256_loadu_pd(c2_elt);

    let mut b_row1 = _mm256_loadu_pd(b_ptr);
    let mut b_row2 = _mm256_loadu_pd(b_ptr.offset(4));

    c_row1 = _mm256_fmadd_pd(a_mult, b_row1, c_row1);
    c_row2 = _mm256_fmadd_pd(a_mult, b_row2, c_row2);

    a_mult = _mm256_broadcast_sd(&a_arr[1]);

    b_row1 = _mm256_loadu_pd(b_ptr.offset(8));
    c_row1 = _mm256_fmadd_pd(a_mult, b_row1, c_row1);
    
    b_row2 = _mm256_loadu_pd(b_ptr.offset(12));
    c_row2 = _mm256_fmadd_pd(a_mult, b_row2, c_row2);

    a_mult = _mm256_broadcast_sd(&a_arr[2]);
    
    b_row1 = _mm256_loadu_pd(b_ptr.offset(16));
    c_row1 = _mm256_fmadd_pd(a_mult, b_row1, c_row1);

    b_row2 = _mm256_loadu_pd(b_ptr.offset(20));
    c_row2 = _mm256_fmadd_pd(a_mult, b_row2, c_row2);

    a_mult = _mm256_broadcast_sd(&a_arr[3]);

    b_row1 = _mm256_loadu_pd(b_ptr.offset(24));
    b_row2 = _mm256_loadu_pd(b_ptr.offset(28));

    c_row1 = _mm256_fmadd_pd(a_mult, b_row1, c_row1);
    c_row2 = _mm256_fmadd_pd(a_mult, b_row2, c_row2);

    _mm256_storeu_pd(c1_elt, c_row1);
    _mm256_storeu_pd(c2_elt, c_row2);
}

/// Calculates C = AB + C for a 4x4 submatrix with AVX2 instructions.
#[target_feature(enable = "avx2")]
#[cfg(any(target_arch = "x86_64"))]
#[inline]
unsafe fn dgemm_microkernel(m_stride: Stride, p_stride: Stride,
                            a: ConstPtr, b: ConstPtr, c: MutPtr) {
    /* For 4x4 matrices, the first row of AB + C can be represented as:
     *
     * A11B11 + A12B21 + A13B31 + A14B41 + C11,
     * A11B12 + A12B22 + A13B32 + A14B42 + C12,
     * A11B13 + A12B23 + A13B33 + A14B43 + C13,
     * A11B14 + A12B24 + A13B34 + A14B44 + C14
     *
     * However, the products and summation can be reordered:
     *
     * C11 = A11B11 + C11, C12 = A11B12 + C12, C13 = A11B13 + C13, C14 = A11B14 + C14,
     * C11 = A12B21 + C11, C12 = A12B22 + C12, C13 = A12B23 + C13, C14 = A12B24 + C14,
     * C11 = A13B31 + C11, C12 = A13B32 + C12, C13 = A13B33 + C13, C14 = A13B34 + C14,
     * C11 = A14B41 + C11, C12 = A14B42 + C12, C13 = A14B43 + C13, C14 = A14B44 + C14,
     *
     * Generalizing this, one row (or 4 columns of one row) of C can be
     * calculated in 4 iterations
     * row(C, i) = A[i][j]*row(B, j) + row(C, i)
     */

    let mut a_arr = [0.0; MICROBLOCKROW * MICROBLOCKM];
    let mut b_arr = [0.0; MICROBLOCKM * MICROBLOCKCOL];

    for row in 0 .. MICROBLOCKROW {
        for column in 0 .. MICROBLOCKM {
            let a_idx = get_idx(row, column, m_stride);
            a_arr[row * MICROBLOCKM + column] = *a.offset(a_idx as isize);
        }
    }

    for row in 0 .. MICROBLOCKM {
        for column in 0 .. MICROBLOCKCOL {
            let b_idx = get_idx(row, column, p_stride);
            b_arr[row * MICROBLOCKCOL + column] = *b.offset(b_idx as isize);
        }
    }

    for row in 0 .. MICROBLOCKROW {
        let c_idx = get_idx(row, 0, p_stride);
        let c1_elt = c.offset(c_idx as isize);
        let mut c_row1: __m256d = _mm256_loadu_pd(c1_elt as *const f64);
        for column in 0 .. MICROBLOCKCOL {
            let a_idx = get_idx(row, column, MICROBLOCKM);
            let a_elt = &a_arr[a_idx];
            let a_mult: __m256d = _mm256_broadcast_sd(a_elt);

            let b_idx = get_idx(column, 0, MICROBLOCKCOL);
            let b_elt1 = &(b_arr[b_idx]) as *const f64;
            let b_row1: __m256d = _mm256_loadu_pd(b_elt1);
            c_row1 = _mm256_fmadd_pd(a_mult, b_row1, c_row1);
        }
        _mm256_storeu_pd(c1_elt, c_row1);
    }
}
