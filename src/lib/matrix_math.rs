use core::arch::x86_64::{__m256d, _mm256_broadcast_sd};
use core::arch::x86_64::{_mm256_setzero_pd, _mm256_loadu_pd, _mm256_storeu_pd};
use core::arch::x86_64::_mm256_fmadd_pd;
use super::utilities::{FMADD, get_idx, delimiters, remainders, total_size};
use std::mem;

const CACHELINE: usize = 8;
/* Microblocks should fit into the register space of amd64. */
pub const MICROBLOCKROW: usize = 4;
pub const MICROBLOCKCOL: usize = 4;
pub const MICROBLOCKM: usize = 4;

/* Miniblocks should fit into L1D cache
 * (- (* 32 1024.0) (* 8 (+ (* 4 16) (* 16 200) (* 4 200)))) 256.0
 * 32KB * 1024B/KB - (8B/f64 * (4*16 + 16*200 + 4*200)) = 256 B of unused space
 * (* 100 (/ 256.0 (* 32 1024))) 0.78125
 * 0.78% of L1 cache unused */
pub const MINIBLOCKROW: usize = 4;
pub const MINIBLOCKCOL: usize = 200;
pub const MINIBLOCKM: usize = 16;
pub const L1_SIZE: usize = 32 * 1024;

/* Blocks should fit into L2 cache
 * (- (* 512 1024) (* 8 (+ (* 64 96) (* 96 368) (* 64 368)))) 4096
 *  512KB * 1024B/KB - (8B/f64 * (64*96 + 96*364 + 64*368)) = 4096
 * (* 100 (/ 4096.0 (* 512 1024))) 0.78125
 * 0.78% of L2 cache unused */
pub const BLOCKROW: usize = 64;
pub const BLOCKCOL: usize = 368;
pub const BLOCKM:   usize = 96;
pub const L2_SIZE: usize = 512 * 1024;

/* Megablocks should fit into L3 cache.
 * This should probably be parameterized.
 * (/ (* 8 1024 1024) 3 8 384) 910
 * (- (* 8 1024 1024.0) (* 8 (+ (* 384 548) (* 548 896) (* 384 896)))) 24576.0 */
pub const MEGABLOCKROW: usize = 160;
pub const MEGABLOCKCOL: usize = 1536;
pub const MEGABLOCKM: usize = 336;
pub const L3_SIZE: usize = 8 * 1024 * 1024;

type Stride = usize;
type Dim = usize;
type ConstPtr = *const f64;
type MutPtr = *mut f64;
type BlockFn<'a> = &'a Fn(Stride, Stride, Dim, Dim, Dim, ConstPtr, ConstPtr, MutPtr);

pub trait Chunk {
    fn get_chunk(&self, row: usize, column: usize, stride: Stride) -> Self;
}

impl Chunk for ConstPtr {
    fn get_chunk(&self, row: usize, column: usize, stride: Stride) -> ConstPtr {
        let idx = get_idx(row, column, stride);
        unsafe {
            self.offset(idx as isize)
        }
    }
}

impl Chunk for MutPtr {
    fn get_chunk(&self, row: usize, column: usize, stride: Stride) -> MutPtr {
        let idx = get_idx(row, column, stride);
        unsafe {
            self.offset(idx as isize)
        }
    }
}

macro_rules! row_grabber4 {
    ($arr:ident, $row:expr, $col:expr, $stride:expr) => {
        (($arr.get_chunk($row, $col + 0, $stride)),
         ($arr.get_chunk($row, $col + 1, $stride)),
         ($arr.get_chunk($row, $col + 2, $stride)),
         ($arr.get_chunk($row, $col + 3, $stride)))
    };
}

macro_rules! column_row_fmadd {
    ($b_idx: expr, $a_elt:expr, $b_ptr:ident,
     $a_mult:ident, $b_row:ident, $c_row:ident) => {
        $a_mult = _mm256_broadcast_sd(&$a_elt);
        $b_row = _mm256_loadu_pd($b_ptr.offset($b_idx as isize));
        $c_row = _mm256_fmadd_pd($a_mult, $b_row, $c_row);
    };
}

#[inline]
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
        let a_row = a.get_chunk(row, 0, m_stride);
        let c_elt = c.get_chunk(row, 0, p_stride);
        single_dot_prod_add(m_dim, a_row, b_tmp[0] as *const f64, c_elt);
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
    let memory = total_size(mem::size_of::<f64>(), n_rows, m_dim, p_cols);
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
    } else if memory <= L2_SIZE {
        middle_matrix_mul_add(m_stride, p_stride,
                           n_rows, m_dim, p_cols,
                           a, b, c)
    } else if memory <= L3_SIZE {
        outer_matrix_mul_add(m_stride, p_stride,
                             n_rows, m_dim, p_cols,
                             a, b, c)
    } else {
        big_matrix_mul_add(m_stride, p_stride,
                           n_rows, m_dim, p_cols,
                           a, b, c)
    }
}

fn big_matrix_mul_add(m_stride: Stride, p_stride: Stride,
                      n_rows: Dim, m_dim: Dim, p_cols: Dim,
                      a: ConstPtr, b: ConstPtr, c: MutPtr) {
    kernel(m_stride, p_stride,
           n_rows, m_dim, p_cols,
           a, b, c,
           MEGABLOCKROW, MEGABLOCKM, MEGABLOCKCOL,
           &outer_matrix_mul_add);
}

fn outer_matrix_mul_add(m_stride: Stride, p_stride: Stride,
                        n_rows: Dim, m_dim: Dim, p_cols: Dim,
                        a: ConstPtr, b: ConstPtr, c: MutPtr) {
    kernel(m_stride, p_stride,
           n_rows, m_dim, p_cols,
           a, b, c,
           BLOCKROW, BLOCKM, BLOCKCOL,
           &middle_matrix_mul_add);
}

fn middle_matrix_mul_add(m_stride: Stride, p_stride: Stride,
                         n_rows: Dim, m_dim: Dim, p_cols: Dim,
                         a: ConstPtr, b: ConstPtr, c: MutPtr) {

    let (row_rem, col_rem, m_rem) = remainders(n_rows, p_cols, m_dim,
                                               MINIBLOCKROW, MINIBLOCKCOL, MINIBLOCKM);
    let (stripes, pillars, blocks) = delimiters(n_rows, p_cols, m_dim,
                                                row_rem, col_rem, m_rem);

    for stripe in (0 .. stripes).step_by(MINIBLOCKROW) {
        for block in (0 .. blocks).step_by(MINIBLOCKM) {
            block_kernel(m_stride, p_stride, MINIBLOCKROW, MINIBLOCKM, MINIBLOCKCOL,
                         stripe, block, pillars, a, b, c, &inner_matrix_mul_add);
        }
    }

    for stripe in (0 .. stripes).step_by(MINIBLOCKROW) {
        /* Finish adding remaining the products of A's columns by b's rows */
        block_kernel(m_stride, p_stride, MINIBLOCKROW, m_rem, MINIBLOCKCOL,
                     stripe, blocks, pillars, a, b, c, &inner_matrix_mul_add);

        if col_rem > 0 {
            let c_chunk = c.get_chunk(stripe, pillars, p_stride);
            for block in (0 .. blocks).step_by(MINIBLOCKM) {
                col_rem_kernel(m_stride, p_stride,
                               MINIBLOCKROW, MINIBLOCKM, col_rem,
                               stripe, block, pillars,
                               a, b, c_chunk, &inner_matrix_mul_add);
            }

            col_rem_kernel(m_stride, p_stride,
                           MINIBLOCKROW, m_rem, col_rem,
                           stripe, blocks, pillars,
                           a, b, c_chunk, &inner_matrix_mul_add);
        }
    }

    if row_rem == 0 {
        return
    }

    unsafe {
        for block in (0 .. blocks).step_by(MINIBLOCKM) {
            let a_chunk = a.get_chunk(stripes, block, m_stride);

            for pillar in (0 .. pillars).step_by(MINIBLOCKCOL) {
                let b_rows = b.get_chunk(block, pillar, p_stride);
                let c_chunk = c.get_chunk(stripes, pillar, p_stride);

                inner_matrix_mul_add(m_stride, p_stride,
                                     row_rem, MINIBLOCKM, MINIBLOCKCOL,
                                     a_chunk, b_rows, c_chunk);
            }
        }

        if m_rem > 0 {
            let a_chunk = a.get_chunk(stripes, blocks, m_stride);
            for pillar in (0 .. pillars).step_by(MINIBLOCKCOL) {                
                let b_rows = b.get_chunk(blocks, pillar, p_stride);
                let c_chunk = c.get_chunk(stripes, pillar, p_stride);

                inner_middle_matrix_mul_add(m_stride, p_stride,
                                            row_rem, m_rem, MINIBLOCKCOL,
                                            a_chunk, b_rows, c_chunk);
            }
        }

        if col_rem == 0 {
            return
        }

        let c_chunk = c.get_chunk(stripes, pillars, p_stride);

        for block in (0 .. blocks).step_by(MINIBLOCKM) {
            let a_rows = a.get_chunk(stripes, block, m_stride);
            let b_cols = b.get_chunk(block, pillars, p_stride);

            inner_matrix_mul_add(m_stride, p_stride,
                                 row_rem, MINIBLOCKM, col_rem,
                                 a_rows, b_cols, c_chunk);

        }

        if m_rem == 0 {
            return
        }

        let a_rows = a.get_chunk(stripes, blocks, m_stride);
        let b_cols = b.get_chunk(blocks, pillars, p_stride);
        let c_chunk = c.get_chunk(stripes, pillars, p_stride);

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
    let cleanup_col_rem = p_cols % CACHELINE;
    let cleanup_pillars = p_cols - cleanup_col_rem;

    /* Take on the big left corner*/
    let mut a_arr = [0.0; MICROBLOCKROW * MICROBLOCKM];
    let a_ptr = a_arr.as_ptr();
    for _stripe in (0 .. stripes).step_by(MICROBLOCKROW) {
        for block in (0 .. blocks).step_by(MICROBLOCKM) {
            unsafe {
                for row in 0 .. MICROBLOCKROW {
                    for column in 0 .. MICROBLOCKM {
                        a_arr[row*MICROBLOCKM+column] = *a.get_chunk(row, column+block, m_stride);
                    }
                }

                for pillar in (0 .. pillars).step_by(MICROBLOCKCOL) {
                    let b_chunk = b.get_chunk(block, pillar, p_stride);
                    let c_chunk = c.offset(pillar as isize);
                    dgemm_microkernel(MICROBLOCKM, p_stride, a_ptr, b_chunk, c_chunk);
                }
            }
        }
    }

    for row in 0 .. stripes {
        /* Finish adding remaining the products of A's columns by b's rows */
        unsafe {
            for k in blocks .. m_dim {
                let a_elt = *a.get_chunk(row, k, m_stride);
                let a_mult = _mm256_broadcast_sd(&a_elt);
                
                for pillar in (0 .. cleanup_pillars).step_by(CACHELINE) {
                    let b1_row = b.get_chunk(k, pillar, p_stride);
                    let b1_row_vec = _mm256_loadu_pd(b1_row);

                    let b2_row = b.get_chunk(k, pillar + 4, p_stride);
                    let b2_row_vec = _mm256_loadu_pd(b2_row);

                    let c1_row = c.get_chunk(row, pillar, p_stride);
                    let mut c1_row_vec = _mm256_loadu_pd(c1_row as *const f64);

                    c1_row_vec = _mm256_fmadd_pd(a_mult, b1_row_vec, c1_row_vec);
                    _mm256_storeu_pd(c1_row, c1_row_vec);

                    let c2_row = c.get_chunk(row, pillar + 4, p_stride);
                    let mut c2_row_vec = _mm256_loadu_pd(c2_row as *const f64);

                    c2_row_vec = _mm256_fmadd_pd(a_mult, b2_row_vec, c2_row_vec);
                    _mm256_storeu_pd(c2_row, c2_row_vec);
                }

                for pillar in (cleanup_pillars .. pillars).step_by(MICROBLOCKCOL) {
                    let b_row = b.get_chunk(k, pillar, p_stride);
                    let b_row_vec = _mm256_loadu_pd(b_row);

                    let c_row = c.get_chunk(row, pillar, p_stride);
                    let mut c_row_vec: __m256d = _mm256_loadu_pd(c_row as *const f64);

                    c_row_vec = _mm256_fmadd_pd(a_mult, b_row_vec, c_row_vec);
                    _mm256_storeu_pd(c_row, c_row_vec);
                }
            }
        }
    }

    if pillars != p_cols {
        for block in (0 .. blocks).step_by(MICROBLOCKM) {
            for stripe in (0 .. stripes).step_by(MICROBLOCKROW) {
                unsafe {
                    let a_chunk = a.get_chunk(stripe, block, m_stride);
                    let b_chunk = b.get_chunk(block, pillars, p_stride);
                    let c_chunk = c.get_chunk(stripe, pillars, p_stride);

                    foo_calculator(m_stride, p_stride, MICROBLOCKROW, MICROBLOCKM, col_rem,             
                                   a_chunk, b_chunk, c_chunk);
                }
            }
        }
        
        for stripe in (0 .. stripes).step_by(MICROBLOCKROW) {
            unsafe {
                let a_chunk = a.get_chunk(stripe, blocks, m_stride);
                let b_chunk = b.get_chunk(blocks, pillars, p_stride);
                let c_chunk = c.get_chunk(stripe, pillars, p_stride);

                inner_small_matrix_mul_add(m_stride, p_stride, n_rows, m_rem, col_rem,
                                           a_chunk, b_chunk, c_chunk);
                
            }
        }
    }

    
    if row_rem == 0 {
        return;
    }

    unsafe {
        for k in (0 .. blocks).step_by(MICROBLOCKM) {
            for pillar in (0 .. cleanup_pillars).step_by(CACHELINE) {
                let mut b_arr = [0.0; 32];
                for kb in 0 .. 4 {
                    for col in 0 .. 8 {
                        b_arr[kb * 8 + col] = *b.get_chunk(kb + k, pillar + col, p_stride);
                    }
                }

                let b_ptr = b_arr.as_ptr();

                for row in stripes .. n_rows {
                    let a_arr = match row_grabber4!(a, row, k, m_stride) {
                        (n, m, p, q) => [*n, *m, *p, *q]
                    };

                    column_cache_kernel(p_stride, row, pillar,
                                        &a_arr, b_ptr, c);
                }
            }

            for pillar in (cleanup_pillars .. pillars).step_by(MICROBLOCKCOL) {
                for row in stripes .. n_rows {
                    let (a1, a2, a3, a4) = match row_grabber4!(a, row, k, m_stride) {
                        (n, m, p, q) => (*n, *m, *p, *q)
                    };

                    let c11 = c.get_chunk(row, pillar + 0, p_stride);
                    let mut c_row_vec = _mm256_loadu_pd(c11);
                    
                    let mut a_mult = _mm256_broadcast_sd(&a1);
                    let b1 = b.get_chunk(k, pillar, p_stride);
                    let mut b_row_vec = _mm256_loadu_pd(b1);
                    c_row_vec = _mm256_fmadd_pd(a_mult, b_row_vec, c_row_vec);

                    let b2_idx = get_idx(k + 1, pillar, p_stride);
                    column_row_fmadd!(b2_idx, a2, b, a_mult, b_row_vec, c_row_vec);

                    let b3_idx = get_idx(k + 2, pillar, p_stride);
                    column_row_fmadd!(b3_idx, a3, b, a_mult, b_row_vec, c_row_vec);
                    
                    let b4_idx = get_idx(k + 3, pillar, p_stride);
                    column_row_fmadd!(b4_idx, a4, b, a_mult, b_row_vec, c_row_vec);
                    
                    _mm256_storeu_pd(c11, c_row_vec);
                }
            }
        }

        for k in blocks .. m_dim {
            for pillar in (0 .. cleanup_pillars).step_by(CACHELINE) {
                let b1 = b.get_chunk(k, pillar, p_stride);
                let b5 = b.get_chunk(k, pillar + 4, p_stride);
                let b1_row_vec = _mm256_loadu_pd(b1);
                let b2_row_vec = _mm256_loadu_pd(b5);

                for row in stripes .. n_rows {
                    let a_elt = *a.get_chunk(row, k, m_dim);
                    let a_mult = _mm256_broadcast_sd(&a_elt);

                    let c11 = c.get_chunk(row, pillar, p_stride);
                    let c15 = c.get_chunk(row, pillar + 4, p_stride);

                    let mut c_row_vec = _mm256_loadu_pd(c11);
                    c_row_vec = _mm256_fmadd_pd(a_mult, b1_row_vec, c_row_vec);
                    _mm256_storeu_pd(c11, c_row_vec);

                    c_row_vec = _mm256_loadu_pd(c15);
                    c_row_vec = _mm256_fmadd_pd(a_mult, b2_row_vec, c_row_vec);
                    _mm256_storeu_pd(c15, c_row_vec);
                }
            }

            for pillar in (cleanup_pillars .. pillars).step_by(MICROBLOCKCOL) {
                let b1 = b.get_chunk(k, pillar, p_stride);
                let b_row = _mm256_loadu_pd(b1);

                for row in stripes .. n_rows {
                    let a_elt = *a.get_chunk(row, k, m_dim);
                    let a_mult = _mm256_broadcast_sd(&a_elt);
                    
                    let c_elt = c.get_chunk(row, pillar + 0, p_stride);
                    let mut c_row = _mm256_loadu_pd(c_elt);
                    c_row = _mm256_fmadd_pd(a_mult, b_row, c_row);
                    _mm256_storeu_pd(c_elt, c_row);
                }
            }
        }
    }

    /* Calculate the last columns of C */
    if col_rem > 0 {
        /*
        unsafe {
            for block in (0 .. blocks).step_by(MICROBLOCKM) {
                let a_stripe = a.get_chunk(stripes, block, m_stride);
                let b_cols = b.get_chunk(block, pillars, p_stride);
                let c_chunk = c.get_chunk(stripes, pillars, p_stride);
                
                foo_calculator(m_stride, p_stride,
                               row_rem, MICROBLOCKM, col_rem,
                               a_stripe, b_cols, c_chunk);
            }

            let a_stripe = a.get_chunk(stripes, blocks, m_stride);
            let b_cols = b.get_chunk(blocks, pillars, p_stride);
            let c_chunk = c.get_chunk(stripes, pillars, p_stride);
            
            foo_calculator(m_stride, p_stride,
                           row_rem, m_rem, col_rem,
                           a_stripe, b_cols, c_chunk);
        }
         */
        unsafe {
            let a_stripe = a.get_chunk(stripes, 0, m_stride);
            let b_cols = b.get_chunk(0, pillars, p_stride);
            let c_chunk = c.get_chunk(stripes, pillars, p_stride);
            inner_small_matrix_mul_add(m_stride, p_stride,
                                       row_rem, m_dim, col_rem,
                                       a_stripe, b_cols, c_chunk);
        }
    }
}

#[inline]
unsafe fn inner_small_matrix_mul_add(m_stride: Stride, p_stride: Stride,
                                     n_rows: Dim, m_dim: Dim, p_cols: Dim,
                                     a: ConstPtr, b: ConstPtr, c: MutPtr) {
    for row in 0 .. n_rows {
        for column in 0 .. p_cols {
            let c_idx = get_idx(row, column, p_stride);
            let mut c_elt = *c.offset(c_idx as isize);
            for k in 0 .. m_dim {
                let b_elt = *b.get_chunk(k, column, p_stride);
                let a_elt = *a.get_chunk(row, k, m_stride);
                c_elt.fmadd(a_elt, b_elt);
            }
            *c.offset(c_idx as isize) = c_elt;
        }
    }
}

#[target_feature(enable = "avx2")]
#[cfg(any(target_arch = "x86_64"))]
// Assume 4x4 block from A, 4x3 block from B, and 4x3 block from C
unsafe fn foo_calculator(m_stride: Stride, p_stride: Stride,
                         n_rows: usize, m_dim: usize, p_cols: usize,
                         a: ConstPtr, b: ConstPtr, c: MutPtr) {

    let mut a_arr = [0.0; MICROBLOCKROW * MICROBLOCKM];
    let mut c_arr = [0.0; MICROBLOCKROW * MICROBLOCKCOL];
    let c_ptr = c_arr.as_mut_ptr();

    for row in 0 .. n_rows {
        for col in 0 .. m_dim {
            a_arr[row * MICROBLOCKM + col] = *a.get_chunk(row, col, m_stride);
        }
    }

    for row in 0 .. n_rows {
        let c_elt = c_ptr.offset((row * MICROBLOCKCOL) as isize);
        let mut c_row = _mm256_loadu_pd(c_elt);
        for k in 0 .. m_dim {
            let a_mult = _mm256_broadcast_sd(&a_arr[k + row * MICROBLOCKM]);
            let b_row = _mm256_loadu_pd(b.get_chunk(k, 0, p_stride));
            c_row = _mm256_fmadd_pd(a_mult, b_row, c_row);
        }
        _mm256_storeu_pd(c_elt, c_row);
    }

    for row in 0 .. n_rows {
        for col in 0 .. p_cols {
            *c.get_chunk(row, col, p_stride) += c_arr[row * MICROBLOCKCOL + col];
        }
    }
}

#[target_feature(enable = "avx2")]
#[cfg(any(target_arch = "x86_64"))]
#[inline]
unsafe fn inner_middle_matrix_mul_add(m_stride: Stride, p_stride: Stride,
                                      n_rows: Dim, m_dim: Dim, p_cols: Dim,
                                      a: ConstPtr, b: ConstPtr, c: MutPtr) {
    let m_dim_rem = m_dim % MICROBLOCKM;
    let blocks = m_dim - m_dim_rem;
    let p_cols_rem = p_cols % CACHELINE;
    let pillars = p_cols - p_cols_rem;
    
    for row in 0 .. n_rows {
        for k in (0 .. blocks).step_by(MICROBLOCKM) {
            let a_arr = [*a.get_chunk(row, k, m_stride),
                         *a.get_chunk(row, k + 1, m_stride),
                         *a.get_chunk(row, k + 2, m_stride),
                         *a.get_chunk(row, k + 3, m_stride)];

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
                let b_elt = *b.get_chunk(k, column, p_stride);
                let c_idx = get_idx(row, column, p_stride);
                let mut c_elt = 0.0;
                c_elt.fmadd(a_arr[0], b_elt);

                let b_elt = *b.get_chunk(k + 1, column, p_stride);
                c_elt.fmadd(a_arr[1], b_elt);

                let b_elt = *b.get_chunk(k + 2, column, p_stride);
                c_elt.fmadd(a_arr[2], b_elt);

                let b_elt = *b.get_chunk(k + 3, column, p_stride);
                c_elt.fmadd(a_arr[3], b_elt);

                *c.offset(c_idx as isize) += c_elt;
            }
        }

        for k in blocks .. m_dim {
            let a_elt = *a.offset(get_idx(row, k, m_stride) as isize);
            let a_mult = _mm256_broadcast_sd(&a_elt);

            for column in (0 .. pillars).step_by(CACHELINE) {
                let b_elt = b.get_chunk(k, column, p_stride);
                let b_row1 = _mm256_loadu_pd(b_elt);
                let b_row2 = _mm256_loadu_pd(b_elt.offset(4));

                let c_elt = c.get_chunk(row, column, p_stride);
                let mut c_row1 = _mm256_loadu_pd(c_elt);
                let mut c_row2 = _mm256_loadu_pd(c_elt.offset(4));
                c_row1 = _mm256_fmadd_pd(a_mult, b_row1, c_row1);
                c_row2 = _mm256_fmadd_pd(a_mult, b_row2, c_row2);
                _mm256_storeu_pd(c_elt, c_row1);
                _mm256_storeu_pd(c_elt.offset(4), c_row2);
            }

            for column in pillars .. p_cols {
                let b_elt = *b.get_chunk(k, column, p_stride);
                let c_elt = c.get_chunk(row, column, p_stride);
                (*c_elt).fmadd(a_elt, b_elt);
            }
        }
    }
}

fn kernel(m_stride: Stride, p_stride: Stride,
          n_rows: Dim, m_dim: Dim, p_cols: Dim,
          a: ConstPtr, b: ConstPtr, c: MutPtr,
          row_block: Dim, m_block: Dim, col_block: Dim,
          block_fn: BlockFn) {

    let (row_rem, col_rem, m_rem) = remainders(n_rows, p_cols, m_dim,
                                               row_block, col_block, m_block);
    let (stripes, pillars, blocks) = delimiters(n_rows, p_cols, m_dim,
                                                row_rem, col_rem, m_rem);

    for stripe in (0 .. stripes).step_by(row_block) {
        /* Do the products in the upper left corner */
        for block in (0 .. blocks).step_by(m_block) {
            block_kernel(m_stride, p_stride, row_block, m_block, col_block,
                         stripe, block, pillars, a, b, c, block_fn);
        }
        /* Finish adding remaining the products of A's columns by b's rows */
        block_kernel(m_stride, p_stride, row_block, m_rem, col_block,
                     stripe, blocks, pillars, a, b, c, &block_fn);


        /* Add columns to the left */
        if col_rem > 0 {
            let c_chunk = c.get_chunk(stripe, pillars, p_stride);

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

    if row_rem == 0 {
        return
    }

    for block in (0 .. blocks).step_by(m_block) {
        let a_chunk = a.get_chunk(stripes, block, m_stride);

        for pillar in (0 .. pillars).step_by(col_block) {
            let b_rows = b.get_chunk(block, pillar, p_stride);
            let c_chunk = c.get_chunk(stripes, pillar, p_stride);

            block_fn(m_stride, p_stride,
                     row_rem, m_block, col_block,
                     a_chunk, b_rows, c_chunk);
        }
    }

    if m_rem > 0 {
        let a_chunk = a.get_chunk(stripes, blocks, m_stride);

        for pillar in (0 .. pillars).step_by(col_block) {
            let b_rows = b.get_chunk(blocks, pillar, p_stride);
            let c_chunk = c.get_chunk(stripes, pillar, p_stride);

            block_fn(m_stride, p_stride,
                     row_rem, m_rem, col_block,
                     a_chunk, b_rows, c_chunk);
        }
    }

    if col_rem == 0 {
        return
    }

    let c_chunk = c.get_chunk(stripes, pillars, p_stride);

    for block in (0 .. blocks).step_by(m_block) {
        let a_rows = a.get_chunk(stripes, block, m_stride);
        let b_cols = b.get_chunk(block, pillars, p_stride);

        block_fn(m_stride, p_stride,
                 row_rem, m_block, col_rem,
                 a_rows, b_cols, c_chunk);
    }

    if m_rem == 0 {
        return
    }

    let a_rows = a.get_chunk(stripes, blocks, m_stride);
    let b_cols = b.get_chunk(blocks, pillars, p_stride);
    
    block_fn(m_stride, p_stride,
             row_rem, m_rem, col_rem,
             a_rows, b_cols, c_chunk);
}

#[inline]
fn block_kernel(m_stride: Stride, p_stride: Stride,
                subrow_step: Dim, block_rem: Dim, subcol_step: Dim,
                stripe: Dim, block: Dim, pillars: Dim,
                a: ConstPtr, b: ConstPtr, c: MutPtr,
                block_fn: BlockFn) {

    let a_chunk = a.get_chunk(stripe, block, m_stride);

    for pillar in (0 .. pillars).step_by(subcol_step) {
        let b_rows = b.get_chunk(block, pillar, p_stride);
        let c_chunk = c.get_chunk(stripe, pillar, p_stride);

        block_fn(m_stride, p_stride,
                 subrow_step, block_rem, subcol_step,
                 a_chunk, b_rows, c_chunk);
    }
}

#[inline(always)]
fn col_rem_kernel(m_stride: Stride, p_stride: Stride,
                  subrow_step: usize, block_step: usize, col_rem: usize,
                  stripe: usize, block: usize, pillars: usize,
                  a: ConstPtr, b: ConstPtr, c_chunk: MutPtr,
                  block_fn: BlockFn) {
    let a_rows = a.get_chunk(stripe, block, m_stride);
    let b_cols = b.get_chunk(block, pillars, p_stride);

    block_fn(m_stride, p_stride,
             subrow_step, block_step, col_rem,
             a_rows, b_cols, c_chunk);
}

#[target_feature(enable = "avx2")]
#[cfg(any(target_arch = "x86_64"))]
#[inline]
unsafe fn column_cache_kernel(p_stride: Stride,
                              row: Dim, pillar: Dim,
                              a_arr: &[f64], b_ptr: ConstPtr, c: MutPtr) {

    let c1_elt = c.get_chunk(row, pillar, p_stride);
    let mut c_row1 = _mm256_loadu_pd(c1_elt);

    let c2_elt = c.get_chunk(row, pillar + 4, p_stride);
    let mut c_row2 = _mm256_loadu_pd(c2_elt);

    for k in 0 .. 4 {
        let a_mult = _mm256_broadcast_sd(&a_arr[k]);
        let b_row1 = _mm256_loadu_pd(b_ptr.offset((k * 8) as isize));
        let b_row2 = _mm256_loadu_pd(b_ptr.offset((k * 8 + 4) as isize));
        c_row1 = _mm256_fmadd_pd(a_mult, b_row1, c_row1);
        c_row2 = _mm256_fmadd_pd(a_mult, b_row2, c_row2);
    }

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

    for row in 0 .. MICROBLOCKM {
        let b_idx = row * p_stride;
        b_arr[row * MICROBLOCKCOL + 0] = *b.offset(b_idx as isize);
        b_arr[row * MICROBLOCKCOL + 1] = *b.offset((b_idx + 1) as isize);
        b_arr[row * MICROBLOCKCOL + 2] = *b.offset((b_idx + 2) as isize);
        b_arr[row * MICROBLOCKCOL + 3] = *b.offset((b_idx + 3) as isize);
    }

    for row in 0 .. MICROBLOCKROW {
        let a_idx = row * m_stride;
        a_arr[row * MICROBLOCKM + 0] = *a.offset(a_idx as isize);
        a_arr[row * MICROBLOCKM + 1] = *a.offset((a_idx + 1) as isize);
        a_arr[row * MICROBLOCKM + 2] = *a.offset((a_idx + 2) as isize);
        a_arr[row * MICROBLOCKM + 3] = *a.offset((a_idx + 3) as isize);
    }

    let mut c_idx = 0;
    for row in 0 .. MICROBLOCKROW {
        let c_elt = c.offset(c_idx as isize);
        let mut c_row: __m256d = _mm256_loadu_pd(c_elt as *const f64);

        for k in 0 .. MICROBLOCKM {
            let a_idx = row * MICROBLOCKM + k;
            let a_elt = &a_arr[a_idx];
            let a_mult: __m256d = _mm256_broadcast_sd(a_elt);

            let b_idx = k * MICROBLOCKCOL;
            let b_elt = &(b_arr[b_idx]) as *const f64;
            let b_row: __m256d = _mm256_loadu_pd(b_elt);
            c_row = _mm256_fmadd_pd(a_mult, b_row, c_row);
        }

        _mm256_storeu_pd(c_elt, c_row);
        c_idx += p_stride;
    }
}
