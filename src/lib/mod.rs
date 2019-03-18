//#![feature(align_offset)]
extern crate ndarray;
extern crate rayon;
extern crate num_cpus;

mod utilities;
mod matrix_math;
mod danger_math;

pub use utilities::{get_idx, random_array, float_eq, test_equality};
pub use utilities::{matrix_madd_n_sq, matrix_madd_nxm, matrix_madd_nmp};
pub use utilities::{matrix_madd_n_sq_parallel, matrix_madd_nxm_parallel,
                    matrix_madd_nmp_parallel};
pub use utilities::matrix_madd_n_sq_chunked;
use utilities::{check_dimensionality, total_size};
use matrix_math::{single_dot_prod_add, small_matrix_mul_add, matrix_mul_add};
use matrix_math::scalar_vector_fmadd;
use matrix_math::{L2_SIZE, L3_SIZE, MEGABLOCKCOL, BLOCKROW};
use rayon::prelude::*;
use danger_math::Ptr;
use std::cmp::{min, max};
use std::mem;

/// Calculates C <= AB + C, where A is a matrix of n rows by m columns,
/// B is a matrix of m rows by p columns, and C is a matrix of n rows
/// by p columns.
pub fn matrix_madd(n_rows: usize, m_dim: usize, p_cols: usize,
                   a: &[f64], b: &[f64], c: &mut [f64]) {
    match check_dimensionality(n_rows, m_dim, p_cols, a, b, c) {
        Ok(_) => {},
        Err(f) => panic!(f)
    }

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    if n_rows == 1 && m_dim == 1 && p_cols >= 4 {

        return scalar_vector_fmadd(p_cols, a[0], b_ptr, c_ptr);
        
    } else if n_rows == 1 && p_cols == 1 && m_dim >= 4 {

        return single_dot_prod_add(m_dim, a_ptr, b_ptr, c_ptr);
        
    } else if p_cols < 4 || n_rows < 4 {

        return small_matrix_mul_add(n_rows, m_dim, p_cols, a, b, c)
            
    } 

    matrix_mul_add(m_dim, p_cols,
                   n_rows, m_dim, p_cols, 
                   a_ptr, b_ptr, c_ptr);
}

/// Calculates C <= AB + C, where A is a matrix of n rows by m
/// columns, B is a matrix of m rows by p columns, and C is a matrix
/// of n rows by p columns. Attempts to use as many threads as there
/// are physical CPU cores on the system.
pub fn matrix_madd_parallel(threads: usize, n_rows: usize, m_dim: usize, p_cols: usize,
                            a: &[f64], b: &[f64], c: &mut [f64]) {     
    if n_rows * m_dim * p_cols <= 8000 {
        return matrix_madd(n_rows, m_dim, p_cols, a, b, c);
    }

    match check_dimensionality(n_rows, m_dim, p_cols, a, b, c) {
        Ok(_) => {},
        Err(f) => panic!(f)
    }

    match threads {
        0 => panic!("0 cores available to do work"),
        1 => return matrix_madd(n_rows, m_dim, p_cols, a, b, c),
        x => return multithreaded(x, n_rows, m_dim, p_cols,
                                  a, b, c)
    }
}

pub fn multithreaded(threads: usize, n_rows: usize, m_dim: usize, p_cols: usize,
                     a: &[f64], b: &[f64], c: &mut [f64]) {
    if total_size(mem::size_of::<f64>(), n_rows, m_dim, p_cols) <= L3_SIZE / 8 {
        return matrix_madd(n_rows, m_dim, p_cols, a, b, c);
    }

    match check_dimensionality(n_rows, m_dim, p_cols, a, b, c) {
        Ok(_) => {},
        Err(f) => panic!(f)
    }

    let rows = n_rows / threads;
    let mut a_ptrs: Vec<&[f64]> = Vec::with_capacity(threads);
    let mut c_ptrs: Vec<&mut [f64]> = Vec::with_capacity(threads);
    let mut a2 = a;
    let mut c2 = c;
    for _ in 0 .. threads {
        let (a1, _a2) = a2.split_at(rows * m_dim);
        let (c1, _c2) = c2.split_at_mut(rows * p_cols);
        a2 = _a2;
        c2 = _c2;
        a_ptrs.push(a1);
        c_ptrs.push(c1);
    }
    let sub_rows = n_rows / threads;
    let row_rem = n_rows % threads;
    let last_rows = sub_rows + row_rem;
    c_ptrs.par_iter_mut().enumerate().zip(a_ptrs.into_par_iter()).for_each(|((i, c_arr), a_arr)| {
        let rows = if i == threads - 1 {
            last_rows
        } else {
            sub_rows
        };
        let a_ptr = a_arr.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c_arr.as_mut_ptr();
        
        matrix_mul_add(m_dim, p_cols,
                       rows, m_dim, p_cols,
                       a_ptr, b_ptr, c_ptr);
    });
}

/// Calculates C <= AB + C, where A is a matrix of n rows by m
/// columns, B is a matrix of m rows by p columns, and C is a matrix
/// of n rows by p columns. Attempts to use as many threads as there
/// are physical CPU cores on the system.
pub fn matrix_madd_chunked(threads: usize, n_rows: usize, m_dim: usize, p_cols: usize,
                            a: &[f64], b: &[f64], c: &mut [f64]) {     
    if n_rows * m_dim * p_cols <= 8000 {
        return matrix_madd(n_rows, m_dim, p_cols, a, b, c);
    }

    match check_dimensionality(n_rows, m_dim, p_cols, a, b, c) {
        Ok(_) => {},
        Err(f) => panic!(f)
    }

    match threads {
        0 => panic!("0 cores available to do work"),
        1 => return matrix_madd(n_rows, m_dim, p_cols, a, b, c),
        x => unsafe {
            return chunked_multithreaded(x, m_dim, p_cols,
                                         n_rows, m_dim, p_cols,
                                         a.as_ptr(), b.as_ptr(), c.as_mut_ptr())
        }
    }
}

type MaddParams = (usize, usize, usize, Ptr, Ptr, Ptr);

unsafe fn chunked_multithreaded(threads: usize, m_stride: usize, p_stride: usize,
                                n_rows: usize, m_dim: usize, p_cols: usize,
                                a: *const f64, b: *const f64, c: *mut f64) {
    if threads == 1 {
        return matrix_mul_add(m_stride, p_stride,
                              n_rows, m_dim, p_cols,
                              a, b, c)
    }

    let min_size = L2_SIZE;    
    let args_vec = param_gather(threads, min_size, m_stride, p_stride,
                                n_rows, m_dim, p_cols,
                                a, b, c);

    args_vec.par_iter().for_each(|(rows, m, cols, a_ptr, b_ptr, c_ptr)| {
        let ap = a_ptr.as_ptr();
        let bp = b_ptr.as_ptr();
        let cp = c_ptr.as_mut_ptr();

        matrix_mul_add(m_stride, p_stride, *rows, *m, *cols, ap, bp, cp)
    });
}

unsafe fn param_gather(threads: usize, min_size: usize, m_stride: usize, p_stride: usize,
                       n_rows: usize, m_dim: usize, p_cols: usize,
                       a: *const f64, b: *const f64, c: *mut f64) -> Vec<MaddParams> {

    
    let mut args_vec: Vec<MaddParams> = Vec::new();
    let size = total_size(mem::size_of::<f64>(), n_rows, m_dim, p_cols);

    if size <= min_size || (n_rows <= BLOCKROW && p_cols <= MEGABLOCKCOL) {
    //if size <= min_size || n_rows <= BLOCKROW {
        return vec!((n_rows, m_dim, p_cols,
                     Ptr::from_ptr(a), Ptr::from_ptr(b), Ptr::from_ptr(c)))
    }
     
    if n_rows >= BLOCKROW * 2 {
        let row_rem = n_rows % BLOCKROW;
        let stripes = n_rows - row_rem;
        for stripe in (0 .. stripes).step_by(BLOCKROW) {
            let a_ptr = a.offset((stripe * m_stride) as isize);
            let c_ptr = c.offset((stripe * p_stride) as isize);
            let mut args1 = param_gather(threads, min_size, m_stride, p_stride,
                                         BLOCKROW, m_dim, p_cols,
                                         a_ptr, b, c_ptr);
            args_vec.append(&mut args1);
        }
        let a_ptr = a.offset((stripes * m_stride) as isize);
        let c_ptr = c.offset((stripes * p_stride) as isize);
        let mut args1 = param_gather(threads, min_size, m_stride, p_stride,
                                     row_rem, m_dim, p_cols,
                                     a_ptr, b, c_ptr);
        args_vec.append(&mut args1);
    } else {
        let col_rem = p_cols % MEGABLOCKCOL;
        let pillars = p_cols - col_rem;
        for pillar in (0 .. pillars).step_by(MEGABLOCKCOL) {
            let b_ptr = b.offset(pillar as isize);
            let c_ptr = c.offset(pillar as isize);

            let mut args1 = param_gather(threads, min_size, m_stride, p_stride,
                                         n_rows, m_dim, MEGABLOCKCOL,
                                         a, b_ptr, c_ptr);
            args_vec.append(&mut args1);
        }
        
        let b_ptr = b.offset(pillars as isize);
        let c_ptr = c.offset(pillars as isize);

        let mut args1 = param_gather(threads, min_size, m_stride, p_stride,
                                     n_rows, m_dim, col_rem,
                                     a, b_ptr, c_ptr);

        args_vec.append(&mut args1);
    }

    return args_vec;
}
