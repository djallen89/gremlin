//#![feature(align_offset)]
extern crate ndarray;
extern crate rayon;
extern crate num_cpus;

mod utilities;
mod matrix_math;

pub use utilities::{get_idx, random_array, float_eq, test_equality};
pub use utilities::{matrix_madd_n_sq, matrix_madd_nxm, matrix_madd_nmp};
pub use utilities::{matrix_madd_n_sq_parallel, matrix_madd_nxm_parallel,
                    matrix_madd_nmp_parallel};
use utilities::check_dimensionality;
use matrix_math::{single_dot_prod_add, small_matrix_mul_add, matrix_mul_add};
use matrix_math::{minimatrix_fmadd_f64, scalar_vector_fmadd};
use rayon::prelude::*;
use std::slice;

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
            
    } else if n_rows == 4 && p_cols == 4 && m_dim == 4 {
        
        return minimatrix_fmadd_f64(m_dim, p_cols, a_ptr, b_ptr, c_ptr);
        
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
    /*
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
     */
    parallel_helper(threads, m_dim, p_cols,
                    n_rows, m_dim, p_cols,
                    a, b, c)
}

fn parallel_helper(threads: usize, m_stride: usize, p_stride: usize,
                   n_rows: usize, m_dim: usize, p_cols: usize,
                   a: &[f64], b: &[f64], c: &mut [f64]) {
    //println!("threads={}, rows={}, cols={}, m_dim={}", threads,
             //n_rows, p_cols, m_dim);
    if threads == 1 {
        return matrix_mul_add(m_dim, p_cols,
                       n_rows, m_dim, p_cols,
                       a.as_ptr(), b.as_ptr(), c.as_mut_ptr())
    }
    /*
    // split A and C horizontally, call parallel_helper for each
    // by spawning 2 threads and passing threads / 2, and appropriate dimensions,
    let n_rows_new = n_rows / 2;
    let n_rows_rem = n_rows - n_rows_new;
    let (a1, a2) = a.split_at(get_idx(n_rows_new, 0, m_stride));
    let (c1, c2) = c.split_at_mut(get_idx(n_rows_new, 0, p_stride));

    let mut c_ptrs = [c1, c2];
    let a_ptrs = [a1, a2];

    c_ptrs.par_iter_mut().enumerate().zip(a_ptrs.into_par_iter()).for_each(|((i, c_arr), a_arr)| {
        let rows = if i == 0 {
            n_rows_new
        } else {
            n_rows_rem
        };
        
        parallel_helper(threads / 2, m_stride, p_stride,
                        rows, m_dim, p_cols,
                        a_arr, b, c_arr);
    });
     */
/*
    if n_rows >= p_cols || n_rows >= m_dim {
*/
        // split A and C horizontally, call parallel_helper for each
        // by spawning 2 threads and passing threads / 2, and appropriate dimensions,
        let n_rows_new = n_rows / 2;
        let n_rows_rem = n_rows - n_rows_new;
        let (a1, a2) = a.split_at(get_idx(n_rows_new, 0, m_stride));
        let (c1, c2) = c.split_at_mut(get_idx(n_rows_new, 0, p_stride));

        let mut c_ptrs = [c1, c2];
        let a_ptrs = [a1, a2];
        
        c_ptrs.par_iter_mut().enumerate().zip(a_ptrs.into_par_iter()).for_each(|((i, c_arr), a_arr)| {
            let sub_rows = if i == 0 {
                n_rows_new
            } else {
                n_rows_rem
            };
            
            parallel_helper(threads / 2, m_stride, p_stride,
                            sub_rows, m_dim, p_cols,
                            a_arr, b, c_arr);
        });
        
        //    } else if p_cols > n_rows || p_cols >= m_dim {
/*
    } else { 

        let p_cols_new = p_cols / 2;
        let p_cols_rem = p_cols - p_cols_new;
        let (b1, b2) = b.split_at(p_cols_new);
        let (c1, c2) = c.split_at_mut(p_cols_new);

        /* This is shameful, but the algorithm won't work with slices otherwise. */
        let mut _b1 = b1;
        let mut _c1 = c1;
        unsafe {
            let b1_ptr = _b1.as_ptr();
            _b1 = slice::from_raw_parts(b1_ptr, b.len());
            let c1_ptr = _c1.as_mut_ptr();
            _c1 = slice::from_raw_parts_mut(c1_ptr, n_rows * p_stride);
        }
        let b1 = _b1;
        let c1 = _c1;

        let mut c_ptrs = [c1, c2];
        let b_ptrs = [b1, b2];
        
        c_ptrs.par_iter_mut().enumerate().zip(b_ptrs.into_par_iter()).for_each(|((i, c_arr), b_arr)| {
            let sub_cols = if i == 0 {
                p_cols_rem
            } else {
                p_cols_new
            };
            
            parallel_helper(threads / 2, thread_id + i + 1,
                            m_stride, p_stride,
                            n_rows, m_dim, sub_cols,
                            a, b_arr, c_arr);
        });
    */
    //}
}
