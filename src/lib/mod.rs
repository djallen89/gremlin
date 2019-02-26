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
use utilities::check_dimensionality;
use matrix_math::{single_dot_prod_add, small_matrix_mul_add, matrix_mul_add};
use matrix_math::scalar_vector_fmadd;
use rayon::prelude::*;
use danger_math::Ptr;

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
        2 => return two_threaded(n_rows, m_dim, p_cols, a, b, c),
        x => unsafe {
            return multi_threaded(x, m_dim, p_cols, n_rows, m_dim, p_cols,
                                  a.as_ptr(), b.as_ptr(), c.as_mut_ptr())
        }
    }
}

fn two_threaded(n_rows: usize, m_dim: usize, p_cols: usize,
                a: &[f64], b: &[f64], c: &mut [f64]) {
    if n_rows >= p_cols {
        let first_rows = n_rows / 2;
        let last_rows = n_rows - first_rows;
        let a1 = Ptr::from_addr(&a[0]);
        let a2 = Ptr::from_addr(&a[first_rows * m_dim]);
        let c1 = Ptr::from_addr(&c[0]);
        let c2 = Ptr::from_addr(&c[first_rows * p_cols]);

        let a_ptrs = [a1, a2];
        let c_ptrs = [c1, c2];

        c_ptrs.par_iter().enumerate().zip(a_ptrs.into_par_iter()).for_each(|((i, c_arr), a_arr)| {
            let rows = if i == 0 {
                first_rows
            } else {
                last_rows
            };

            let a_ptr = a_arr.as_ptr();
            let b_ptr = b.as_ptr();
            let c_ptr = c_arr.as_mut_ptr();
            
            matrix_mul_add(m_dim, p_cols,
                           rows, m_dim, p_cols,
                           a_ptr, b_ptr, c_ptr);
        });
        
    } else {
        let first_cols = p_cols / 2;
        let last_cols = p_cols - first_cols;

        let b1 = Ptr::from_addr(&b[0]);
        let b2 = Ptr::from_addr(&b[first_cols]);
        let c1 = Ptr::from_addr(&c[0]);
        let c2 = Ptr::from_addr(&c[first_cols]);

        let b_ptrs = [b1, b2];
        let c_ptrs = [c1, c2];

        c_ptrs.par_iter().enumerate().zip(b_ptrs.into_par_iter()).for_each(|((i, c_arr), b_arr)| {
            let cols = if i == 0 {
                first_cols
            } else {
                last_cols
            };

            let a_ptr = a.as_ptr();
            let b_ptr = b_arr.as_ptr();
            let c_ptr = c_arr.as_mut_ptr();
            
            matrix_mul_add(m_dim, p_cols,
                           n_rows, m_dim, cols,
                           a_ptr, b_ptr, c_ptr);
        });

    }
}

unsafe fn multi_threaded(threads: usize, m_stride: usize, p_stride: usize,
                         n_rows: usize, m_dim: usize, p_cols: usize,
                         a: *const f64, b: *const f64, c: *mut f64) {
    if threads == 1 {
        return matrix_mul_add(m_stride, p_stride,
                              n_rows, m_dim, p_cols,
                              a, b, c)
    } 
    
    /* threads > 1 */
    let new_threads = threads / 2;
    let rem_threads = threads - new_threads;

    if n_rows >= p_cols {
        let b_p = Ptr::from_ptr(b);
        let first_rows = n_rows / 2;
        let last_rows = n_rows - first_rows;
        
        let a1 = Ptr::from_ptr(a);
        let a2 = Ptr::from_ptr(a.offset((first_rows * m_stride) as isize));
        let c1 = Ptr::from_ptr(c);
        let c2 = Ptr::from_ptr(c.offset((first_rows * p_stride) as isize));

        let a_ptrs = [a1, a2];
        let c_ptrs = [c1, c2];

        c_ptrs.par_iter().enumerate().zip(a_ptrs.into_par_iter()).for_each(|((i, c_arr), a_arr)| {
            let (rows, sub_threads) = if i == 0 {
                (first_rows, new_threads)
            } else {
                (last_rows, rem_threads)
            };

            let a_ptr = a_arr.as_ptr();
            let b_ptr = b_p.as_ptr();
            let c_ptr = c_arr.as_mut_ptr();
            
            multi_threaded(sub_threads, m_stride, p_stride,
                           rows, m_dim, p_cols,
                           a_ptr, b_ptr, c_ptr);
        });
        
    } else {
        let a_p = Ptr::from_ptr(a);
        let first_cols = p_cols / 2;
        let last_cols = p_cols - first_cols;

        let b1 = Ptr::from_ptr(b);
        let b2 = Ptr::from_ptr(b.offset(first_cols as isize));
        let c1 = Ptr::from_ptr(c);
        let c2 = Ptr::from_ptr(c.offset(first_cols as isize));

        let b_ptrs = [b1, b2];
        let c_ptrs = [c1, c2];

        c_ptrs.par_iter().enumerate().zip(b_ptrs.into_par_iter()).for_each(|((i, c_arr), b_arr)| {
            let (cols, sub_threads) = if i == 0 {
                (first_cols, new_threads)
            } else {
                (last_cols, rem_threads)
            };

            let a_ptr = a_p.as_ptr();
            let b_ptr = b_arr.as_ptr();
            let c_ptr = c_arr.as_mut_ptr();
            
            multi_threaded(sub_threads, m_stride, p_stride,
                           n_rows, m_dim, cols,
                           a_ptr, b_ptr, c_ptr);
        });

    }
}
