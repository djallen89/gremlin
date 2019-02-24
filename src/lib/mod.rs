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
pub fn matrix_madd_parallel(n_rows: usize, m_dim: usize, p_cols: usize,
                            a: &[f64], b: &[f64], c: &mut [f64]) {

    match check_dimensionality(n_rows, m_dim, p_cols, a, b, c) {
        Ok(_) => {},
        Err(f) => panic!(f)
    }

    let threads = num_cpus::get_physical();

    let rows = n_rows / threads;

    let mut a_ptrs: Vec<&[f64]> = Vec::with_capacity(threads);
    let mut c_ptrs: Vec<&mut [f64]> = Vec::with_capacity(threads);

    let mut a2 = a;
    let mut c2 = c;
    for _ in 0 .. threads - 1 {
        let (a1, _a2) = a2.split_at(rows * m_dim);
        let (c1, _c2) = c2.split_at_mut(rows * p_cols);
        a2 = _a2;
        c2 = _c2;
        a_ptrs.push(a1);
        c_ptrs.push(c1);
    }

    a_ptrs.push(a2);
    c_ptrs.push(c2);

    c_ptrs.par_iter_mut().zip(a_ptrs.into_par_iter()).for_each(|(c_arr, a_arr)| {
        let sub_rows = a_arr.len() / m_dim;
        if sub_rows != c_arr.len() / p_cols {
            panic!("BAD")
        }
        let a_ptr = a_arr.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c_arr.as_mut_ptr();
        
        matrix_mul_add(m_dim, p_cols,
                       sub_rows, m_dim, p_cols,
                       a_ptr, b_ptr, c_ptr);
    });

}
