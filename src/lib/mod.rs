//#![feature(align_offset)]
extern crate ndarray;
extern crate rayon;

mod utilities;
mod matrix_math;

pub use utilities::{get_idx, random_array, float_eq, test_equality};
pub use utilities::{matrix_madd_n_sq, matrix_madd_nxm, matrix_madd_nmp};
pub use utilities::matrix_multithread;
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

    let a_ptr = &a[0] as *const f64;
    let b_ptr = &b[0] as *const f64;
    let c_ptr = &mut c[0] as *mut f64;

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
                   a_ptr, b_ptr, c_ptr)
}

pub fn matrix_madd_multithread(threads: usize, n: usize,
                               a: & [f64],
                               b: & [f64],
                               c: &mut [f64]) {
    let rows = n / threads;
    let (a1, a2) = a.split_at(rows * n);
    let (c1, c2) = c.split_at_mut(rows * n);

    let a_ptrs = [a1, a2];
    let mut c_ptrs = [c1, c2];

    let thread_arr: Vec<usize> = (0 .. threads).map(|x| x).collect();

    c_ptrs.par_iter_mut().zip(a_ptrs.into_par_iter()).for_each(|(mut c_arr, a_arr)| {
        let a_ptr = &a_arr[0] as *const f64;
        let b_ptr = &b[0] as *const f64;
        //let c_ptr = &mut (c_ptrs[thread])[n * rows] as *mut f64;
        let c_ptr = &mut (c_arr)[0] as *mut f64;
        //println!("len a: {}, len c: {}", a_arr.len(), c_arr.len());
        
        matrix_mul_add(n, n,
                       rows, n, n,
                       a_ptr, b_ptr, c_ptr);
    });

}
