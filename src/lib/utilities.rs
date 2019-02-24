extern crate rand;

use rand::prelude::*;
use ndarray::Array;
use ndarray::linalg::general_mat_mul;
use super::matrix_madd;
use super::matrix_madd_parallel;

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
pub fn get_idx(row: usize, col: usize, n_cols: usize) -> usize {
    row * n_cols + col
}

pub fn float_eq(a: f64, b: f64) -> bool {
    use std::f64;
    
    let abs_a = a.abs();
    let abs_b = b.abs();
    let diff = (a - b).abs();
    let epsilon = 0.00001;

    if a == b {
	true
    } else if a == 0.0 || b == 0.0 || diff < f64::MIN_POSITIVE {
	diff < (epsilon * f64::MIN_POSITIVE)
    } else { 
	(diff / f64::min(abs_a + abs_b, f64::MAX)) < epsilon
    }

}

pub fn random_array<T>(cols: usize, rows: usize, low: T, high: T) -> Vec<T>
    where T: rand::distributions::uniform::SampleUniform
{
    use rand::distributions::Uniform;
    use std::usize;
    
    assert!(usize::MAX / rows > cols);
    
    let interval = Uniform::from(low .. high);
    let mut rng = rand::thread_rng();
    let mut arr = Vec::with_capacity(rows * cols);
    
    for _ in 0 .. rows * cols {
        arr.push(interval.sample(&mut rng))
    }

    return arr;
}

pub fn matrix_madd_n_sq(n: usize) {
    matrix_madd_nmp(n, n, n);
}

pub fn matrix_madd_nxm(n: usize, m: usize) {
    matrix_madd_nmp(n, m, n);
}

pub fn matrix_madd_nmp(n: usize, m: usize, p: usize) {
    let a: Vec<f64> = random_array(n, m, -10000.0, 10000.0);
    let b = random_array(m, p, -10000.0, 10000.0);
    let mut c = random_array(n, p, -10000.0, 10000.0);
    
    let aarr = Array::from_vec(a.clone()).into_shape((n, m)).unwrap();
    let barr = Array::from_vec(b.clone()).into_shape((m, p)).unwrap();
    let mut carr = Array::from_vec(c.clone()).into_shape((n, p)).unwrap();

    general_mat_mul(1.0, &aarr, &barr, 1.0, &mut carr);
    let slice = carr.as_slice().unwrap();
    
    matrix_madd(n, m, p, &a, &b, &mut c);

    test_equality(n, p, &c, &slice);
}

pub fn matrix_madd_n_sq_parallel(n: usize) {
    matrix_madd_nmp_parallel(n, n, n);
}

pub fn matrix_madd_nxm_parallel(n: usize, m: usize) {
    matrix_madd_nmp_parallel(n, m, n);
}

pub fn matrix_madd_nmp_parallel(n: usize, m: usize, p: usize) {
    let threads = num_cpus::get_physical();
    let a: Vec<f64> = random_array(n, m, -10000.0, 10000.0);
    let b = random_array(m, p, -10000.0, 10000.0);
    let mut c = random_array(n, p, -10000.0, 10000.0);
    
    let aarr = Array::from_vec(a.clone()).into_shape((n, m)).unwrap();
    let barr = Array::from_vec(b.clone()).into_shape((m, p)).unwrap();
    let mut carr = Array::from_vec(c.clone()).into_shape((n, p)).unwrap();

    general_mat_mul(1.0, &aarr, &barr, 1.0, &mut carr);
    let slice = carr.as_slice().unwrap();
    
    matrix_madd_parallel(threads, n, m, p, &a, &b, &mut c);

    test_equality(n, p, &c, &slice);
}

pub fn test_equality(rows: usize, cols: usize, c: &[f64], correct: &[f64]) {
    let mut i_msgs = String::new();
    let mut equal = true;
    let mut inequalities = 0;
    const LIM: usize = 50;
    for i in 0 .. rows {
        for j in 0 .. cols {
            if !float_eq(c[i * cols + j], correct[i * cols + j]) {
                inequalities += 1;
                equal = false;
                if rows * cols < LIM {
                    i_msgs = format!("{}\n{},{}", i_msgs, i + 1, j + 1);
                }
            }
        }
    }

    if !equal {
        if rows * cols < LIM {
            panic!("{}", i_msgs);
        } else {
            panic!("{} inequalities", inequalities);
        }
    }
}

pub fn check_dimensionality(n_rows: usize, m_dim: usize, p_cols: usize,
                            a: &[f64],
                            b: &[f64],
                            c: &[f64]) -> Result<(), String> {
    /* Check dimensionality */
    let a_len = a.len();
    let b_len = b.len();
    let c_len = c.len();
    /* Check for zeros before checking dimensions to prevent division by zero */
    match (n_rows, p_cols, m_dim, a_len, b_len, c_len) {
        (0, _, _, _, _, _) |  (_, 0, _, _, _, _) | (_, _, 0, _, _, _) |
        (_, _, _, 0, _, _) | (_, _, _, _, 0, _) => {
            return Err(format!("Cannot do matrix multiplication where A or B have 0 elements"))
        },
        (_, _, _, _, _, 0) => {
            return Err(format!("Cannot do matrix multiplication where C has 0 elements"))
        },
        _ => {}
    }

    if a_len / n_rows != m_dim {
        /* A is n * m*/
        Err(format!("{}\n{}*{} == {} != {}",
                    "Dimensionality of A does not match parameters.",
                    n_rows, m_dim, n_rows * m_dim, a_len))
    } else if b_len / p_cols != m_dim {
        /* B is m * p */
        Err(format!("{}\n{}*{} == {} != {}",
                    "Dimensionality of B does not match parameters.",
                    p_cols, m_dim, p_cols * m_dim, b_len))
    } else if c_len / n_rows != p_cols {
        /* C is n * p */
        Err(format!("{}\n{}*{} == {} != {}",
                    "Dimensionality of C does not match parameters.",
                    n_rows, p_cols, n_rows * p_cols, c_len))
    } else {
        Ok(())
    }
}
