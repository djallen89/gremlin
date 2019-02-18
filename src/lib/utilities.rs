extern crate rand;

use rand::prelude::*;
use ndarray::Array;
use ndarray::linalg::general_mat_mul;
use super::matrix_madd;

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
pub fn get_elt(row: usize, col: usize, n_cols: usize) -> usize {
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
    let a: Vec<f64> = random_array(n, n, -100.0, 100.0);
    let b = random_array(n, n, -100.0, 100.0);
    let mut c = random_array(n, n, -100.0, 100.0);
    
    let aarr = Array::from_vec(a.clone()).into_shape((n, n)).unwrap();
    let barr = Array::from_vec(b.clone()).into_shape((n, n)).unwrap();
    let mut carr = Array::from_vec(c.clone()).into_shape((n, n)).unwrap();

    general_mat_mul(1.0, &aarr, &barr, 1.0, &mut carr);
    let slice = carr.as_slice().unwrap();
    
    matrix_madd(n, n, n, &a, &b, &mut c);

    test_equality(n, n, &c, &slice);
}

pub fn matrix_madd_nxm(n: usize, m: usize) {
    let a: Vec<f64> = random_array(n, m, -100.0, 100.0);
    let b = random_array(m, n, -100.0, 100.0);
    let mut c = random_array(n, n, -100.0, 100.0);
    
    let aarr = Array::from_vec(a.clone()).into_shape((n, m)).unwrap();
    let barr = Array::from_vec(b.clone()).into_shape((m, n)).unwrap();
    let mut carr = Array::from_vec(c.clone()).into_shape((n, n)).unwrap();

    general_mat_mul(1.0, &aarr, &barr, 1.0, &mut carr);
    let slice = carr.as_slice().unwrap();
    
    matrix_madd(n, m, n, &a, &b, &mut c);

    test_equality(n, m, &c, &slice);
}

pub fn matrix_madd_nmp(n: usize, m: usize, p: usize) {
    let a: Vec<f64> = random_array(n, m, -100.0, 100.0);
    let b = random_array(m, p, -100.0, 100.0);
    let mut c = random_array(n, p, -100.0, 100.0);
    
    let aarr = Array::from_vec(a.clone()).into_shape((n, m)).unwrap();
    let barr = Array::from_vec(b.clone()).into_shape((m, p)).unwrap();
    let mut carr = Array::from_vec(c.clone()).into_shape((n, p)).unwrap();

    general_mat_mul(1.0, &aarr, &barr, 1.0, &mut carr);
    let slice = carr.as_slice().unwrap();
    
    matrix_madd(n, m, p, &a, &b, &mut c);

    test_equality(n, p, &c, &slice);
}

pub fn test_equality(rows: usize, cols: usize, c: &[f64], correct: &[f64]) {
    let mut i_msgs = String::new();
    let mut equal = true;
    let mut inequalities = 0;
    for i in 0 .. rows {
        for j in 0 .. cols {
            if !float_eq(c[i], correct[i]) {
                inequalities += 1;
                equal = false;
                if rows * cols < 25 {
                    i_msgs = format!("{} {},{}", i_msgs, i + 1, j + 1);
                }
            }
        }
        if i_msgs.len() > 0 {
            i_msgs = format!("{}\n", i_msgs);
        }
    }

    if !equal {
        if rows * cols < 25 {
            panic!("{}", i_msgs);
        } else {
            panic!("{} inequalities", inequalities);
        }
    }
}

