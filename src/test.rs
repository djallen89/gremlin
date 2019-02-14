use super::lib::random_array;
use super::lib::{minimatrix_fmadd64};
use super::lib::{matrix_madd, floateq};
use ndarray::Array;
use ndarray::linalg::general_mat_mul;

#[test]
fn matrix_madd_8x8() {
    matrix_madd_nxm(8,8);
}

/*
#[test]
fn matrix_madd_1to5() {
    for n in 1 .. 6 {
        matrix_madd_nxm(n, n);
    }
}

#[test]
fn matrix_madd_test() {
    for n in 1 .. 8 {
        matrix_madd_nxm(n * 4, n * 4);
    }
}
*/

#[test]
fn matrix_4sq_test() {
    matrix_madd_nxm(4, 4);
}

#[test]
fn matrix_256sq_test() {
    matrix_madd_nxm(256, 256);
}

#[test]
fn matrix_2048x1_test() {
    matrix_madd_nmp(1, 1, 2048);
}

pub fn matrix_madd_nmp(n: usize, m: usize, p: usize) {
    let a: Vec<f64> = random_array(n, p, -100.0, 100.0);
    let b = random_array(p, m, -100.0, 100.0);
    let mut c = random_array(n, m, -100.0, 100.0);
    
    let aarr = Array::from_vec(a.clone()).into_shape((n, p)).unwrap();
    let barr = Array::from_vec(b.clone()).into_shape((p, m)).unwrap();
    let mut carr = Array::from_vec(c.clone()).into_shape((n, m)).unwrap();

    general_mat_mul(1.0, &aarr, &barr, 1.0, &mut carr);
    let slice = carr.as_slice().unwrap();
    
    matrix_madd(n, m, p, &a, &b, &mut c);

    test_equality(n, m, &c, &slice);
}

pub fn matrix_madd_nxm(n: usize, m: usize) {
    let a: Vec<f64> = random_array(n, m, -100.0, 100.0);
    let b = random_array(n, m, -100.0, 100.0);
    let mut c = random_array(n, m, -100.0, 100.0);
    
    let aarr = Array::from_vec(a.clone()).into_shape((n, m)).unwrap();
    let barr = Array::from_vec(b.clone()).into_shape((n, m)).unwrap();
    let mut carr = Array::from_vec(c.clone()).into_shape((n, m)).unwrap();

    general_mat_mul(1.0, &aarr, &barr, 1.0, &mut carr);
    let slice = carr.as_slice().unwrap();
    
    matrix_madd(n, m, n, &a, &b, &mut c);

    test_equality(n, m, &c, &slice);
}

fn test_equality(rows: usize, cols: usize, c: &[f64], correct: &[f64]) {
    for i in 0 .. rows * cols {
        if !floateq(c[i], correct[i]) {
            if rows * cols <= 16 {
                for i in 0 .. rows * cols {
                    println!("{} != {}",  c[i], correct[i]);
                }
            }
            panic!("{}, {} != {}", i, c[i], correct[i]);
        }
        assert!(floateq(c[i], correct[i]));
    }
}
    
