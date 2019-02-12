use super::lib::random_array;
use super::lib::{minimatrix_fmadd64};
use super::lib::{matrix_madd, floateq};
use ndarray::Array;
use ndarray::linalg::general_mat_mul;

#[test]
fn test_minimatrix_aligned() {
    let a_arr = vec!(1.0, 2.0, 3.0, 4.0,
                     7.0, 6.0, 5.0, 4.0,
                     0.5, 1.0, 2.0, 4.0,
                     8.0, 2.0, 0.5, 0.125);

    let _align1 = vec!(0.0, 0.0, 0.0, 0.0);

    let b_arr = vec!(12.0,  13.0, 15.0, 17.0,
                     01.0,   2.0,  3.0,  4.0,
                     42.0, 404.0, 13.0,  9.0,
                     01.0,   2.0,  3.0,  4.0);

    let _align2 = vec!(0.0, 0.0, 0.0, 0.0);

    let mut c_arr = vec!(10.0, 20.0,  30.0,  40.0,
                         02.0,  6.0,  24.0, 120.0,
                         01.0,  1.0,   2.0,   3.0,
                         05.0, 25.0, 125.0, 625.0);

    let res_arr = vec!(154.000, 1257.00, 102.000, 108.0,
                       306.000, 2137.00, 224.000, 324.0,
                       096.000,  825.50,  50.500,  49.5,
                       124.125,  335.25, 257.875, 447.0);

    minimatrix_fmadd64(4, &a_arr, &b_arr, &mut c_arr);
    test_equality(4, 4, &c_arr, &res_arr);
}

#[test]
fn test_minimatrix_unaligned() {
    let a_arr = vec!(1.0, 2.0, 3.0, 4.0,
                     7.0, 6.0, 5.0, 4.0,
                     0.5, 1.0, 2.0, 4.0,
                     8.0, 2.0, 0.5, 0.125);

    let b_arr = vec!(12.0,  13.0, 15.0, 17.0,
                     01.0,   2.0,  3.0,  4.0,
                     42.0, 404.0, 13.0,  9.0,
                     01.0,   2.0,  3.0,  4.0);

    let mut c_arr = vec!(10.0, 20.0,  30.0,  40.0,
                         02.0,  6.0,  24.0, 120.0,
                         01.0,  1.0,   2.0,   3.0,
                         05.0, 25.0, 125.0, 625.0);

    let res_arr = vec!(154.000, 1257.00, 102.000, 108.0,
                       306.000, 2137.00, 224.000, 324.0,
                       096.000,  825.50,  50.500,  49.5,
                       124.125,  335.25, 257.875, 447.0);

    minimatrix_fmadd64(4, &a_arr, &b_arr, &mut c_arr);

    test_equality(4, 4, &c_arr, &res_arr);
}

#[test]
fn matrix_madd_4x4() {
    let a: Vec<f64> = random_array(4, 4, -100.0, 100.0);
    let b = random_array(4, 4, -100.0, 100.0);
    let mut c = random_array(4, 4, -100.0, 100.0);


    let a_arr = Array::from_vec(a.clone()).into_shape((4, 4)).unwrap();
    let b_arr = Array::from_vec(b.clone()).into_shape((4, 4)).unwrap();
    let mut c_arr = Array::from_vec(c.clone()).into_shape((4, 4)).unwrap();

    general_mat_mul(1.0, &a_arr, &b_arr, 1.0, &mut c_arr);
    matrix_madd(4, 4, &a, &b, &mut c);
    let slice = c_arr.as_slice().unwrap();

    test_equality(4, 4, &c, slice);
}

#[test]
fn matrix_madd_test() {
    for n in 1 .. 8 {
        matrix_madd_nxm(n * 4, n * 4);
    }
}

pub fn matrix_madd_nxm(n: usize, m: usize) {
    let a: Vec<f64> = random_array(n, m, -100.0, 100.0);
    let b = random_array(n, m, -100.0, 100.0);
    let mut c = random_array(n, m, -100.0, 100.0);

    let a_arr = Array::from_vec(a.clone()).into_shape((n, m)).unwrap();
    let b_arr = Array::from_vec(b.clone()).into_shape((n, m)).unwrap();
    let mut c_arr = Array::from_vec(c.clone()).into_shape((n, m)).unwrap();

    general_mat_mul(1.0, &a_arr, &b_arr, 1.0, &mut c_arr);
    matrix_madd(n, m, &a, &b, &mut c);
    let slice = c_arr.as_slice().unwrap();

    test_equality(n, m, &c, slice);
}

fn test_equality(rows: usize, cols: usize, c: &[f64], correct: &[f64]) {
    for row in 0 .. rows {
        let ridx = row * cols;
        for col in 0 .. cols {
            assert!(floateq(c[ridx + col], c[ridx + col]));
        }
    }
}
    
