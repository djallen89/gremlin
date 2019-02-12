use super::lib::random_array;
use super::lib::{minimatrix_fmadd64};
use super::lib::{matrix_madd, floateq};
use ndarray::Array;
use ndarray::linalg::general_mat_mul;
use matrixmultiply::dgemm;

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

    minimatrix_fmadd64(4, 4, &a_arr, &b_arr, &mut c_arr);
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

    minimatrix_fmadd64(4, 4, &a_arr, &b_arr, &mut c_arr);

    test_equality(4, 4, &c_arr, &res_arr);
}

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

#[test]
fn matrix_1024_test() {
    matrix_madd_nxm(1024, 1024);
}

pub fn matrix_madd_nxm(n: usize, m: usize) {
    let a: Vec<f64> = random_array(n, m, -100.0, 100.0);
    let b = random_array(n, m, -100.0, 100.0);
    let mut c = random_array(n, m, -100.0, 100.0);
    
    let a_arr = a.clone();
    let b_arr = b.clone();
    let mut c_arr = c.clone();

    unsafe { 
        dgemm(n, m, n, 1.0,
              &a_arr[0] as *const f64, 4, 4,
              &b_arr[0] as *const f64, 4, 4,
              1.0, &mut c_arr[0] as *mut f64, 4, 4)
    }

    matrix_madd(n, m, &a, &b, &mut c);

    test_equality(n, m, &c, &c_arr);
}

fn test_equality(rows: usize, cols: usize, c: &[f64], correct: &[f64]) {
    /*
    for i in 0 .. rows * cols {
            assert!(floateq(c[i], correct[i]));
    }
     */

    for row in 0 .. 4 {
        let ridx = row * 4;
        for col in 0 .. 4 {
            //assert!(res_arr[ridx + col] == c_arr[ridx + col]);
            assert!(floateq(c[ridx + col], correct [ridx + col]));
        }
    }
}
    
