use super::lib::random_array;
//use super::lib::{minimatrix_fmadd64};
use crate::lib::matrix_madd;
use crate::lib::{matrix_madd_nxm, matrix_madd_nmp};

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
    matrix_madd_nmp(1, 2048, 1);
}

