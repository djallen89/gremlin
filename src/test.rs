use super::lib::random_array;
//use super::lib::{minimatrix_fmadd64};
use crate::lib::matrix_madd;
use crate::lib::{matrix_madd_nxm, matrix_madd_nmp};

#[test]
fn matrix_madd_8x8() {
    matrix_madd_nxm(8,8);
}


#[test]
fn matrix_madd_1to5() {
    for n in 1 .. 6 {
        matrix_madd_nxm(n, n);
    }
}

#[test]
fn matrix_madd_test() {
    for n in 1 .. 33 {
        matrix_madd_nxm(n, n);
    }
}


#[test]
fn matrix_4sq_test() {
    matrix_madd_nxm(4, 4);
}

#[test]
fn matrix_256sq_test() {
    matrix_madd_nxm(256, 256);
}

#[test]
fn matrix_480_sq_test() {
    matrix_madd_nxm(480, 480);
}

#[test]
fn matrix_508_sq_test() {
    matrix_madd_nxm(508, 508);
}

#[test]
fn matrix_512_sq_test() {
    matrix_madd_nxm(512, 512);
}

#[test]
fn matrix_516_sq_test() {
    matrix_madd_nxm(516, 516);
}

#[test]
fn matrix_736sq_test() {
    matrix_madd_nxm(736, 736);
}
/*
#[test]
fn matrix_768sq_test() {
    matrix_madd_nxm(768, 768);
}

#[test]
fn matrix_992_sq_test() {
    matrix_madd_nxm(992, 992);
}

#[test]
fn matrix_1024sq_test() {
    matrix_madd_nxm(1024, 1024);
}

#[test]
fn matrix_1408_sq_test() {
    matrix_madd_nxm(1408, 1408);
}
*/
#[test]
fn matrix_2048x1_test() {
    matrix_madd_nmp(1, 2048, 1);
}

