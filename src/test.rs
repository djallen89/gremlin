use super::lib::random_array;
use crate::lib::matrix_madd;
use crate::lib::{matrix_madd_n_sq, matrix_madd_nmp};

#[test]
fn matrix_madd_1to15_sq() {
    for n in 1 .. 16 {
        if n % 4 != 0 {
            matrix_madd_n_sq(n);
        }
    }
}

/*
#[test]
fn matrix_4sq_test() {
    matrix_madd_n_sq(4);
}

#[test]
fn matrix_madd_8sq() {
    matrix_madd_n_sq(8);
}

#[test]
fn matrix_madd_16sq() {
    matrix_madd_n_sq(16);
}

#[test]
fn matrix_madd_32sq() {
    matrix_madd_n_sq(32);
}
 */
/*
#[test]
fn matrix_madd_64sq() {
    matrix_madd_n_sq(64);
}
*/
/*
#[test]
fn matrix_madd_128sq() {
    matrix_madd_n_sq(128);
}


#[test]
fn matrix_256sq_test() {
    matrix_madd_n_sq(256);
}

#[test]
fn matrix_480_sq_test() {
    matrix_madd_n_sq(480);
}

#[test]
fn matrix_508_sq_test() {
    matrix_madd_n_sq(508);
}

#[test]
fn matrix_512_sq_test() {
    matrix_madd_n_sq(512);
}

#[test]
fn matrix_516_sq_test() {
    matrix_madd_n_sq(516);
}
*/

/*
#[test]
fn matrix_768sq_test() {
    matrix_madd_n_sq(768, 768);
}

#[test]
fn matrix_992_sq_test() {
    matrix_madd_n_sq(992, 992);
}

#[test]
fn matrix_1024sq_test() {
    matrix_madd_n_sq(1024, 1024);
}

#[test]
fn matrix_1408_sq_test() {
    matrix_madd_n_sq(1408, 1408);
}
 */

/*
#[test]
fn matrix_2048x1_test() {
    matrix_madd_nmp(1, 2048, 1);
}
*/
