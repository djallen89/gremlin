//use super::lib::random_array;
use crate::lib::matrix_madd;
use crate::lib::{matrix_madd_n_sq, matrix_madd_nmp};
use crate::lib::test_equality;

fn test_range(begin: usize, end: usize, func: &Fn(usize)) {
    for n in begin ..= end {
        func(n)
    }
}

#[test]
fn matrix_1to20_sq() {
    test_range(1, 20, &matrix_madd_n_sq)
}

#[test]
fn matrix_7x6_6xn() {
    test_range(2, 10, &|n| { matrix_madd_nmp(7,6,n) })
}

#[test]
fn matrix_28_to_36_sq() {
    test_range(28, 36, &matrix_madd_n_sq);
}

#[test]
fn matrix_60_to_68_sq() {
    test_range(60, 68, &matrix_madd_n_sq)
}

#[test]
fn matrix_124_to_132_sq() {
    test_range(124, 132, &matrix_madd_n_sq)
}

#[test]
fn matrix_252_to_260_sq_test() {
    test_range(252, 260, &matrix_madd_n_sq)
}

#[test]
fn matrix_480_sq_test() {
    matrix_madd_n_sq(480);
}


#[test]
fn matrix_508_sq_test() {
    matrix_madd_n_sq(508);
}
/*
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
