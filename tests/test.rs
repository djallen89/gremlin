use gremlin_lib::{matrix_madd, matrix_multithread};
use gremlin_lib::{matrix_madd_n_sq, matrix_madd_nmp};
use gremlin_lib::test_equality;

fn test_range(begin: usize, end: usize, func: &Fn(usize)) {
    for n in begin ..= end {
        func(n)
    }
}

#[test]
fn matrix_1xn_test() {
    test_range(1, 512, &|n| { matrix_madd_nmp(1, n, 1) })
}

#[test]
fn matrix_1x20xp_test() {
    test_range(1, 512, &|p| { matrix_madd_nmp(1, 20, p) })
}

#[test]
fn matrix_20xmx1_test() {
    test_range(2, 32, &|m| { matrix_madd_nmp(20, m, 1) });
}

#[test]
fn matrix_40xmx2_test() {
    test_range(2, 64, &|m| { matrix_madd_nmp(40, m, 2) });
}

#[test]
fn matrix_1to16_sq() {
    test_range(1, 16, &|n| {
        matrix_madd_n_sq(n * 32)
    })
}

#[test]
fn matrix_7x6_6xn() {
    test_range(2, 10, &|n| {
        println!("{}", n);
        matrix_madd_nmp(7,6,n)
    })
}

#[test]
fn matrix_512x1024x1024() {
    matrix_madd_nmp(512, 1024, 1024)
}

/*
#[test]
fn matrix_nmp_small() {
    for n in 1 .. 64 {
        for m in 1 .. 64 {
            test_range(1, 64, &|p| {
                println!("{},{},{}",n,m,p);
                matrix_madd_nmp(n, m, p)
            })
        }
    }
}
*/
/*
#[test]
fn matrix_8_1_p() {
    test_range(4, 10, &|p| {
        matrix_madd_nmp(8, 1, p)
    })
}
*/
#[test]
fn matrix_28_to_36_sq() {
    test_range(28, 36, &matrix_madd_n_sq);
}

/*
#[test]
fn matrix_60_to_68_sq() {
    test_range(60, 68, &matrix_madd_n_sq);
    //matrix_madd_n_sq(64)
}

#[test]
fn matrix_124_to_132_sq() {
    test_range(124, 132, &|n| {
        println!("{}", n);
        matrix_madd_n_sq(n)
    })
}
*/
/*
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
*/
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
 */
/*
#[test]
fn matrix_1024sq_test() {
    matrix_madd_n_sq(1024);
}
 */
#[test]
fn matrix_2t128() {
    matrix_multithread(2, 128);
}

#[test]
fn matrix_4t128() {
    matrix_multithread(4, 128);
}

#[test]
fn matrix_8t128() {
    matrix_multithread(8, 128);
}

#[test]
fn matrix_2t256() {
    matrix_multithread(2, 256);
}

#[test]
fn matrix_4t256() {
    matrix_multithread(4, 256);
}

#[test]
fn matrix_8t256() {
    matrix_multithread(8, 256);
}

#[test]
fn matrix_2t512() {
    matrix_multithread(2, 512);
}

#[test]
fn matrix_4t512() {
    matrix_multithread(4, 512);
}

#[test]
fn matrix_8t512() {
    matrix_multithread(8, 512);
}

#[test]
fn matrix_2t1024() {
    matrix_multithread(2, 1024);
}

#[test]
fn matrix_4t1024() {
    matrix_multithread(4, 1024);
}

#[test]
fn matrix_8t1024() {
    matrix_multithread(8, 1024);
}
/*
#[test]
fn matrix_1408_sq_test() {
    matrix_madd_n_sq(1408, 1408);
}
 */


