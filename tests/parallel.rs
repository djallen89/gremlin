use gremlin_lib::{matrix_madd_n_sq_parallel, matrix_madd_nmp_parallel};

fn test_range(begin: usize, end: usize, func: &Fn(usize)) {
    for n in begin ..= end {
        func(n)
    }
}

#[test]
fn matrix_1xn_test_parallel() {
    test_range(1, 512, &|n| { matrix_madd_nmp_parallel(1, n, 1) })
}

/*
#[test]
fn matrix_1x20xp_test_parallel() {
    test_range(1, 512, &|p| { matrix_madd_nmp_parallel(1, 20, p) })
}
*/

#[test]
fn matrix_20xmx1_test_parallel() {
    test_range(2, 32, &|m| { matrix_madd_nmp_parallel(20, m, 1) });
}

#[test]
fn matrix_40xmx2_test_parallel() {
    test_range(2, 64, &|m| { matrix_madd_nmp_parallel(40, m, 2) });
}

#[test]
fn matrix_1to20_sq_parallel() {
    test_range(1, 20, &|n| {
        println!("n = {}", n);
        matrix_madd_n_sq_parallel(n)
    })
}

#[test]
fn matrix_7x6_6xn_parallel() {
    test_range(1, 10, &|n| {
        println!("{}", n);
        matrix_madd_nmp_parallel(7,6,n)
    })
}

#[test]
fn matrix_nmp_small_parallel() {
    for n in 20 .. 35 {
        for m in 20 .. 35 {
            test_range(1, 35, &|p| {
                matrix_madd_nmp_parallel(n, m, p)
            })
        }
    }
}

#[test]
fn matrix_28_to_36_sq_parallel() {
    test_range(28, 36, &|n| {
        println!("{}", n);
        matrix_madd_n_sq_parallel(n);
    })
}

#[test]
fn matrix_28_40_56_parallel() {
    matrix_madd_nmp_parallel(28, 40, 56)
}

#[test]
fn matrix_60_to_68_sq_parallel() {
    test_range(60, 68, &matrix_madd_n_sq_parallel);
}

#[test]
fn matrix_124_to_132_sq_parallel() {
    test_range(1, 4, &|n| {
        println!("{}", 120 + n * 4);
        matrix_madd_n_sq_parallel(120 + n * 4)
    })
}

#[test]
fn matrix_252_to_260_sq_test_parallel() {
    test_range(252, 260, &matrix_madd_n_sq_parallel)
}

#[test]
fn matrix_480_sq_test_parallel() {
    matrix_madd_n_sq_parallel(480);
}

#[test]
fn matrix_507_sq_test_parallel() {
    matrix_madd_n_sq_parallel(507);
}

#[test]
fn matrix_508_sq_test_parallel() {
    matrix_madd_n_sq_parallel(508);
}

#[test]
fn matrix_512_sq_test_parallel() {
    matrix_madd_n_sq_parallel(512);
}

#[test]
fn matrix_516_sq_test_parallel() {
    matrix_madd_n_sq_parallel(516);
}

#[test]
fn matrix_613_sq_test_parallel() {
    matrix_madd_n_sq_parallel(613);
}

#[test]
fn matrix_768sq_test_parallel() {
    matrix_madd_n_sq_parallel(768);
}

#[test]
fn bench_1492_1150_1201_parallel() {
    matrix_madd_nmp_parallel(1492, 1150, 1201)
}

