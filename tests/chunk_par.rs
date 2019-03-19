use gremlin_lib::{matrix_madd_n_sq_chunked, matrix_madd_nmp_parallel};

fn test_range(begin: usize, end: usize, func: &Fn(usize)) {
    for n in begin ..= end {
        func(n)
    }
}

#[test]
fn matrix_60_to_68_sq_chunked() {
    test_range(60, 68, &|n| {
        println!("Iteration: {}", n);
        matrix_madd_n_sq_chunked(n);
    });
}

#[test]
fn matrix_480_sq_test_parallel() {
    matrix_madd_n_sq_chunked(480);
}

#[test]
fn matrix_507_sq_test_parallel() {
    matrix_madd_n_sq_chunked(507);
}

#[test]
fn matrix_512_sq_test_parallel() {
    matrix_madd_n_sq_chunked(512);
}

#[test]
fn matrix_613_sq_test_parallel() {
    matrix_madd_n_sq_chunked(613);
}

#[test]
fn matrix_1200_test_parallel() {
    matrix_madd_n_sq_chunked(1200);
}

#[test]
fn matrix_2400_test_parallel() {
    matrix_madd_n_sq_chunked(2400);
}


