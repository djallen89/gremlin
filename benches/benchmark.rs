#[macro_use]
extern crate criterion;
use gremlin::{random_array, matrix_madd};

use ndarray::Array;
use ndarray::linalg::general_mat_mul;

use criterion::{Benchmark, Criterion};

//#[bench]
fn benchdgemm(crit: &mut Criterion) {
    let n = 32;
    let m = 32;
    
    let a: Vec<f64> = random_array(n, m, -100.0, 100.0);
    let b = random_array(n, m, -100.0, 100.0);
    let mut c = random_array(n, m, -100.0, 100.0);

    let a_arr = Array::from_vec(a.clone()).into_shape((n, m)).unwrap();
    let b_arr = Array::from_vec(b.clone()).into_shape((n, m)).unwrap();
    let mut c_arr = Array::from_vec(c.clone()).into_shape((n, m)).unwrap();

    let bench_def = Benchmark::new(
        "my dgemm 32x32",
        move |bch| bch.iter(|| {
            matrix_madd(n, m, &a, &b, &mut c)
        }))
        .with_function("ndarray dgemm 32x32", move |bch| bch.iter(|| {
            general_mat_mul(1.0, &a_arr, &b_arr, 1.0, &mut c_arr)
        }))
        .sample_size(50);
    
    crit.bench("dgemm", bench_def);
}

criterion_group!(benches, benchdgemm);
criterion_main!(benches);
