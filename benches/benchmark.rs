#[macro_use]
extern crate criterion;

use gremlin::{random_array, matrix_madd};
use ndarray::Array;
use ndarray::linalg::general_mat_mul;

use criterion::{Benchmark, Criterion};

//#[bench]
fn benchdgemm(crit: &mut Criterion) {
    let n = 4*256;
    let m = 4*256;

    let my_name = format!("my dgemm {}sq", n);
    let other_name = format!("ndarray {}sq", n);
    
    let a: Vec<f64> = random_array(n, m, -100.0, 100.0);
    let b = random_array(n, m, -100.0, 100.0);
    let mut c = random_array(n, m, -100.0, 100.0);

    let aarr = Array::from_vec(a.clone()).into_shape((n, m)).unwrap();
    let barr = Array::from_vec(b.clone()).into_shape((n, m)).unwrap();
    let mut carr = Array::from_vec(c.clone()).into_shape((4, 4)).unwrap();

    let bench_def = Benchmark::new(
        &my_name, move |bch| bch.iter(|| {
            matrix_madd(n, m, &a, &b, &mut c)
        }))
        .with_function(&other_name, move |bch| bch.iter(|| unsafe {
            general_mat_mull(1.0, &aarr, &barr, 1.0, &mut carr)
        }))
        .sample_size(10);
    
    crit.bench("dgemm", bench_def);
}

criterion_group!(benches, benchdgemm);
criterion_main!(benches);
