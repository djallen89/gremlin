#[macro_use]
extern crate criterion;

use gremlin::{random_array, matrix_madd};
//use ndarray::Array;
//use ndarray::linalg::general_mat_mul;
use matrixmultiply::dgemm;

use criterion::{Benchmark, Criterion};

//#[bench]
fn benchdgemm(crit: &mut Criterion) {
    let n = 1024;
    let m = 1024;
    
    let a: Vec<f64> = random_array(n, m, -100.0, 100.0);
    let b = random_array(n, m, -100.0, 100.0);
    let mut c = random_array(n, m, -100.0, 100.0);

    let adgm = a.clone();
    let bdgm = b.clone();
    let mut cdgm = c.clone();

    let bench_def = Benchmark::new(
        "my dgemm 1024sq",
        move |bch| bch.iter(|| {
            matrix_madd(n, m, &a, &b, &mut c)
        }))
        .with_function("matrixmultiply dgemm 1024 sq", move |bch| bch.iter(|| unsafe {
            dgemm(1024, 1024, 1024, 1.0,
                  &adgm[0] as *const f64, 128, 32,
                  &bdgm[0] as *const f64, 32, 128,
                  1.0, &mut cdgm[0] as *mut f64, 32, 32)
        }))
        .sample_size(10);
    
    crit.bench("dgemm", bench_def);
}

criterion_group!(benches, benchdgemm);
criterion_main!(benches);
