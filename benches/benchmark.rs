#[macro_use]
extern crate criterion;
use gremlin_lib;

use gremlin_lib::{random_array, matrix_madd};
use ndarray::Array;
use ndarray::linalg::general_mat_mul;

use criterion::{Benchmark, Criterion};

fn benchnxn(crit: &mut Criterion, i: usize) {
    let n = i;
    let m = i;

    let my_name = format!("my dgemm {}sq", n);
    let other_name = format!("ndarray {}sq", n);
    
    let a: Vec<f64> = random_array(n, m, -100.0, 100.0);
    let b = random_array(n, m, -100.0, 100.0);
    let mut c = random_array(n, m, -100.0, 100.0);

    let aarr = Array::from_vec(a.clone()).into_shape((n, m)).unwrap();
    let barr = Array::from_vec(b.clone()).into_shape((n, m)).unwrap();
    let mut carr = Array::from_vec(c.clone()).into_shape((n, m)).unwrap();

    let bench_def = Benchmark::new(
        my_name, move |bch| bch.iter(|| {
            matrix_madd(n, m, n, &a, &b, &mut c)
        }))
        .with_function(other_name, move |bch| bch.iter(|| {
            general_mat_mul(1.0, &aarr, &barr, 1.0, &mut carr)
        }))
        .sample_size(10);
    
    crit.bench("dgemm", bench_def);
}

fn bench4x4(crit: &mut Criterion) {
    benchnxn(crit, 4);
}

fn bench16x16(crit: &mut Criterion) {
    benchnxn(crit, 16);
}

fn bench32x32(crit: &mut Criterion) {
    benchnxn(crit, 32);
}

fn bench256x256(crit: &mut Criterion) {
    benchnxn(crit, 256);
}

fn bench1024x1024(crit: &mut Criterion) {
    benchnxn(crit, 1024);
}

fn bench2048x1(crit: &mut Criterion) {
    let n = 1;
    let m = 1;
    let p = 2048;
    
    let my_name = format!("my dgemm {}x1", p);
    let other_name = format!("ndarray {}x1", p);

    let a: Vec<f64> = random_array(n, p, -100.0, 100.0);
    let b = random_array(p, m, -100.0, 100.0);
    let mut c = random_array(n, m, -100.0, 100.0);
    
    let aarr = Array::from_vec(a.clone()).into_shape((n, p)).unwrap();
    let barr = Array::from_vec(b.clone()).into_shape((p, m)).unwrap();
    let mut carr = Array::from_vec(c.clone()).into_shape((n, m)).unwrap();
    
    let bench_def = Benchmark::new(
        my_name, move |bch| bch.iter(|| {
            matrix_madd(n, m, p, &a, &b, &mut c);
        }))
        .with_function(other_name, move |bch| bch.iter(|| {
            general_mat_mul(1.0, &aarr, &barr, 1.0, &mut carr);
        }))
        .sample_size(10);
    
    crit.bench("dgemm", bench_def);
}

criterion_group!(benches, bench2048x1, bench4x4, bench16x16, bench32x32,
                 bench256x256, bench1024x1024);
criterion_main!(benches);

