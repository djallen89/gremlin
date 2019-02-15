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
    let naive_name = format!("naive {}sq", n);
    
    let a: Vec<f64> = random_array(n, m, -100.0, 100.0);
    let b = random_array(n, m, -100.0, 100.0);
    let mut c = random_array(n, m, -100.0, 100.0);

    let aarr = Array::from_vec(a.clone()).into_shape((n, m)).unwrap();
    let barr = Array::from_vec(b.clone()).into_shape((n, m)).unwrap();
    let mut carr = Array::from_vec(c.clone()).into_shape((n, m)).unwrap();

    let bench_def;
    bench_def = Benchmark::new(
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

fn bench_224_sq(crit: &mut Criterion) {
    benchnxn(crit, 224);
}

fn bench_256_sq(crit: &mut Criterion) {
    benchnxn(crit, 256);
}

fn bench_288_sq(crit: &mut Criterion) {
    benchnxn(crit, 288);
}

fn bench_480_sq(crit: &mut Criterion) {
    benchnxn(crit, 480);
}

fn bench_512_sq(crit: &mut Criterion) {
    benchnxn(crit, 512);
}

fn bench_544_sq(crit: &mut Criterion) {
    benchnxn(crit, 544);
}

fn bench_736_sq(crit: &mut Criterion) {
    benchnxn(crit, 736);
}

fn bench_768_sq(crit: &mut Criterion) {
    benchnxn(crit, 768);
}

fn bench_800_sq(crit: &mut Criterion) {
    benchnxn(crit, 800);
}

fn bench_992_sq(crit: &mut Criterion) {
    benchnxn(crit, 992);
}

fn bench_1024_sq(crit: &mut Criterion) {
    benchnxn(crit, 1024);
}

fn bench_1056_sq(crit: &mut Criterion) {
    benchnxn(crit, 1056);
}

fn bench2048x1(crit: &mut Criterion) {
    let n = 1;
    let m = 2048;
    let p = 1;
    
    let my_name = format!("my dgemm {}x1", m);
    let other_name = format!("ndarray {}x1", m);

    let a: Vec<f64> = random_array(n, m, -100.0, 100.0);
    let b = random_array(m, p, -100.0, 100.0);
    let mut c = random_array(n, p, -100.0, 100.0);
    
    let aarr = Array::from_vec(a.clone()).into_shape((n, m)).unwrap();
    let barr = Array::from_vec(b.clone()).into_shape((m, p)).unwrap();
    let mut carr = Array::from_vec(c.clone()).into_shape((n, p)).unwrap();
    
    let bench_def = Benchmark::new(
        my_name, move |bch| bch.iter(|| {
            matrix_madd(n, m, p, &a, &b, &mut c);
        }))
        //.with_function(other_name, move |bch| bch.iter(|| {
            //general_mat_mul(1.0, &aarr, &barr, 1.0, &mut carr);
        //}))
        .sample_size(10);
    
    crit.bench("dgemm", bench_def);
}


//criterion_group!(benches, bench2048x1, bench4x4, bench16x16, bench32x32, bench256x256,
//                 bench_768_sq, bench1024x1024);
//criterion_group!(benches, bench_256_sq, bench_512_sq, bench_768_sq, bench_1024_sq);
//criterion_group!(benches, bench_1024_sq);
//criterion_group!(benches, bench_480_sq, bench_736_sq, bench_992_sq);
criterion_group!(benches, bench32x32, bench_224_sq, bench_256_sq, bench_288_sq,
                 bench_480_sq, bench_512_sq, bench_544_sq,
                 bench_736_sq, bench_768_sq, bench_800_sq,
                 bench_992_sq, bench_1056_sq);
criterion_main!(benches);

