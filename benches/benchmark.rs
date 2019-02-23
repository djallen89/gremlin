#[macro_use]
extern crate criterion;
use gremlin_lib;

use gremlin_lib::{random_array, matrix_madd};
use ndarray::Array;
use ndarray::linalg::general_mat_mul;

use criterion::{Benchmark, Criterion};

fn bench_n_sq(crit: &mut Criterion, i: usize) {
    let n = i;
    let m = i;

    let my_name = format!("my dgemm {}sq", n);
    //let other_name = format!("ndarray {}sq", n);
    //let naive_name = format!("naive {}sq", n);
    
    let a: Vec<f64> = random_array(n, m, -100.0, 100.0);
    let b = random_array(n, m, -100.0, 100.0);
    let mut c = random_array(n, m, -100.0, 100.0);

    //let aarr = Array::from_vec(a.clone()).into_shape((n, m)).unwrap();
    //let barr = Array::from_vec(b.clone()).into_shape((n, m)).unwrap();
    //let mut carr = Array::from_vec(c.clone()).into_shape((n, m)).unwrap();

    let bench_def;
    bench_def = Benchmark::new(
        my_name, move |bch| bch.iter(|| {
            matrix_madd(n, m, n, &a, &b, &mut c)
        }))
        //.with_function(other_name, move |bch| bch.iter(|| {
            //general_mat_mul(1.0, &aarr, &barr, 1.0, &mut carr)
        //}))
        .sample_size(10);
    
    crit.bench("dgemm", bench_def);
}

fn bench_nmp(n: usize, m: usize, p: usize, crit: &mut Criterion) {
    
    let my_name = format!("my dgemm {}x{}x{}", n, m, p);
    let other_name = format!("ndarray {}x{}x{}", n, m, p);

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
        .with_function(other_name, move |bch| bch.iter(|| {
            general_mat_mul(1.0, &aarr, &barr, 1.0, &mut carr);
        }))
        .sample_size(10);
    
    crit.bench("dgemm", bench_def);
}

fn bench4x4(crit: &mut Criterion) {
    bench_n_sq(crit, 4);
}

fn bench_12_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 12);
}

fn bench_16_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 16);
}

fn bench_20_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 20);
}

fn bench_27_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 27);
}

fn bench_28_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 28);
}

fn bench_29_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 29);
}

fn bench_30_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 30);
}

fn bench_31_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 31);
}

fn bench_32_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 32);
}

fn bench_33_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 33);
}

fn bench_34_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 34);
}

fn bench_35_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 35);
}

fn bench_36_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 36);
}

fn bench_40_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 40);
}

fn bench_44_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 44);
}

fn bench_52_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 52);
}

fn bench_56_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 56);
}

fn bench_60_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 60);
}

fn bench_64_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 64);
}

fn bench_68_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 68);
}

fn bench_80_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 80);
}

fn bench_124_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 124);
}

fn bench_126_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 126);
}

fn bench_127_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 127);
}

fn bench_128_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 128);
}

fn bench_129_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 129);
}

fn bench_130_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 130);
}

fn bench_132_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 132);
}

fn bench_136_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 136);
}

fn bench_180_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 180);
}

fn bench_224_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 224);
}

fn bench_256_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 256);
}

fn bench_288_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 288);
}

fn bench_480_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 480);
}

fn bench_508_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 508);
}

fn bench_512_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 512);
}

fn bench_516_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 516);
}

fn bench_544_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 544);
}

fn bench_736_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 736);
}

fn bench_768_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 768);
}

fn bench_800_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 800);
}

fn bench_992_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 992);
}

fn bench_1024_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1024);
}

fn bench_1056_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1056);
}

fn bench_1408_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1408);
}

fn bench_1492_1150_1201(crit: &mut Criterion) {
    bench_nmp(1492, 1150, 1201, crit)
}

fn bench_2048x1(crit: &mut Criterion) {
    let n = 1;
    let m = 2048;
    let p = 1;
    
    bench_nmp(n, m, p, crit);
}

fn bench_1xmx2048(crit: &mut Criterion) {
    let n = 1;
    let m = 1;
    let p = 1024;
    
    bench_nmp(n, m, p, crit);
}

criterion_group!(small_non4_matrices, bench_27_sq, bench_29_sq, bench_30_sq,
                 bench_31_sq, bench_33_sq, bench_34_sq, bench_35_sq);
criterion_group!(small_4x_matrices, bench4x4, bench_12_sq, bench_16_sq,
                 bench_20_sq, bench_28_sq, bench_32_sq, bench_36_sq,
                 bench_40_sq, bench_44_sq, bench_52_sq, bench_56_sq,
                 bench_60_sq, bench_64_sq, bench_68_sq, bench_80_sq,
                 bench_124_sq, bench_128_sq);
criterion_group!(mid_non4_matrices, bench_126_sq, bench_127_sq,
                 bench_129_sq, bench_130_sq);
criterion_group!(mid_4x_matrices, bench_132_sq, bench_136_sq,
                 bench_180_sq, bench_224_sq, bench_256_sq,
                 bench_288_sq, bench_480_sq);
criterion_group!(big_4x_matrices, bench_508_sq, bench_512_sq,
                 bench_516_sq, bench_544_sq);
criterion_group!(very_big_matrices, bench_736_sq, bench_768_sq,
                 bench_800_sq, bench_992_sq, bench_1024_sq,
                 bench_1056_sq, bench_1408_sq, bench_1492_1150_1201);                 
criterion_group!(vectors, bench_2048x1, bench_1xmx2048);
criterion_group!(various_4x, bench_32_sq, bench_64_sq, bench_128_sq,
                 bench_256_sq, bench_512_sq, bench_768_sq, bench_1024_sq);

criterion_main!(small_non4_matrices, small_4x_matrices, mid_non4_matrices,
                mid_4x_matrices, big_4x_matrices, very_big_matrices,
                vectors, various_4x);

//criterion_main!(very_big_matrices, big_4x_matrices, mid_4x_matrices, 
//                vectors, various_4x);
//criterion_main!(various_4x);
//criterion_main!(small_4x_matrices);
