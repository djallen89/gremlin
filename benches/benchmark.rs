#[macro_use]
extern crate criterion;
use gremlin_lib;

use gremlin_lib::{random_array, matrix_madd};
use ndarray::Array;
use ndarray::linalg::general_mat_mul;

use criterion::{Benchmark, Criterion};

fn bench_n_sq(crit: &mut Criterion, n: usize) {
    let my_name = format!("my dgemm {}sq", n);
    let other_name = format!("ndarray {}sq", n);
    
    let a: Vec<f64> = random_array(n, n, -10000.0, 10000.0);
    let b = random_array(n, n, -10000.0, 10000.0);
    let mut c = random_array(n, n, -10000.0, 10000.0);

    let aarr = Array::from_vec(a.clone()).into_shape((n, n)).unwrap();
    let barr = Array::from_vec(b.clone()).into_shape((n, n)).unwrap();
    let mut carr = Array::from_vec(c.clone()).into_shape((n, n)).unwrap();

    let bench_def;
    bench_def = Benchmark::new(
        my_name, move |bch| bch.iter(|| {
            matrix_madd(n, n, n, &a, &b, &mut c)
        }))
        .with_function(other_name, move |bch| bch.iter(|| {
            general_mat_mul(1.0, &aarr, &barr, 1.0, &mut carr)
        }))
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

macro_rules! bench_n_sq {
    ($name:ident, $n:expr) => {
        fn $name(crit: &mut Criterion) {
            bench_n_sq(crit, $n)
        }
    }
}

bench_n_sq!(bench_4_sq, 4);
criterion_group!(small_4x_matrices, bench_4_sq);
criterion_main!(small_4x_matrices);
/*
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
    bench_nmp(1, 2048, 1, crit);
}

fn bench_1xmx2048(crit: &mut Criterion) {
    let n = 1;
    let m = 1;
    let p = 1024;
    
    bench_nmp(n, m, p, crit);
}

fn bench_300_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 300);
}

fn bench_320_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 320);
}

fn bench_340_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 340);
}

fn bench_352_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 352);
}

fn bench_360_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 360);
}

fn bench_384_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 384);
}

fn bench_416_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 416);
}

fn bench_448_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 448);
}

fn bench_576_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 576);
}
    
fn bench_608_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 544);
}

fn bench_640_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 640);
}

fn bench_672_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 672);
}

fn bench_704_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 704);
}

fn bench_840_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 840);
}

fn bench_880_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 880);
}

fn bench_896_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 896);
}

fn bench_928_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 928);
}

fn bench_960_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 960);
}

fn bench_1088_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1088);
}

fn bench_1152_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1152);
}

fn bench_1184_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1184);
}

fn bench_1216_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1216);
}

fn bench_1248_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1248);
}

fn bench_1280_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1280);
}

fn bench_1312_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1312);
}

fn bench_1344_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1344);
}

fn bench_1376_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1376);
}
criterion_group!(others, bench_300_sq, bench_320_sq, bench_340_sq, bench_352_sq,
                 bench_360_sq, bench_384_sq, bench_416_sq, bench_448_sq,
                 bench_576_sq, bench_608_sq, bench_640_sq, bench_672_sq,
                 bench_704_sq, bench_840_sq, bench_880_sq, bench_896_sq,
                 bench_928_sq, bench_960_sq, bench_1088_sq, bench_1152_sq,
                 bench_1184_sq, bench_1216_sq, bench_1248_sq, bench_1280_sq,
                 bench_1312_sq, bench_1344_sq, bench_1376_sq);

fn bench_1000_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1000);
}

fn bench_1008_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1008);
}

fn bench_1016_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1016);
}

fn bench_1020_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1020);
}

fn bench_1021_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1021);
}

fn bench_1022_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1022);
}

fn bench_1023_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1023);
}

fn bench_1025_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1025);
}

fn bench_1026_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1026);
}

fn bench_1027_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1027);
}

fn bench_1028_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1028);
}

fn bench_1032_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1032);
}

fn bench_1036_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1036);
}

fn bench_1600_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1600);
}

fn bench_1800_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 1800);
}

fn bench_2000_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 2000);
}

fn bench_2200_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 2200);
}

fn bench_2400_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 2400);
}

fn bench_2600_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 2600);
}

fn bench_2800_sq(crit: &mut Criterion) {
    bench_n_sq(crit, 2800);
}

criterion_group!(thousand, bench_1000_sq, bench_1008_sq, bench_1016_sq,
                 bench_1020_sq, bench_1021_sq, bench_1022_sq, bench_1023_sq,
                 bench_1025_sq, bench_1026_sq, bench_1027_sq, bench_1028_sq,
                 bench_1032_sq, bench_1036_sq, bench_1376_sq);

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
/*
criterion_main!(small_non4_matrices, small_4x_matrices, mid_non4_matrices,
                mid_4x_matrices, big_4x_matrices, very_big_matrices,
                vectors);
 */
//criterion_main!(others);
criterion_group!(k68, bench_1600_sq, bench_1800_sq);
criterion_group!(t26, bench_1026_sq);
criterion_group!(twok, bench_2000_sq, bench_2200_sq, bench_2400_sq,
                 bench_2600_sq, bench_2800_sq);

criterion_main!(k68);
*/
