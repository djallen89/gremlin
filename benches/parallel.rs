#[macro_use]
extern crate criterion;
extern crate num_cpus;
extern crate ndarray;

use gremlin_lib;
use gremlin_lib::{random_array, matrix_madd_parallel};
use ndarray::Array;
use ndarray::linalg::general_mat_mul;

use criterion::{Benchmark, Criterion};

fn bench_n_sq(crit: &mut Criterion, n: usize) {
    let my_name = format!("{:04}sq", n);
    //let other_name = format!("ndarray {}sq", n);
    
    let a: Vec<f64> = random_array(n, n, -10000.0, 10000.0);
    let b = random_array(n, n, -10000.0, 10000.0);
    let mut c = random_array(n, n, -10000.0, 10000.0);

    //let aarr = Array::from_vec(a.clone()).into_shape((n, n)).unwrap();
    //let barr = Array::from_vec(b.clone()).into_shape((n, n)).unwrap();
    //let mut carr = Array::from_vec(c.clone()).into_shape((n, n)).unwrap();
    let threads = num_cpus::get_physical();
    let samples = if n < 1200 {
        20
    } else {
        10
    };

    let bench_def;
    bench_def = Benchmark::new(
        my_name, move |bch| bch.iter(|| {
            matrix_madd_parallel(threads, n, n, n, &a, &b, &mut c)
        }))
        //.with_function(other_name, move |bch| bch.iter(|| {
            //general_mat_mul(1.0, &aarr, &barr, 1.0, &mut carr)
        //}))
        .sample_size(samples);
    
    crit.bench("final_dgemm_parallel", bench_def);
}

fn bench_nmp(n: usize, m: usize, p: usize, crit: &mut Criterion) {
    
    let my_name = format!("my dgemm parallel {}x{}x{}", n, m, p);
    let other_name = format!("ndarray parallel {}x{}x{}", n, m, p);

    let a: Vec<f64> = random_array(n, m, -100.0, 100.0);
    let b = random_array(m, p, -100.0, 100.0);
    let mut c = random_array(n, p, -100.0, 100.0);
    
    let aarr = Array::from_vec(a.clone()).into_shape((n, m)).unwrap();
    let barr = Array::from_vec(b.clone()).into_shape((m, p)).unwrap();
    let mut carr = Array::from_vec(c.clone()).into_shape((n, p)).unwrap();

    let threads = num_cpus::get_physical();
    
    let bench_def = Benchmark::new(
        my_name, move |bch| bch.iter(|| {
            matrix_madd_parallel(threads, n, m, p, &a, &b, &mut c);
        }))
        .with_function(other_name, move |bch| bch.iter(|| {
            general_mat_mul(1.0, &aarr, &barr, 1.0, &mut carr);
        }))
        .sample_size(10);
    
    crit.bench("dgemm", bench_def);
}

macro_rules! bench_num_sq {    
    ($name:ident) => {
        fn $name(crit: &mut Criterion) {
            let words = stringify!($name).split("_").collect::<Vec<&str>>();
            let n = words[1].parse::<usize>().unwrap();
            bench_n_sq(crit, n)
        }
    }
}

bench_num_sq!(bench_60_sq);
bench_num_sq!(bench_64_sq);
bench_num_sq!(bench_68_sq);
bench_num_sq!(bench_80_sq);
bench_num_sq!(bench_96_sq);
bench_num_sq!(bench_100_sq);
bench_num_sq!(bench_124_sq);
criterion_group!(small_4x_matrices,
                 bench_60_sq, bench_64_sq, bench_68_sq, bench_80_sq,
                 bench_96_sq, bench_100_sq, bench_124_sq);

bench_num_sq!(bench_126_sq);
bench_num_sq!(bench_127_sq);
bench_num_sq!(bench_129_sq);
bench_num_sq!(bench_130_sq);
criterion_group!(mid_non4_matrices, bench_126_sq, bench_127_sq,
                 bench_129_sq, bench_130_sq);

bench_num_sq!(bench_128_sq);
bench_num_sq!(bench_132_sq);
bench_num_sq!(bench_136_sq);
bench_num_sq!(bench_180_sq);
bench_num_sq!(bench_224_sq);
bench_num_sq!(bench_256_sq);
bench_num_sq!(bench_288_sq);
bench_num_sq!(bench_300_sq);
bench_num_sq!(bench_320_sq);
bench_num_sq!(bench_340_sq);
bench_num_sq!(bench_352_sq);
bench_num_sq!(bench_360_sq);
bench_num_sq!(bench_384_sq);
bench_num_sq!(bench_416_sq);
bench_num_sq!(bench_448_sq);
bench_num_sq!(bench_480_sq);
criterion_group!(mid_4x_matrices, bench_124_sq, bench_128_sq,
                 bench_132_sq, bench_136_sq, bench_180_sq,
                 bench_224_sq, bench_256_sq, bench_288_sq,
                 bench_300_sq, bench_320_sq, bench_340_sq,
                 bench_352_sq, bench_360_sq, bench_384_sq,
                 bench_416_sq, bench_448_sq, bench_480_sq);

bench_num_sq!(bench_508_sq);
bench_num_sq!(bench_512_sq);
bench_num_sq!(bench_516_sq);
bench_num_sq!(bench_544_sq);
bench_num_sq!(bench_576_sq);
bench_num_sq!(bench_608_sq);
bench_num_sq!(bench_640_sq);
bench_num_sq!(bench_672_sq);
bench_num_sq!(bench_704_sq);
bench_num_sq!(bench_736_sq);
bench_num_sq!(bench_768_sq);
bench_num_sq!(bench_800_sq);
bench_num_sq!(bench_840_sq);
bench_num_sq!(bench_880_sq);
bench_num_sq!(bench_896_sq);
bench_num_sq!(bench_928_sq);
bench_num_sq!(bench_960_sq);
bench_num_sq!(bench_992_sq);
criterion_group!(big_4x_matrices,
                 bench_508_sq, bench_512_sq, bench_516_sq, bench_544_sq,
                 bench_576_sq, bench_608_sq, bench_640_sq, bench_672_sq,
                 bench_704_sq, bench_736_sq, bench_768_sq,
                 bench_800_sq, bench_840_sq, bench_880_sq, bench_896_sq,
                 bench_928_sq, bench_960_sq, bench_992_sq);

bench_num_sq!(bench_1016_sq);
bench_num_sq!(bench_1020_sq);
bench_num_sq!(bench_1022_sq);
bench_num_sq!(bench_1023_sq);
bench_num_sq!(bench_1024_sq);
bench_num_sq!(bench_1025_sq);
bench_num_sq!(bench_1026_sq);
bench_num_sq!(bench_1028_sq);
bench_num_sq!(bench_1032_sq);
bench_num_sq!(bench_1036_sq);
bench_num_sq!(bench_1056_sq);

criterion_group!(very_big_matrices,
                 bench_1016_sq,
                 bench_1020_sq,
                 bench_1022_sq, bench_1023_sq,
                 bench_1024_sq,
                 bench_1025_sq, bench_1026_sq,
                 bench_1028_sq,
                 bench_1032_sq, bench_1036_sq, bench_1056_sq);

bench_num_sq!(bench_1152_sq);
bench_num_sq!(bench_1248_sq);
bench_num_sq!(bench_1280_sq);
bench_num_sq!(bench_1312_sq);
bench_num_sq!(bench_1408_sq);
bench_num_sq!(bench_1504_sq);
bench_num_sq!(bench_1536_sq);
bench_num_sq!(bench_1568_sq);
bench_num_sq!(bench_1760_sq);
bench_num_sq!(bench_1888_sq);
bench_num_sq!(bench_2016_sq);
bench_num_sq!(bench_2048_sq);
bench_num_sq!(bench_2080_sq);
bench_num_sq!(bench_2208_sq);
bench_num_sq!(bench_2336_sq);
bench_num_sq!(bench_2528_sq);
bench_num_sq!(bench_2560_sq);
bench_num_sq!(bench_2592_sq);
bench_num_sq!(bench_2720_sq);
bench_num_sq!(bench_2912_sq);

/*
fn bench_1492_1150_1201(crit: &mut Criterion) {
    bench_nmp(1492, 1150, 1201, crit);
}
*/

criterion_group!(huge_matrices,
                 bench_1152_sq,
                 bench_1248_sq,
                 bench_1280_sq,
                 bench_1312_sq,
                 bench_1408_sq,
                 bench_1504_sq,
                 bench_1536_sq,
                 bench_1568_sq,
                 bench_1760_sq,
                 bench_1888_sq,
                 bench_2016_sq,
                 bench_2048_sq,
                 bench_2080_sq,
                 bench_2208_sq,
                 bench_2336_sq,
                 bench_2528_sq,
                 bench_2560_sq,
                 bench_2592_sq,
                 bench_2720_sq,
                 bench_2912_sq);

bench_num_sq!(bench_3040_sq);
bench_num_sq!(bench_3072_sq);
bench_num_sq!(bench_3104_sq);
bench_num_sq!(bench_3328_sq);
bench_num_sq!(bench_3520_sq);
bench_num_sq!(bench_3584_sq);
bench_num_sq!(bench_3648_sq);
bench_num_sq!(bench_3776_sq);
bench_num_sq!(bench_3968_sq);
bench_num_sq!(bench_4032_sq);
bench_num_sq!(bench_4096_sq);
bench_num_sq!(bench_4160_sq);
bench_num_sq!(bench_4256_sq);
bench_num_sq!(bench_4448_sq);
bench_num_sq!(bench_4544_sq);
bench_num_sq!(bench_4608_sq);
bench_num_sq!(bench_4744_sq);
bench_num_sq!(bench_4936_sq);
bench_num_sq!(bench_4992_sq);
bench_num_sq!(bench_5120_sq);
bench_num_sq!(bench_5248_sq);
criterion_group!(gigantic,
                 bench_3040_sq,
                 bench_3072_sq,
                 bench_3104_sq,
                 bench_3328_sq,
                 bench_3520_sq,
                 bench_3584_sq,
                 bench_3648_sq,
                 bench_3776_sq,
                 bench_3968_sq,
                 bench_4032_sq,
                 bench_4096_sq,
                 bench_4160_sq,
                 bench_4256_sq,
                 bench_4448_sq,
                 bench_4544_sq,
                 bench_4608_sq,
                 bench_4744_sq,
                 bench_4936_sq,
                 bench_4992_sq,
                 bench_5120_sq,
                 bench_5248_sq);

bench_num_sq!(bench_252_sq);
bench_num_sq!(bench_254_sq);
bench_num_sq!(bench_255_sq);
bench_num_sq!(bench_257_sq);
bench_num_sq!(bench_258_sq);
bench_num_sq!(bench_260_sq);
bench_num_sq!(bench_510_sq);
bench_num_sq!(bench_511_sq);
bench_num_sq!(bench_513_sq);
bench_num_sq!(bench_514_sq);
bench_num_sq!(bench_764_sq);
bench_num_sq!(bench_766_sq);
bench_num_sq!(bench_767_sq);
bench_num_sq!(bench_769_sq);
bench_num_sq!(bench_770_sq);
bench_num_sq!(bench_772_sq);

criterion_group!(hot_spots,
                 bench_124_sq, bench_126_sq, bench_127_sq,
                 bench_128_sq,
                 bench_129_sq, bench_130_sq, bench_132_sq,
                 
                 bench_252_sq, bench_254_sq, bench_255_sq,
                 bench_256_sq,
                 bench_257_sq, bench_258_sq, bench_260_sq,
                 
                 bench_508_sq,
                 bench_510_sq, bench_511_sq,
                 bench_512_sq,
                 bench_513_sq, bench_514_sq,
                 bench_516_sq,

                 bench_764_sq,
                 bench_766_sq, bench_767_sq,
                 bench_768_sq,
                 bench_769_sq, bench_770_sq,
                 bench_772_sq,
                 
                 bench_1020_sq,
                 bench_1022_sq, bench_1023_sq,
                 bench_1024_sq,
                 bench_1025_sq, bench_1026_sq,
                 bench_1028_sq,

                 bench_1504_sq,
                 bench_1536_sq,
                 bench_1568_sq,

                 bench_2016_sq,
                 bench_2048_sq,
                 bench_2080_sq,

                 bench_2528_sq,
                 bench_2560_sq,
                 bench_2592_sq,

                 bench_3040_sq,
                 bench_3072_sq,
                 bench_3104_sq,

                 bench_3520_sq,
                 bench_3584_sq,
                 bench_3648_sq,

                 bench_4032_sq,
                 bench_4096_sq,
                 bench_4160_sq);

//criterion_main!(hot_spots);
bench_num_sq!(bench_10496_sq);
criterion_group!(omega,
                 bench_4032_sq,
                 bench_4096_sq,
                 bench_4160_sq,
                 bench_4256_sq,
                 bench_4448_sq,
                 bench_4544_sq,
                 bench_4608_sq,
                 bench_4744_sq,
                 bench_4936_sq,
                 bench_4992_sq,
                 bench_5120_sq,
                 bench_5248_sq,
                 bench_10496_sq,
);
criterion_main!(omega);
/*
(/ 19111600057.906113 1e9) 19.111600057906113
(/ 1153.9 55) 20.98
(* (/ 19.1116 20.98) 100) 91.09437559580552
(/ 64.147 55) 1.166309090909091
(/ 88.909 55) 1.6165272727272728
(/ 70.967 55) 1.2903090909090909
 */

/*
criterion_main!(
    small_4x_matrices,
    mid_4x_matrices,
    hot_spots,
    big_4x_matrices,
    very_big_matrices,
    huge_matrices,
    gigantic);
*/
