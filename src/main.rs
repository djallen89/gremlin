#![feature(align_offset)]

extern crate ndarray;

pub mod lib;

#[cfg(test)]
mod test;

pub use lib::random_array;
pub use lib::matrix_madd;
pub use lib::minimatrix_fmadd64;
pub use lib::floateq;
use ndarray::Array;
use ndarray::linalg::general_mat_mul;

#[cfg(bench)]
mod benchmark;
    
fn main() {
    /*
    for _i in 0 .. 10 {
        let a = random_array(4, 4, -100.0, 100.0);
        let b = random_array(4, 4, -100.0, 100.0);
        let mut c = random_array(4, 4, -100.0, 100.0);

        let aarr = Array::from_vec(a.clone()).into_shape((4, 4)).unwrap();
        let barr = Array::from_vec(b.clone()).into_shape((4, 4)).unwrap();
        let mut carr = Array::from_vec(c.clone()).into_shape((4, 4)).unwrap();

        general_mat_mul(1.0, &aarr, &barr, 1.0, &mut carr);

        let slice = carr.as_slice().unwrap();
        minimatrix_fmadd64(4, 4, &a, &b, &mut c);
        for i in 0 .. 16 {
            if !floateq(slice[i], c[i]) {
                println!("{}: {} != {}", i, slice[i], c[i]);
            }
        }
    }
     */

    let n = 8;
    let a = random_array(n, n, -100.0, 100.0);
    let b = random_array(n, n, -100.0, 100.0);
    let mut c = random_array(n, n, -100.0, 100.0);

    let aarr = Array::from_vec(a.clone()).into_shape((n, n)).unwrap();
    let barr = Array::from_vec(b.clone()).into_shape((n, n)).unwrap();
    let mut carr = Array::from_vec(c.clone()).into_shape((n, n)).unwrap();

    general_mat_mul(1.0, &aarr, &barr, 1.0, &mut carr);

    let slice = carr.as_slice().unwrap();
    matrix_madd(n, n, &a, &b, &mut c);
    for i in 0 .. n * n {
        if !floateq(slice[i], c[i]) {
            println!("{}: {} != {}", i, slice[i], c[i]);
        }
    }

}
