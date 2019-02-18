//#![feature(align_offset)]
extern crate ndarray;

pub mod lib;
#[cfg(test)]
mod test;
#[cfg(bench)]
mod benchmark;

use std::env;
use lib::{matrix_madd, random_array, float_eq};
use ndarray::Array;
use ndarray::linalg::general_mat_mul;

fn arg_to_int(arg: &str) -> usize {
    match arg.parse::<usize>() {
        Ok(x) => x,
        Err(f) => {
            panic!("{}\nExpected numeric input, got {}.",
                   f, arg);
        }
    }
}

pub fn matrix_madd_nmp(n: usize, m: usize, p: usize) {
    let a: Vec<f64> = random_array(n, p, -100.0, 100.0);
    let b = random_array(p, m, -100.0, 100.0);
    let mut c = random_array(n, m, -100.0, 100.0);
    
    let aarr = Array::from_vec(a.clone()).into_shape((n, p)).unwrap();
    let barr = Array::from_vec(b.clone()).into_shape((p, m)).unwrap();
    let mut carr = Array::from_vec(c.clone()).into_shape((n, m)).unwrap();

    general_mat_mul(1.0, &aarr, &barr, 1.0, &mut carr);
    let slice = carr.as_slice().unwrap();
    
    matrix_madd(n, m, p, &a, &b, &mut c);

    test_equality(n, m, &c, &slice);
}

fn test_equality(rows: usize, cols: usize, c: &[f64], correct: &[f64]) {
    for i in 0 .. rows * cols {
        if !float_eq(c[i], correct[i]) {
            if rows * cols <= 16 {
                for i in 0 .. rows * cols {
                    println!("{} != {}",  c[i], correct[i]);
                }
            }
            panic!("{}, {} != {}", i, c[i], correct[i]);
        }
        assert!(float_eq(c[i], correct[i]));
    }
    println!("Matrices are equal.");
}

fn main() {
    let mut n = 128;
    let mut m = 128;
    let mut p = 128;
    let args: Vec<String> = env::args().collect();
    match args.len() {
        1 => {},
        2 => {
            let d = arg_to_int(&args[1]);
            n = d;
            m = d;
            p = d;
        },
        3 => {
            let d1 = arg_to_int(&args[1]);
            let d2 = arg_to_int(&args[2]);
            n = d1;
            m = d2;
            p = d2;
        },
        4 => {
            let d1 = arg_to_int(&args[1]);
            let d2 = arg_to_int(&args[2]);
            let d3 = arg_to_int(&args[3]);
            n = d1;
            m = d2;
            p = d3;
        },
        x => {
            println!("Expected 0, 1, 2, or 3 arguments, got {}. Exiting.", x);
            return;
        }
    }
    for i in 0 .. 10 {
        matrix_madd_nmp(n, m, p);
    }
}

