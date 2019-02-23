//#![feature(align_offset)]
extern crate ndarray;

pub mod lib;
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
    println!("Returned from matrix madd");
    test_equality(n, m, &c, &slice);
}

fn test_equality(rows: usize, cols: usize, c: &[f64], correct: &[f64]) {
    let mut equal = true;
    let mut inequalities = String::new();
    let mut count = 0;
    for i in 0 .. rows {
        for j in 0 .. cols {
            if !float_eq(c[i * cols + j], correct[i * cols + j]) {
                equal = false;
                if rows * cols < 128 {
                    inequalities = format!("{}\n{},{}: {} !={}", inequalities,
                                           i + 1, j + 1, c[i], correct[i]);
                }
                count += 1;
            }
        }
    }
    
    if equal {
        println!("Matrices are equal.");
    } else if !equal && rows * cols <= 128 {
        println!("Matrices are inequal");
        println!("{}", inequalities);
    } else {
        println!("Matrices are inequal. {} inequalities", count);
    }
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

    matrix_madd_nmp(n, m, p);
}

