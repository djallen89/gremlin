//#![feature(align_offset)]
extern crate ndarray;

pub mod lib;
#[cfg(bench)]
mod benchmark;

use std::env;
use lib::{matrix_madd_parallel, random_array, float_eq};
use ndarray::Array;
use ndarray::linalg::general_mat_mul;


/// Converts an argument to an unsigned integer.
fn arg_to_int(arg: &str) -> usize {
    match arg.parse::<usize>() {
        Ok(x) => x,
        Err(f) => {
            panic!("{}\nExpected numeric input, got {}.",
                   f, arg);
        }
    }
}

/// Generates 3 random arrays, a, b, and c, where a is n rows by p
/// columns, b is p rows by m columns, and c is n rows by m columns.
/// These arrays are then cloned into ndarray's Array type, and it
/// performs C = AB + C. After that, c is set equal to ab + c using
/// matrix_madd_parallel, and the two are compared for floating point
/// equality using test_equality.
pub fn matrix_madd_nmp(n: usize, m: usize, p: usize) {
    let a: Vec<f64> = random_array(n, p, -100.0, 100.0);
    let b = random_array(p, m, -100.0, 100.0);
    let mut c = random_array(n, m, -100.0, 100.0);
    
    let aarr = Array::from_vec(a.clone()).into_shape((n, p)).unwrap();
    let barr = Array::from_vec(b.clone()).into_shape((p, m)).unwrap();
    let mut carr = Array::from_vec(c.clone()).into_shape((n, m)).unwrap();

    general_mat_mul(1.0, &aarr, &barr, 1.0, &mut carr);
    let slice = carr.as_slice().unwrap();
    
    matrix_madd_parallel(8, n, m, p, &a, &b, &mut c);
    println!("Returned from matrix madd");
    test_equality(n, m, &c, &slice);
}

/// Compares two arrays for floating point equality of their elements.
fn test_equality(rows: usize, cols: usize, c: &[f64], correct: &[f64]) {
    let mut equal = true;
    let mut inequalities = String::new();
    let mut count = 0;
    for i in 0 .. rows {
        println!("{}", i);
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

/// gremlin can be called with no arguments to run with m, n, and p
/// equal to 128.  Alternatively, if 1 argument is given, m, n, and p
/// will be equal to that size; if two arguments are given, n will be
/// equal to the first argument, and m and p will be equal to the
/// second argument; if three arguments given, n will be assigned the
/// first, m the second, and p the third.
pub fn main() {
    let default = 128;
    let mut n = default; 
    let mut m = default;
    let mut p = default;
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

