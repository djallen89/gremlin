extern crate rand;

use rand::prelude::*;
use std::ops::Range;

#[inline(always)]
pub fn get_elt(row: usize, col: usize, n_cols: usize) -> usize {
    row * n_cols + col
}

pub fn floateq(a: f64, b: f64) -> bool {
    use std::f64;
    
    let abs_a = a.abs();
    let abs_b = b.abs();
    let diff = (a - b).abs();
    let epsilon = 0.00001;

    if a == b {
	true
    } else if a == 0.0 || b == 0.0 || diff < f64::MIN_POSITIVE {
	diff < (epsilon * f64::MIN_POSITIVE)
    } else { 
	(diff / f64::min(abs_a + abs_b, f64::MAX)) < epsilon
    }

}

pub fn random_array<T>(cols: usize, rows: usize, low: T, high: T) -> Vec<T>
    where T: rand::distributions::uniform::SampleUniform
{
    use rand::distributions::Uniform;
    use std::usize;
    
    assert!(usize::MAX / rows > cols);
    
    let interval = Uniform::from(low .. high);
    let mut rng = rand::thread_rng();
    let mut arr = Vec::with_capacity(rows * cols);
    
    for _ in 0 .. rows * cols {
        arr.push(interval.sample(&mut rng))
    }

    return arr;
}

