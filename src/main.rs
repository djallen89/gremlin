#![feature(align_offset)]

extern crate matrixmultiply;

pub mod lib;

#[cfg(test)]
mod test;

use lib::random_array;
use lib::matrix_madd;
use matrixmultiply::dgemm;

#[cfg(bench)]
mod benchmark;
    
fn main() {
}
