#![feature(align_offset)]

pub mod lib;
use lib::{minimatrix_fmadd64, floateq};

fn main() {
    let a_arr = vec!(1.0, 2.0, 3.0, 4.0,
                     7.0, 6.0, 5.0, 4.0,
                     0.5, 1.0, 2.0, 4.0,
                     8.0, 2.0, 0.5, 0.125);

    let _align1 = vec!(0.0, 0.0, 0.0, 0.0);

    let b_arr = vec!(12.0,  13.0, 15.0, 17.0,
                     01.0,   2.0,  3.0,  4.0,
                     42.0, 404.0, 13.0,  9.0,
                     01.0,   2.0,  3.0,  4.0);

    let _align2 = vec!(0.0, 0.0, 0.0, 0.0);

    let mut c_arr = vec!(10.0, 20.0,  30.0,  40.0,
                     02.0,  6.0,  24.0, 120.0,
                     01.0,  1.0,   2.0,   3.0,
                     05.0, 25.0, 125.0, 625.0);

    let res_arr = vec!(154.000, 1257.00, 102.000, 108.0,
                       306.000, 2137.00, 224.000, 324.0,
                       096.000,  825.50,  50.500,  49.5,
                       124.125,  335.25, 257.875, 447.0);

    /*
    let bptr_ofst = (&b_arr[0] as *const f64).align_offset(32);
    let cptr_ofst = (&c_arr[0] as *const f64).align_offset(32);
    println!("b offset = {}, c offset = {}", bptr_ofst, cptr_ofst);
     */
    
    minimatrix_fmadd64(&a_arr, &b_arr, &mut c_arr);

    for row in 0 .. 4 {
        let ridx = row * 4;
        for col in 0 .. 3 {
            //assert!(res_arr[ridx + col] == c_arr[ridx + col]);
            assert!(floateq(res_arr[ridx + col], c_arr[ridx + col]));
        }
    }
}
