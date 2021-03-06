use gremlin_lib::{matrix_madd_n_sq, matrix_madd_nmp};

fn test_range(begin: usize, end: usize, func: &Fn(usize)) {
    for n in begin ..= end {
        func(n)
    }
}

#[test]
fn matrix_7x7_test() {
    matrix_madd_n_sq(7);
}

#[test]
fn matrix_4x4_test() {
    matrix_madd_n_sq(4);
}

#[test]
fn matrix_5x5_test() {
    matrix_madd_n_sq(5);
}

#[test]
fn matrix_8x8_test() {
    matrix_madd_n_sq(8);
}

#[test]
fn matrix_8x9x8_test() {
    matrix_madd_nmp(8,9,8);
}

#[test]
fn matrix_8x10x8_test() {
    matrix_madd_nmp(8,10,8);
}

#[test]
fn matrix_8x11x8_test() {
    matrix_madd_nmp(8,11,8);
}

#[test]
fn matrix_1xn_test() {
    test_range(1, 512, &|n| { matrix_madd_nmp(1, n, 1) })
}

#[test]
fn matrix_1x20xp_test() {
    test_range(1, 512, &|p| { matrix_madd_nmp(1, 20, p) })
}

#[test]
fn matrix_20xmx1_test() {
    test_range(2, 32, &|m| { matrix_madd_nmp(20, m, 1) });
}

#[test]
fn matrix_40xmx2_test() {
    test_range(2, 64, &|m| { matrix_madd_nmp(40, m, 2) });
}

#[test]
fn matrix_1to20_sq() {
    test_range(1, 20, &|n| {
        println!("{}", n);
        matrix_madd_n_sq(n)
    })
}

#[test]
fn matrix_7x6_6xn() {
    test_range(1, 10, &|n| {
        println!("{}", n);
        matrix_madd_nmp(7,6,n)
    })
}

#[test]
fn matrix_nmp_small() {
    for n in 8 .. 32 {
        for m in 8 .. 32 {
            test_range(8, 32, &|p| {
                if m != n || m != p || n != p {
                    println!("{} {} {}", m, n, p);
                    matrix_madd_nmp(n, m, p)
                }
            })
        }
    }
}

#[test]
fn matrix_28_to_36_sq() {
    test_range(28, 36, &|n| {
        println!("{}", n);
        matrix_madd_n_sq(n)
    });
}

#[test]
fn matrix_60_to_68_sq() {
    test_range(60, 68, &|n| {
        println!("{}", n);
        matrix_madd_n_sq(n);
    });
}

#[test]
fn matrix_124_to_132_sq() {
    test_range(1, 4, &|n| {
        println!("{}", 120 + n * 4);
        matrix_madd_n_sq(120 + n * 4)
    })
}

#[test]
fn matrix_252_to_260_sq_test() {
    test_range(252, 260, &matrix_madd_n_sq)
}

#[test]
fn matrix_480_sq_test() {
    matrix_madd_n_sq(480);
}

#[test]
fn matrix_507_sq_test() {
    matrix_madd_n_sq(507);
}

#[test]
fn matrix_508_sq_test() {
    matrix_madd_n_sq(508);
}

#[test]
fn matrix_512_sq_test() {
    matrix_madd_n_sq(512);
}

#[test]
fn matrix_516_sq_test() {
    matrix_madd_n_sq(516);
}

#[test]
fn matrix_613_sq_test() {
    matrix_madd_n_sq(613);
}

#[test]
fn matrix_768sq_test() {
    matrix_madd_n_sq(768);
}

#[test]
fn test_recurse_big_n() {
    matrix_madd_nmp(959, 240, 240)
}

#[test]
fn test_recurse_very_big_n() {
    matrix_madd_nmp(1921, 240, 240)
}

#[test]
fn test_recurse_big_m() {
    matrix_madd_nmp(240, 959, 240)
}

#[test]
fn test_recurse_very_big_m() {
    matrix_madd_nmp(240, 1921, 240)
}

#[test]
fn test_recurse_big_p() {
    matrix_madd_nmp(240, 240, 959)
}

#[test]
fn test_recurse_very_big_p() {
    matrix_madd_nmp(240, 240, 1921)
}

#[test]
fn test_recurse_big_nm() {
    matrix_madd_nmp(959, 959, 240)
}

#[test]
fn test_recurse_big_np() {
    matrix_madd_nmp(959, 240, 959)
}

#[test]
fn test_recurse_big_mp() {
    matrix_madd_nmp(240, 959, 959)
}

#[test]
fn test_1492_1150_1201() {
    matrix_madd_nmp(1492, 1150, 1201)
}

