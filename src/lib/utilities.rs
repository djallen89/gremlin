use std::ops::Range;

#[inline(always)]
pub fn get_row(row: usize, len: usize, n_cols: usize) -> Range<usize> {
    let begin = row * n_cols;
    begin .. begin + len - 1
}

#[inline(always)]
pub fn get_elt(row: usize, col: usize, n_cols: usize) -> usize {
    row * n_cols + col
}

#[inline(always)]
pub fn get_chunk(row: usize, col: usize, rows: usize, cols: usize, n_cols: usize) -> Range<usize> {
    let begin = row * n_cols + col;
    let end = begin + (rows - 1) * n_cols + cols;
    begin .. end
}
