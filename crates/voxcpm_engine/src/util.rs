pub(crate) fn prod_usize(xs: &[usize]) -> usize {
    xs.iter().copied().product()
}
