use crate::{
    types::{DivRing, IntoComplex64},
    ClError,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis};
use num::complex::Complex64;

mod tests;

/**
 * Main automorphism of multivector representation.
 * Element `c == M[i,j]` of matrix `M` is converted as follows:
 * `M[i,j] -> alpha(M)[i,j]`
 */
#[inline(always)]
pub(crate) fn alpha<T>(c: &T, i: usize, j: usize) -> T
where
    T: DivRing + Clone,
{
    if ((i ^ j).count_ones() & 1) == 1 {
        c.ref_neg()
    } else {
        c.clone()
    }
}

/*
 * log2(size) if size is a power of 2, Err(ClError::InvalidShape) otherwise
 */
fn cl_dim(size: usize) -> Result<usize, ClError> {
    if size == 0 {
        return Err(ClError::InvalidShape);
    }

    let dim = size.ilog2() as usize;
    if (1 << dim) != size {
        return Err(ClError::InvalidShape);
    }

    Ok(dim)
}

/**
 * Fast Clifford-Fourier transform in Cl(n, n)
 */
pub(crate) fn clifft_nn<T>(coeffs: ArrayView1<T>) -> Result<Array2<T>, ClError>
where
    T: DivRing + Clone,
{
    let size = coeffs.len();
    let dim = cl_dim(size)?;

    // square root of size - side length of resulting matrix
    let mside = 1 << (dim / 2);

    let mut res = Array2::<T>::zeros((mside, mside));

    clifft_nn_into(coeffs, res.view_mut())?;

    Ok(res)
}

fn clifft_nn_into<T>(coeffs: ArrayView1<T>, mut dest: ArrayViewMut2<T>) -> Result<(), ClError>
where
    T: DivRing + Clone,
{
    let size = coeffs.len();
    if size == 1 {
        dest[(0, 0)] = coeffs[0].clone();
        return Ok(());
    }

    let dim = cl_dim(size)?;

    let quarter_size = size / 4;
    // square root of size - side length of resulting matrix
    let mside = 1 << (dim / 2);
    // after each quarter is processed it becomes a matrix of quarter_mside*quarter_mside
    let quarter_mside = mside / 2;

    debug_assert!(mside * mside == size);
    debug_assert!(quarter_mside * quarter_mside == quarter_size);

    // split mv into quarters
    let (mv_0, mv_1) = coeffs.split_at(Axis(0), size / 2);
    let (mv_00, mv_01) = mv_0.split_at(Axis(0), quarter_size);
    let (mv_10, mv_11) = mv_1.split_at(Axis(0), quarter_size);

    // (a b)
    // (c d)
    let (ab, cd) = dest.split_at(Axis(0), quarter_mside);
    let (mut a, mut b) = ab.split_at(Axis(1), quarter_mside);
    let (mut c, mut d) = cd.split_at(Axis(1), quarter_mside);

    // Process each part
    let mx_00 = clifft_nn(mv_00)?;
    let mx_01 = clifft_nn(mv_01)?;
    let mx_10 = clifft_nn(mv_10)?;
    let mx_11 = clifft_nn(mv_11)?;

    // Put sums/diffs into the resulting ndarray.
    // Avoiding extra data copying that `concatenate!` would do.
    for i in 0..quarter_mside {
        for j in 0..quarter_mside {
            // left half
            a[(i, j)] = mx_00[(i, j)].ref_add(&mx_11[(i, j)]);
            c[(i, j)] = mx_01[(i, j)].ref_sub(&mx_10[(i, j)]);
            // right half
            b[(i, j)] = alpha(&mx_01[(i, j)].ref_add(&mx_10[(i, j)]), i, j);
            d[(i, j)] = alpha(&mx_00[(i, j)].ref_sub(&mx_11[(i, j)]), i, j);
        }
    }

    Ok(())
}

/**
 * Fast inverse Clifford-Fourier transform in Cl(n, n)
 */
fn iclifft_nn<T>(matrix: ArrayView2<T>) -> Result<Array1<T>, ClError>
where
    T: DivRing + Clone,
{
    let mut res = Array1::zeros(matrix.len());

    iclifft_nn_into(matrix, res.view_mut())?;

    Ok(res)
}

fn iclifft_nn_into<T>(matrix: ArrayView2<T>, mut dest: ArrayViewMut1<T>) -> Result<(), ClError>
where
    T: DivRing + Clone,
{
    let mside = matrix.nrows();
    if mside != matrix.ncols() || mside.count_ones() != 1 {
        return Err(ClError::InvalidShape);
    }
    if mside == 1 {
        dest[0] = matrix[(0, 0)].clone();
        return Ok(());
    }
    let quarter_mside = mside / 2;

    let size = mside * mside;
    let quarter_size = quarter_mside * quarter_mside;

    debug_assert!(quarter_size == size / 4);

    // split matrix into quarters
    let (ab, cd) = matrix.split_at(Axis(0), quarter_mside);
    let (a, b) = ab.split_at(Axis(1), quarter_mside);
    let (c, d) = cd.split_at(Axis(1), quarter_mside);

    let mv_a = iclifft_nn(a)?;
    let mv_b = iclifft_nn(b)?;
    let mv_c = iclifft_nn(c)?;
    let mv_d = iclifft_nn(d)?;

    // Fill the quarters of the dest multivector
    let (mv_0, mv_1) = dest.split_at(Axis(0), size / 2);
    let (mut mv_00, mut mv_01) = mv_0.split_at(Axis(0), quarter_size);
    let (mut mv_10, mut mv_11) = mv_1.split_at(Axis(0), quarter_size);

    let two = T::one() + T::one();
    for i in 0..quarter_size {
        mv_00[i] = mv_a[i].ref_add(&alpha(&mv_d[i], i, 0)).ref_div(&two);
        mv_01[i] = alpha(&mv_b[i], i, 0).ref_add(&mv_c[i]).ref_div(&two);
        mv_10[i] = alpha(&mv_b[i], i, 0).ref_sub(&mv_c[i]).ref_div(&two);
        mv_11[i] = mv_a[i].ref_sub(&alpha(&mv_d[i], i, 0)).ref_div(&two);
    }

    Ok(())
}

/**
 * Fast Clifford-Fourier transform of Cl(n) multivector coeff array.
 *
 * Index in the array is the blade index, i.e. `coeffs[0b1101]` is a coefficient in front of (e3^e2^e0).
 *
 * Example: Pauli matrices
 * ```
 * use tclifford::clifft::{clifft, iclifft};
 * use ndarray::{array, Array1, Array2};
 * use num::{One, Zero};
 * use num::complex::Complex64;
 *
 * let c0 = Complex64::zero();
 * let c1 = Complex64::one();
 * let ci = Complex64::i();
 *
 * let x = array![0., 1., 0., 0.];
 * let y = array![0., 0., 1., 0.];
 *
 * let sigma_x = clifft(x.view()).unwrap();
 * let sigma_y = clifft(y.view()).unwrap();
 * let sigma_z = (&sigma_x).dot(&sigma_y) * ci;
 *
 * assert!(
 *     sigma_x ==
 *     array![[c0,c1],
 *            [c1,c0]]
 * );
 * assert!(
 *     sigma_y ==
 *     array![[c0,ci],
 *            [-ci,c0]]
 * );
 * assert!(
 *     sigma_z ==
 *     array![[c1,c0],
 *            [c0,-c1]]
 * );
 *
 * assert!(
 *     iclifft(sigma_z.view()).unwrap() == array![c0, c0, c0, -ci]
 * );
 *
 * ```
 */
pub fn clifft<T>(coeffs: ArrayView1<T>) -> Result<Array2<Complex64>, ClError>
where
    T: IntoComplex64 + Clone,
{
    let mut size = coeffs.len();
    let mut dim = cl_dim(size)?;
    // round up n dimensions to 2m
    if dim & 1 == 1 {
        dim += 1;
    }
    size = 1 << dim;

    let mut arr = Array1::<Complex64>::zeros(size);

    // Multiply half of basis vectors by i, making Cl(m, m) algebra from Cl(2m)
    // ei = i*Ei
    // where ei*ei = 1; Ei*Ei = -1
    // We choose odd basis vectors to become negative
    let idx_mask = 0xAAAAAAAAAAAAAAAAusize;
    const I_POWERS: [Complex64; 4] = [
        Complex64 { re: 1., im: 0. },
        Complex64 { re: 0., im: 1. },
        Complex64 { re: -1., im: 0. },
        Complex64 { re: 0., im: -1. },
    ];
    for (idx, c) in coeffs.iter().enumerate() {
        arr[idx] = c.into_complex() * I_POWERS[((idx & idx_mask).count_ones() % 4) as usize];
    }

    clifft_nn(arr.view())
}

pub(crate) fn clifft_into<T>(
    coeffs: ArrayView1<T>,
    dest: ArrayViewMut2<Complex64>,
) -> Result<(), ClError>
where
    T: IntoComplex64 + Clone,
{
    let mut size = coeffs.len();
    let mut dim = cl_dim(size)?;
    // round up n dimensions to 2m
    if dim & 1 == 1 {
        dim += 1;
    }
    size = 1 << dim;

    let mut arr = Array1::<Complex64>::zeros(size);

    // Multiply half of basis vectors by i, making Cl(m, m) algebra from Cl(2m)
    // ei = i*Ei
    // where ei*ei = 1; Ei*Ei = -1
    // We choose odd basis vectors to become negative
    let idx_mask = 0xAAAAAAAAAAAAAAAAusize;
    const I_POWERS: [Complex64; 4] = [
        Complex64 { re: 1., im: 0. },
        Complex64 { re: 0., im: 1. },
        Complex64 { re: -1., im: 0. },
        Complex64 { re: 0., im: -1. },
    ];
    for (idx, c) in coeffs.iter().enumerate() {
        arr[idx] = c.into_complex() * I_POWERS[((idx & idx_mask).count_ones() % 4) as usize];
    }

    clifft_nn_into(arr.view(), dest)
}

/**
 * Fast inverse Clifford-Fourier transform back into Cl(n) multivector coefficient array.
 */
pub fn iclifft(matrix: ArrayView2<Complex64>) -> Result<Array1<Complex64>, ClError> {
    let mut arr = Array1::zeros(matrix.len());
    iclifft_into(matrix, arr.view_mut())?;
    Ok(arr)
}

pub fn iclifft_into(
    matrix: ArrayView2<Complex64>,
    mut dest: ArrayViewMut1<Complex64>,
) -> Result<(), ClError> {
    iclifft_nn_into(matrix, dest.view_mut())?;

    // Multiply imaginary basis vectors of Cl(m, m) by -i to produce Cl(2m) multivector
    let idx_mask = 0xAAAAAAAAAAAAAAAAusize;
    const I_POWERS: [Complex64; 4] = [
        Complex64 { re: 1., im: 0. },
        Complex64 { re: 0., im: -1. },
        Complex64 { re: -1., im: 0. },
        Complex64 { re: 0., im: 1. },
    ];
    for (idx, c) in dest.indexed_iter_mut() {
        *c = *c * I_POWERS[((idx & idx_mask).count_ones() % 4) as usize];
    }

    Ok(())
}

fn imaginary_flip_inplace<T>(m: ArrayViewMut2<T>)
where
    T: DivRing + Clone,
{
    if (&m).len() == 1 {
        return;
    }

    let w = (&m).ncols();
    let h = (&m).nrows();

    let (ab, cd) = m.split_at(Axis(0), h / 2);
    let (mut a, mut b) = ab.split_at(Axis(1), w / 2);
    let (mut c, mut d) = cd.split_at(Axis(1), w / 2);

    for i in 0..(h / 2) {
        for j in 0..(w / 2) {
            let tmp = a[(i, j)].clone();
            a[(i, j)] = alpha(&d[(i, j)], i, j);
            d[(i, j)] = alpha(&tmp, i, j);

            let tmp = b[(i, j)].clone();
            b[(i, j)] = alpha(&c[(i, j)], i, j);
            c[(i, j)] = alpha(&tmp, i, j);
        }
    }
    imaginary_flip_inplace(a);
    imaginary_flip_inplace(b);
    imaginary_flip_inplace(c);
    imaginary_flip_inplace(d);
}

/// Flip every axis: e -> -e
pub fn parity_flip<T>(mv: ArrayView2<T>) -> Array2<T>
where
    T: DivRing + Clone,
{
    let mut ret = Array2::zeros((mv.nrows(), mv.ncols()));
    for ((i, j), c) in mv.indexed_iter() {
        ret[(i, j)] = alpha(c, i, j)
    }
    ret
}

/// The Beta automorphism.
/// Flips the imaginary axes.
pub fn imaginary_flip<T>(m: ArrayView2<T>) -> Result<Array2<T>, ClError>
where
    T: DivRing + Clone,
{
    if m.nrows() != m.ncols() || m.nrows().count_ones() != 1 {
        return Err(ClError::InvalidShape);
    }
    let mut ret = m.clone().to_owned();
    imaginary_flip_inplace(ret.view_mut());
    Ok(ret)
}

/// Representation of the reversal of the multivector.
pub fn reversal<T>(m: ArrayView2<T>) -> Result<Array2<T>, ClError>
where
    T: DivRing + Clone,
{
    imaginary_flip(m.t())
}
