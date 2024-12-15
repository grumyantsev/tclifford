use ndarray::Axis;
use ndarray::{Array3, ArrayView3, ArrayViewMut3};
use num::complex::Complex64;

fn wfft_repr_alpha_inplace(mut a: ArrayViewMut3<Complex64>) {
    for ((idx, i, j), c) in a.indexed_iter_mut() {
        if (idx.count_ones() + (i ^ j).count_ones()) & 1 == 1 {
            *c = -*c;
        }
    }
}

fn wmul_impl(
    a: ArrayViewMut3<Complex64>,
    b: ArrayView3<Complex64>,
    mut dest: ArrayViewMut3<Complex64>,
) {
    let size = a.dim().0;
    if size == 1 {
        let mut dst = dest.index_axis_mut(Axis(0), 0);
        let a0 = a.index_axis(Axis(0), 0);
        let b0 = b.index_axis(Axis(0), 0);

        dst.assign(&a0.dot(&b0));
        return;
    }

    // (a0 + e ^ a1) * (b0 + e ^ b1) =
    // a0 ^ b0 + e * (a1 ^ b0 + ~a0 ^ b1)
    let (mut a0, a1) = a.split_at(Axis(0), size / 2);
    let (b0, b1) = b.split_at(Axis(0), size / 2);
    let (mut bottom, mut top) = dest.split_at(Axis(0), size / 2);

    // Use bottom as the temporary storage to avoid any new allocations
    let mut tmp = bottom.view_mut();
    wmul_impl(a1, b0, top.view_mut());
    wfft_repr_alpha_inplace(a0.view_mut());
    wmul_impl(a0.view_mut(), b1, tmp.view_mut());
    // fill the top
    for (index, c) in top.indexed_iter_mut() {
        *c = *c + tmp[index];
    }
    // return a0 back to it's original state
    wfft_repr_alpha_inplace(a0.view_mut());
    // fill the bottom
    wmul_impl(a0, b0, bottom);
}

// The shapes of arrays are unchecked here. The caller MUST ensure their validity.
pub(super) fn wmul(a: ArrayView3<Complex64>, b: ArrayView3<Complex64>) -> Array3<Complex64> {
    if a.dim().0 == 1 {
        // Optimization for non-degen signatures that avoids extra memory allocations
        return a
            .index_axis(Axis(0), 0)
            .dot(&b.index_axis(Axis(0), 0))
            .into_shape_clone(a.dim())
            .unwrap();
    }
    let mut ret = Array3::zeros(a.dim());
    let mut ca = a.into_owned();
    wmul_impl(ca.view_mut(), b, ret.view_mut());
    ret
}
