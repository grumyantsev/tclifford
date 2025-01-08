// Using ndarray_linalg just to do some matrix inverse is annoying.
// It brings the heavy openblas dependency with it, increasing the compile time, and it is very poorly maintained.
//
// Let's just make a simple Gauss inverse.

use crate::types::DivRing;
use crate::Norm;

use ndarray::{s, Array2, ArrayBase};
use num::Zero;

//
// FIXME: This breaks on things like complex quaternions,
// because some of them are not invertible, despite implementing the DivRing trait.
//
// Since this is only intended for usage in the FFTRepr inverse operation, it can really only operate on complex numbers.
// But when the FFTRepr is made more generic, this is going to become an issue.
//
pub(crate) fn matrix_inv<T: DivRing + Clone + Norm>(
    mat: &ArrayBase<impl ndarray::Data<Elem = T>, ndarray::Ix2>,
) -> Option<Array2<T>> {
    let shape = mat.dim();
    if shape.0 != shape.1 {
        return None;
    }
    let n = shape.0;

    let mut w = Array2::zeros((n, 2 * n));
    w.slice_mut(s![0..n, 0..n]).assign(mat);
    for i in 0..n {
        w[(i, n + i)] = T::one();
    }

    // To upper triangle
    for i in 0..n {
        // Find the row with the biggest first element, and swap it with the current one.
        // It helps to avoid division by zero or near-zero (latter greatly reducing precision).
        let mut max_k = w[(i, i)].norm();
        let mut max_idx = i;
        for j in i + 1..n {
            let kn = w[(j, i)].norm();
            if kn > max_k {
                max_k = kn;
                max_idx = j;
            }
        }
        if max_k.is_zero() {
            return None;
        }
        for m in 0..2 * n {
            w.swap((i, m), (max_idx, m));
        }
        let k = w[(i, i)].clone();

        // normalize the i-th row
        for m in i..2 * n {
            // Accommodate for non-commutative T, such as quaternions.
            w[(i, m)] = T::one().ref_div(&k).ref_mul(&w[(i, m)]);
        }
        for j in i + 1..n {
            let coeff = w[(j, i)].clone();
            for m in i..2 * n {
                w[(j, m)] = w[(j, m)].ref_sub(&coeff.ref_mul(&w[(i, m)]));
                if w[(j, m)] != w[(j, m)] {
                    // NaN check
                    return None;
                }
            }
            if w[(j, j)].is_zero() {
                return None;
            }
        }
    }
    // To diagonal
    for i in (0..n).rev() {
        for j in (0..i).rev() {
            // Accommodate for non-commutative T, such as quaternions.
            let coeff = T::one().ref_div(&w[(i, i)]).ref_mul(&w[(j, i)]);
            for m in j..2 * n {
                w[(j, m)] = w[(j, m)].ref_sub(&coeff.ref_mul(&w[(i, m)]));
            }
        }
    }

    Some(w.slice(s![0..n, n..2 * n]).to_owned())
}

#[cfg(test)]
mod test {
    use super::matrix_inv;
    use crate::{quaternion::Quaternion, Norm};
    use num::{One, Zero};

    use ndarray::{arr2, Array2};
    //use num::complex::Complex64;

    #[test]
    fn minv_test() {
        let cases = [
            arr2(&[
                [0.00000000000001, 2., 3.], //
                [3., 0., 5.],               //
                [5., 7., 0.],               //
            ]),
            arr2(&[
                [0.00000000000001, 2., 3.],  //
                [0.000000000000001, 1., 5.], //
                [5., 7., 0.],                //
            ]),
            arr2(&[
                [1., 0., 1000.],                    //
                [0., 1., 0.],                       //
                [1000., 0., 0.0000000000000000001], //
            ]),
            arr2(&[
                [1., 0., 0.],                    //
                [0., 1., 0.],                    //
                [0., 0., 0.0000000000000000001], //
            ]),
            arr2(&[
                [1e-16, 0., 0.], //
                [0., 1e-16, 0.], //
                [0., 0., 1e-16], //
            ]),
            arr2(&[
                [1e-16, 1e-16, 1e-16], //
                [1e-15, 1e-13, 1e-16], //
                [1e-14, 1e-16, 1e-16], //
            ]),
        ];

        for m in cases {
            if let Some(minv) = matrix_inv(&m) {
                println!("{:4.3}", minv.dot(&m));
                assert!((minv.dot(&m) - Array2::<f64>::eye(m.dim().0))
                    .iter()
                    .all(|c| c.abs() < 1e-12))
            } else {
                assert!(false, "Inverse not found for an invertible matrix");
            }
        }

        // Check non-commutative case
        let qcases = [
            //
            arr2::<Quaternion<f64>, 3>(&[
                [Quaternion::i(), Quaternion::j(), Quaternion::j()], //
                [Quaternion::k(), Quaternion::k(), Quaternion::j()], //
                [Quaternion::one(), Quaternion::i(), Quaternion::j()], //
            ]),
            arr2::<Quaternion<f64>, 3>(&[
                [Quaternion::i(), Quaternion::j(), Quaternion::j()], //
                [
                    Quaternion::k(),
                    Quaternion::from_scalar(1e-16),
                    Quaternion::j(),
                ], //
                [Quaternion::one(), Quaternion::zero(), Quaternion::zero()], //
            ]),
        ];

        for m in qcases {
            if let Some(minv) = matrix_inv(&m) {
                println!("{:4.3}", minv.dot(&m));
                assert!((&minv.dot(&m) - &Array2::<Quaternion<f64>>::eye(m.dim().0))
                    .iter()
                    .all(|c| c.norm() < 1e-12))
            } else {
                assert!(false, "Inverse not found for an invertible matrix");
            }
        }
    }
}
