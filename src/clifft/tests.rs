#[cfg(test)]
mod tests {
    use crate::clifft::{clifft, iclifft};
    use crate::clifft::{imaginary_flip, parity_flip, reversal};
    use crate::index_utils::grade_iter;
    use crate::ClError;
    use ndarray::{array, Array1, Array2, ArrayView1};
    use num::complex::Complex64;
    use num::{Integer, One, Zero};

    fn alpha_1d(mv: ArrayView1<Complex64>) -> Array1<Complex64> {
        let mut ret = Array1::zeros(mv.len());
        for idx in 0..mv.len() {
            ret[idx] = if (idx.count_ones() & 1) == 1 {
                -mv[idx]
            } else {
                mv[idx]
            }
        }
        ret
    }

    #[test]
    fn expected_values_test() {
        let n = 16;

        let ci = Complex64::i();
        let c1 = Complex64::one();
        let cz = Complex64::zero();

        let mut arr = Array1::zeros(n);
        arr[8] = Complex64::one();
        let arr_cft = clifft(arr.view()).unwrap();
        assert!(
            arr_cft
                == array!(
                    [cz, cz, ci, cz,],
                    [cz, cz, cz, ci,],
                    [-ci, cz, cz, cz,],
                    [cz, -ci, cz, cz,],
                )
        );
        println!("{}", iclifft(arr_cft.view()).unwrap());
        assert!(iclifft(arr_cft.view()).unwrap() == arr);

        let mut arr = Array1::zeros(n);
        arr[4] = Complex64::one();
        let arr_cft = clifft(arr.view()).unwrap();
        assert!(
            arr_cft
                == array!(
                    [cz, cz, c1, cz,],
                    [cz, cz, cz, c1,],
                    [c1, cz, cz, cz,],
                    [cz, c1, cz, cz,],
                )
        );
        assert!(iclifft(arr_cft.view()).unwrap() == arr);

        let mut arr = Array1::zeros(1 << 6);
        arr[2] = Complex64::one();
        let arr_cft = clifft(arr.view()).unwrap();
        assert!(
            arr_cft
                == array!(
                    [cz, ci, cz, cz, cz, cz, cz, cz,],
                    [-ci, cz, cz, cz, cz, cz, cz, cz,],
                    [cz, cz, cz, -ci, cz, cz, cz, cz,],
                    [cz, cz, ci, cz, cz, cz, cz, cz,],
                    [cz, cz, cz, cz, cz, -ci, cz, cz,],
                    [cz, cz, cz, cz, ci, cz, cz, cz,],
                    [cz, cz, cz, cz, cz, cz, cz, ci,],
                    [cz, cz, cz, cz, cz, cz, -ci, cz,],
                )
        );
        assert!(iclifft(arr_cft.view()).unwrap() == arr);
    }

    #[test]
    fn clifft_ndarray_test() {
        let n = 1 << 20;
        let mut c = Vec::<f64>::new();
        c.reserve(n);
        for i in 1..(n + 1) {
            c.push(i as f64)
        }
        let matrix = clifft(Array1::from_vec(c).view()).unwrap();

        let res = iclifft(matrix.view()).unwrap();
        for i in 0..n {
            assert!(res[i] == ((i + 1) as f64).into());
        }
    }

    #[test]
    fn clifft_mul_test() {
        // check that all the basis vectors are multiplied into basis bivectors
        let dim = 12;
        let n = 1 << dim;

        let mut x = Array1::zeros(n);
        let mut y = Array1::zeros(n);

        for i in 0..dim {
            x[1 << i] = Complex64::one();
            let cx = clifft(x.view()).unwrap();

            // check that cx squares to 1
            let expected_unit = (&cx).dot(&cx);
            assert!(expected_unit == Array2::from_diag(&Array1::<Complex64>::ones(1 << (dim / 2))));
            let mut expected_one = iclifft(expected_unit.view()).unwrap();
            assert!(expected_one[0] == Complex64::one());
            expected_one[0] = Complex64::zero();
            assert!(expected_one == Array1::<Complex64>::zeros(1 << dim));

            for j in (i + 1)..dim {
                y[1 << j] = Complex64::one();

                let cy = clifft(y.view()).unwrap();

                let xy_image = cx.dot(&cy);
                let mut xy = iclifft(xy_image.view()).unwrap();

                let yx_image = cy.dot(&cx);
                let yx = iclifft(yx_image.view()).unwrap();

                println!(
                    "e{} * e{} = ({}) * e{}^e{}",
                    i,
                    j,
                    xy[(1 << i) | (1 << j)],
                    j,
                    i,
                );

                // check anticommutativity of base vectors
                assert_eq!(xy[(1 << i) | (1 << j)], -yx[(1 << i) | (1 << j)]);
                // check that we get the expected bivector (coeff should be -1 due to multiplication order)
                assert_eq!(xy[(1 << i) | (1 << j)], -Complex64::one());
                // and the rest of coefficients is zero
                xy[(1 << i) | (1 << j)] = Complex64::zero();
                assert!(xy.iter().all(Complex64::is_zero));

                y[1 << j] = Complex64::zero();
            }
            x[1 << i] = Complex64::zero();
        }
    }

    #[test]
    fn alpha_test() {
        let dim = 12;
        let mut x = Array1::zeros(1 << dim);
        for i in 0..(1 << dim) {
            x[i] = Complex64::from((i + 1) as f64);
        }
        assert!(
            // Maybe using view was not a good idea after all...
            iclifft(parity_flip(clifft(x.view()).unwrap().view()).view()).unwrap()
                == alpha_1d(x.view())
        );
    }

    #[test]
    fn corner_cases_test() {
        assert_eq!(clifft::<f64>(array![].view()), Err(ClError::InvalidShape));
        assert_eq!(iclifft(array![[]].view()), Err(ClError::InvalidShape));
        assert_eq!(
            clifft(
                array![
                    Complex64::from(1.),
                    Complex64::from(2.),
                    Complex64::from(3.)
                ]
                .view()
            ),
            Err(ClError::InvalidShape)
        );
        assert_eq!(
            imaginary_flip(
                array![[1., 2., 3.], [4., 5., 6.]]
                    .mapv(Complex64::from)
                    .view(),
            ),
            Err(ClError::InvalidShape)
        );
    }

    #[test]
    fn imaginary_flip_test() {
        let dim = 10;
        let mut coeffs = Array1::<Complex64>::zeros(1 << dim);
        for idx in 0..(1 << dim) {
            coeffs[idx] = Complex64::from((idx + 1) as f64);
        }
        let m = clifft(coeffs.view()).unwrap();
        let bm = imaginary_flip(m.view()).unwrap();
        let flipped = iclifft(bm.view()).unwrap();

        for idx in 0..(1 << dim) {
            if (idx & 0xAAAAAAAAAAAAAAAAusize).count_ones().is_even() {
                assert_eq!(flipped[idx], coeffs[idx])
            } else {
                assert_eq!(flipped[idx], -coeffs[idx])
            }
        }
    }

    #[test]
    fn transposition_test() {
        // Only for REAL Cl(2n), transposition of its complex representation conjugate is reversal.
        // For complex Cl(2n), complex conjugate does NOT correspond to reflection of imaginary axes.
        let dim = 8;
        // check 2-vectors
        for idx in grade_iter(dim, 2) {
            let mut coeffs = Array1::<Complex64>::zeros(1 << dim);
            coeffs[idx] = (1.).into();
            let m = clifft(coeffs.view()).unwrap();
            let reversed = iclifft(m.map(|c| c.conj()).t()).unwrap();
            assert_eq!(reversed, -coeffs);
        }
        // check 3-vectors
        for idx in grade_iter(dim, 3) {
            let mut coeffs = Array1::<Complex64>::zeros(1 << dim);
            coeffs[idx] = (1.).into();
            let m = clifft(coeffs.view()).unwrap();
            let reversed = iclifft(m.map(|c| c.conj()).t()).unwrap();
            assert_eq!(reversed, -coeffs);
        }
        // 4-vectors are invariant under reversal
        for idx in grade_iter(dim, 4) {
            let mut coeffs = Array1::<Complex64>::zeros(1 << dim);
            coeffs[idx] = (1.).into();
            let m = clifft(coeffs.view()).unwrap();
            let reversed = iclifft(m.map(|c| c.conj()).t()).unwrap();
            assert_eq!(reversed, coeffs);
        }

        // Universal reversal for any Cl(2n)
        let dim = 12;
        let coeffs = Array1::from_iter(
            (0..(1 << dim)).map(|x| Complex64::new((x + 1) as f64, (x - 1) as f64)),
        );
        let a = clifft(coeffs.view()).unwrap();
        let rev = reversal(a.view()).unwrap();
        for (idx, &c) in iclifft(rev.view()).unwrap().indexed_iter() {
            match idx.count_ones() % 4 {
                0 | 1 => assert_eq!(c, coeffs[idx]),
                2 | 3 => assert_eq!(c, -coeffs[idx]),
                _ => unreachable!(),
            }
        }
    }
}
