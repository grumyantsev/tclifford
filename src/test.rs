#[cfg(test)]
mod test {
    use crate::algebra::{ClBasis, NonDegenerate};
    use crate::types::{DivRing, FromComplex, IntoComplex64};
    use core::f64;
    use std::fmt::Debug;
    use std::hint::black_box;
    use std::time;

    use crate::algebra::ClAlgebra;
    use crate::declare_algebra;
    use crate::{Multivector, SparseMultivector};
    use ndarray::Array2;
    use num::complex::{Complex32, Complex64};
    use num::One;
    use num::Zero;

    #[test]
    fn basis_test() {
        declare_algebra!(Cl44, [+,+,+,+,-,-,-,-], ["e1", "e2", "e3", "e4", "g1", "g2", "g3", "g4"]);
        // Check unpacking
        let [e1, e2, e3, e4, g1, g2, g3, g4] = Cl44::basis::<f64>();
        println!("{}", e1 + e2 + e3 + e4 + g1 + g2 + g3 + g4);

        // Check basis multiplication
        let sig = Cl44::signaturef();
        let b = Cl44::basis_sparse::<Complex64>();
        for i in 0..b.len() {
            assert_eq!(
                &b[i] * &b[i],
                SparseMultivector::<Complex64, Cl44>::from_scalar(sig[i].into())
            );
            for j in 0..i {
                assert_eq!(&b[i] * &b[j], -(&b[j] * &b[i]))
            }
        }
    }

    #[test]
    fn fft_test() {
        fn fft_test_case<
            const DIM: usize,
            const REPR_DIM: usize,
            T: DivRing + Clone + IntoComplex64 + FromComplex + Debug,
            A: ClAlgebra + ClBasis<DIM> + NonDegenerate + Debug,
        >() {
            let e = A::basis::<T>();
            for i in 0..e.len() {
                let fei = e[i].fft();
                // Check the the square of fft square is negative identity
                assert_eq!(
                    (&fei * &fei).into_array2(),
                    Array2::from_diag_elem(REPR_DIM, Complex64::one() * A::signaturef()[i])
                );
                for j in 0..i {
                    let eij = &fei * &e[j].fft();
                    let eji = &e[j].fft() * &fei;
                    // Check anticommutativity
                    assert_eq!(eij, -&eji);

                    let prod = eij.ifft();
                    // Check that naive and fft products agree
                    assert_eq!(prod, e[i].naive_wedge_impl(&e[j]));
                    // And that the fft product is correct at all
                    assert!(prod.get_by_idx((1 << i) | (1 << j)).is_one());
                    assert!(prod.set_by_idx((1 << i) | (1 << j), T::zero()).is_zero());
                }
            }
        }

        declare_algebra!(Oct, [-,-,-,-,-,-]);
        fft_test_case::<6, 8, f32, Oct>();
        fft_test_case::<6, 8, f64, Oct>();
        fft_test_case::<6, 8, Complex32, Oct>();
        fft_test_case::<6, 8, Complex64, Oct>();
        declare_algebra!(ClOdd, [-,-,-,+,-,+,-]);
        fft_test_case::<7, 16, f32, ClOdd>();
        fft_test_case::<7, 16, f64, ClOdd>();
        fft_test_case::<7, 16, Complex32, ClOdd>();
        fft_test_case::<7, 16, Complex64, ClOdd>();
    }

    #[test]
    fn ops_test() {
        declare_algebra!(Oct, [-,-,-,-,-,-]);
        type MV = SparseMultivector<f64, Oct>;

        // Check values of blade exponents
        let mut theta = 0.0;
        while theta < f64::consts::TAU {
            let b = MV::zero().set_by_idx(0b11, theta);

            let r = b.exp();
            let expected_r = MV::zero()
                .set_by_idx(0, theta.cos())
                .set_by_idx(0b11, theta.sin());

            assert!(r.approx_eq(&expected_r, 1e-12));
            theta += 0.1;
        }

        let mut theta = 0.0;
        while theta < 100. {
            let b = MV::zero().set_by_idx(0b101010, theta);

            let r = b.exp();
            let expected_r = MV::zero()
                .set_by_idx(0, theta.cosh())
                .set_by_idx(0b101010, theta.sinh());

            assert!(r.approx_eq(&expected_r, theta.exp() / 1e12));
            theta += 0.3;
        }
    }

    #[test]
    fn nested_test() {
        declare_algebra!(Cl6, [+,+,+,+,+,+], ["e1", "e2", "e3", "e4", "e5", "e6"]);
        declare_algebra!(Quat, [-,-], ["i", "j"]);
        type MVQ = Multivector<f64, Quat>;
        type MV = Multivector<MVQ, Cl6>;
        let a = MV::zero()
            .set_by_idx(0b111000, MVQ::from_scalar(1.).set_by_idx(0b11, 1.))
            .set_by_idx(
                0b110001,
                MVQ::from_scalar(1.)
                    .set_by_idx(0b01, 1.)
                    .set_by_idx(0b10, 1.),
            );
        let b = MV::zero().set_by_idx(0b000001, MVQ::from_scalar(1.).set_by_idx(0b01, 1.));
        println!("({a})\n*\n({b})\n=\n{}", &a * &b);
    }

    #[test]
    fn bench_rev() {
        declare_algebra!(A, [+,+,+,+,0,0,0], ["w", "x", "y", "z", "e0", "e1", "e3"]);
        type MV = Multivector<Complex64, A>;
        let a = MV::from_indexed_iter(A::index_iter().map(|idx| {
            (
                idx,
                Complex64 {
                    re: rand::random::<f64>(),
                    im: rand::random::<f64>(),
                },
            )
        }))
        .unwrap();
        let st = time::Instant::now();
        for _ in 0..10000 {
            let _ = black_box(a.rev().fft());
        }
        println!("r f {:?}", st.elapsed());

        let st = time::Instant::now();
        for _ in 0..10000 {
            let _ = black_box(a.fft().rev());
        }
        println!("f r {:?}", st.elapsed());
    }
}
