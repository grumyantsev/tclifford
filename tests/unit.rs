use num::{One, Zero};
use std::hint::black_box;
use std::time;

use tclifford::declare_algebra;
use tclifford::FFTRepr;
use tclifford::Multivector;
use tclifford::SparseMultivector;
use tclifford::TAlgebra;

use num::complex::Complex64;

fn random_mv_real<A: TAlgebra>() -> Multivector<f64, A> {
    Multivector::<f64, A>::from_indexed_iter(A::index_iter().map(|idx| (idx, rand::random())))
        .unwrap()
}

fn random_mv_complex<A: TAlgebra>() -> Multivector<Complex64, A> {
    Multivector::<Complex64, A>::from_indexed_iter(A::index_iter().map(|idx| {
        (
            idx,
            Complex64 {
                re: rand::random::<f64>(),
                im: rand::random::<f64>(),
            },
        )
    }))
    .unwrap()
}

#[test]
fn fft_repr_test() {
    declare_algebra!(A, [+,+,+,+,-,0,0,0], ["w", "x", "y", "z", "t", "e0", "e1", "e3"]);
    type MV = Multivector<Complex64, A>;

    // rev and flip test
    for _ in 0..10 {
        let a: MV = random_mv_complex();
        let b: MV = random_mv_complex();

        let a_repr = a.gfft();
        let b_repr = b.gfft();

        assert!(a_repr.rev().igfft().approx_eq(&a.rev(), 1e-10));
        assert!(b_repr.rev().rev().igfft().approx_eq(&b, 1e-10),);
        assert!((a_repr.rev() * b_repr.rev())
            .igfft()
            .approx_eq(&(b.naive_mul_impl(&a)).rev(), 1e-10));
        assert!(a_repr.flip().igfft().approx_eq(&a.flip(), 1e-10));
        assert!(b_repr.flip().flip().igfft().approx_eq(&b, 1e-10));
    }

    // Basis anticommutativity and metrics
    let metrics = [1., 1., 1., 1., -1., 0., 0., 0.];
    let e = FFTRepr::<A>::basis();
    for (j, ej) in e.iter().enumerate() {
        for i in 0..j {
            assert_eq!(&(&e[i] * ej), &(-ej * &e[i]));
        }
        assert_eq!(ej * ej, FFTRepr::<A>::one() * metrics[j]);
    }
    assert_eq!(
        e.iter().fold(FFTRepr::<A>::one(), |acc, ei| acc * ei).rev(),
        MV::zero().set_by_mask(0b11111111, Complex64::one()).gfft()
    );
}

#[test]
fn fft_repr_mul_test() {
    declare_algebra!(A, [+,+,+,+,0,0], ["w", "x", "y", "z", "e0", "e1"]);
    type MV = Multivector<f64, A>;

    // Check multiplication of basis blades
    for idx in A::index_iter() {
        let ei = MV::zero().set_by_mask(idx, 1.);
        let wfi = ei.gfft();
        //println!("{ei}");
        assert_eq!(wfi.igfft(), ei.clone());

        for jdx in A::index_iter() {
            let ej = MV::zero().set_by_mask(jdx, 1.);
            let wfj = ej.gfft();
            let wfij = &wfi * &wfj;

            let actual = wfij.igfft();
            let expected = ei.naive_mul_impl(&ej);

            assert_eq!(actual, expected);
        }
    }

    for _ in 0..100 {
        let a =
            MV::from_indexed_iter(A::index_iter().map(|idx| (idx, rand::random::<f64>()))).unwrap();
        let b =
            MV::from_indexed_iter(A::index_iter().map(|idx| (idx, rand::random::<f64>()))).unwrap();

        let expected = a.naive_mul_impl(&b);
        let ra = a.gfft();
        let rb = b.gfft();
        let actual = (&ra * &rb).igfft::<f64>();
        assert!(actual.approx_eq(&expected, 1e-10));
    }
}

#[test]
fn wedge_test() {
    declare_algebra!(Gr4, [0, 0, 0, 0], ["w", "x", "y", "z"]);

    let a = Multivector::<f64, Gr4>::from_vector([1., 2., 3., 4.].iter()).unwrap();
    let b = Multivector::<f64, Gr4>::from_vector([4., 3., 2., 1.].iter()).unwrap();
    let c = a.naive_wedge_impl(&b);

    let expected_c = Multivector::<f64, Gr4>::from_indexed_iter(
        [
            (0b0011, 5.),
            (0b0101, 10.),
            (0b0110, 5.),
            (0b1001, 15.),
            (0b1010, 10.),
            (0b1100, 5.),
        ]
        .into_iter(),
    )
    .unwrap();

    assert_eq!(&expected_c, &c);
    assert_eq!(
        expected_c.to_sparse(),
        a.to_sparse().naive_wedge_impl(&b.to_sparse())
    );
    assert_eq!(&expected_c, &a.naive_mul_impl(&b));
}

#[test]
fn rcho_test() {
    // Test for nesting multivectors
    declare_algebra!(C, [-], ["i"]);
    declare_algebra!(H, [-,-], ["I", "J"]);
    declare_algebra!(O, [-,-,-,-,-,-], ["e1", "e2", "e3", "e4", "e5", "e6"]);

    type Cmplx = Multivector<f64, C>;
    type CH = Multivector<Cmplx, H>;
    type CHO = SparseMultivector<CH, O>;

    let one = CHO::one();
    // Complex i
    let i = CHO::from_scalar(CH::from_scalar(Cmplx::basis()[0].clone()));
    // Quaternionic units
    let qi = CHO::from_scalar(CH::basis()[0].clone());
    let qj = CHO::from_scalar(CH::basis()[1].clone());
    let qk = &qi * &qj;
    // Octonionic basis
    let mut e = CHO::basis();
    e.insert(0, one.clone());
    e.push(e.iter().fold(CHO::one(), |acc, c| &acc * c));
    // The basis of the whole RCHO
    let mut rcho_basis = vec![];
    for c in [&one, &i] {
        for q in [&one, &qi, &qj, &qk] {
            for ei in &e {
                rcho_basis.push(ei * q * c);
            }
        }
    }
    let product = rcho_basis.iter().fold(CHO::one(), |acc, c| &acc * c);
    assert_eq!(product, one);
}
