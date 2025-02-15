use core::f64::consts::PI;
use ndarray::arr1;
use ndarray::arr3;
use ndarray::Array1;
use ndarray::Array2;
use std::fmt::Debug;
use tclifford::types::FromComplex;
use tclifford::types::IntoComplex64;
//use ndarray_linalg::Determinant;
use num::complex::{Complex32, Complex64, ComplexFloat};
use num::{One, Zero};
use tclifford::algebra::NonDegenerate;
use tclifford::pga::PGAMV;
use tclifford::types::DivRing;
use tclifford::types::WedgeProduct;
use tclifford::ClError;

use tclifford::algebra::ClAlgebraBase;
use tclifford::algebra::ClBasis;
use tclifford::declare_algebra;
use tclifford::ClAlgebra;
use tclifford::FFTRepr;
use tclifford::Multivector;
use tclifford::SparseMultivector;

fn random_mv_real<A: ClAlgebra>() -> Multivector<f64, A> {
    Multivector::<f64, A>::from_indexed_iter(A::index_iter().map(|idx| (idx, rand::random())))
        .unwrap()
}

fn random_mv_complex<A: ClAlgebra>() -> Multivector<Complex64, A> {
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

fn random_unitary_blade<T, A: ClAlgebra>(blade_dim: usize) -> Multivector<T, A>
where
    T: DivRing + Clone + num::Float,
    rand::distributions::Standard: rand::prelude::Distribution<T>,
{
    let mut blade;
    loop {
        blade = Multivector::<T, A>::one();
        for _ in 0..blade_dim {
            blade = blade.wedge(
                &Multivector::<T, A>::from_vector((0..A::dim()).map(|_| rand::random::<T>()))
                    .unwrap(),
            )
        }
        if !blade.mag2().is_zero() {
            break;
        }
    }
    &blade / blade.mag2().sqrt()
}

fn random_rotor<T, A: ClAlgebra>() -> Multivector<f32, A>
where
    T: DivRing + Clone + num::Float,
    rand::distributions::Standard: rand::prelude::Distribution<T>,
{
    let a = rand::random::<f32>();
    Multivector::<f32, A>::one() * a.cos() + random_unitary_blade::<f32, A>(2) * a.sin()
}

#[test]
fn basic_test() {
    // Test conversions between mv types and operations on them
    declare_algebra!(Cl3, [+,+,+], ["x","y","z"]);
    type MV = Multivector<f64, Cl3>;
    type SMV = SparseMultivector<f64, Cl3>;

    let e = MV::basis();
    let [x, y, z] = e.each_ref();

    let r = x + y + z + x * y;
    let b: SMV = r.grade_extract_as(2);
    assert_eq!(b, x.to_sparse() * y.to_sparse());
    assert_eq!(b.rev(), y.to_sparse() * x.to_sparse());
    assert_eq!(b.dual(), -z.to_sparse());
    assert!((&b * (PI / 4.)).exp().to_dense().approx_eq(
        &(MV::one() * (0.5 as f64).sqrt() + x * y * (0.5 as f64).sqrt()),
        1e-10
    ));
    assert!((b.to_dense() * (PI / 4.)).exp().approx_eq(
        &(MV::one() * (0.5 as f64).sqrt() + x * y * (0.5 as f64).sqrt()),
        1e-10
    ));
    assert_eq!(r.flip().to_sparse(), r.to_sparse().flip());
}

#[test]
fn fft_repr_test() {
    fn fft_test_case<
        const DIM: usize,
        const REPR_DIM: usize,
        T: DivRing + Clone + IntoComplex64 + FromComplex + Debug,
        A: ClAlgebra + ClBasis<DIM> + Debug,
    >() {
        // Check the basic relations between the generators
        let e: [Multivector<T, A>; DIM] = A::basis::<T>();
        let fe: [FFTRepr<A>; DIM] = FFTRepr::<A>::basis();
        for i in 0..e.len() {
            let fei = e[i].fft();
            assert_eq!(fei, fe[i]);
            // Check the the square of fft square is negative identity
            assert_eq!((&fei * &fei), FFTRepr::<A>::one() * A::signaturef()[i]);
            assert_eq!(fei.shape().0, 1 << A::proj_dim());
            assert_eq!(fei.shape().1, REPR_DIM);
            assert_eq!(fei.shape().2, REPR_DIM);
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
        // Check that multiplying representations of all the basis vectors gives
        // a representation of a pseudoscalar (reversed due to the blade order)
        assert_eq!(
            fe.iter()
                .fold(FFTRepr::<A>::one(), |acc, ei| acc * ei)
                .rev()
                .ifft(),
            Multivector::<Complex64, A>::zero().set_by_idx(
                A::real_mask() | A::imag_mask() | A::proj_mask(),
                Complex64::one()
            ),
        );
    }

    fn test_random_mul_real<A: ClAlgebra>() {
        // Check that FFT multiplication corresponds to the naive multiplication
        for _ in 0..10 {
            let a = random_mv_real::<A>();
            let b = random_mv_real::<A>();

            let expected = a.naive_mul_impl(&b);
            let actual = (a.fft() * b.fft()).ifft::<f64>();
            assert!(actual.approx_eq(&expected, 1e-12));
        }
    }

    fn test_random_mul_complex<A: ClAlgebra>() {
        // Check that FFT multiplication corresponds to the naive multiplication
        for _ in 0..10 {
            let a = random_mv_complex::<A>();
            let b = random_mv_complex::<A>();

            let expected = a.naive_mul_impl(&b);
            let ra = a.fft();
            let rb = b.fft();
            let actual = (&ra * &rb).ifft::<Complex64>();
            assert!(actual.approx_eq(&expected, 1e-12));
        }
    }

    fn test_automorphisms<A: ClAlgebra>() {
        // rev and flip test
        for _ in 0..10 {
            let a = random_mv_complex::<A>();
            let b = random_mv_complex::<A>();

            let a_repr = a.fft();
            let b_repr = b.fft();

            assert!(a_repr.rev().ifft().approx_eq(&a.rev(), 1e-12));
            assert!(b_repr.rev().rev().ifft().approx_eq(&b, 1e-12),);
            assert!((a_repr.rev() * b_repr.rev())
                .ifft()
                .approx_eq(&(b.naive_mul_impl(&a)).rev(), 1e-12));
            assert!(a_repr.flip().ifft().approx_eq(&a.flip(), 1e-12));
            assert!(b_repr.flip().flip().ifft().approx_eq(&b, 1e-12));
        }
    }
    fn test_arr3<A: ClAlgebra + Debug>() {
        // Check (from|to)_array3
        let a = random_mv_complex();
        let b = FFTRepr::<A>::from_array3(a.fft().into_array3()).unwrap();
        assert_eq!(a.fft(), b);
        assert_eq!(
            FFTRepr::<A>::from_array3(arr3(&[[[]]])),
            Err(ClError::InvalidShape)
        );
    }

    declare_algebra!(Oct, [-,-,-,-,-,-]);
    fft_test_case::<6, 8, f32, Oct>();
    fft_test_case::<6, 8, f64, Oct>();
    fft_test_case::<6, 8, Complex32, Oct>();
    fft_test_case::<6, 8, Complex64, Oct>();
    test_random_mul_real::<Oct>();
    test_random_mul_complex::<Oct>();
    test_automorphisms::<Oct>();
    test_arr3::<Oct>();

    declare_algebra!(ClOdd, [-,-,-,+,-,+,-]);
    fft_test_case::<7, 16, f32, ClOdd>();
    fft_test_case::<7, 16, f64, ClOdd>();
    fft_test_case::<7, 16, Complex32, ClOdd>();
    fft_test_case::<7, 16, Complex64, ClOdd>();
    test_random_mul_real::<ClOdd>();
    test_random_mul_complex::<ClOdd>();
    test_automorphisms::<ClOdd>();
    test_arr3::<ClOdd>();

    declare_algebra!(ClDg, [+,+,+,+,-,0,0,0]);
    fft_test_case::<8, 8, f32, ClDg>();
    fft_test_case::<8, 8, f64, ClDg>();
    fft_test_case::<8, 8, Complex32, ClDg>();
    fft_test_case::<8, 8, Complex64, ClDg>();
    test_random_mul_real::<ClDg>();
    test_random_mul_complex::<ClDg>();
    test_automorphisms::<ClDg>();
    test_arr3::<ClDg>();

    // In a split-algebra,
    // check that fft of a reversal is a transposition of fft of imaginary_flip
    declare_algebra!(Cl33, [+,-,+,-,+,-]);
    let m = random_mv_real::<Cl33>();
    assert_eq!(
        m.flip_subspace(Cl33::imag_mask()).fft().into_array2().t(),
        m.rev().fft().into_array2()
    );
}

#[test]
fn fft_repr_pow_test() {
    declare_algebra!(A, [+,+,+,+,0,0,0], ["w", "x", "y", "z", "e0", "e1", "e3"]);
    for _ in 0..100 {
        let a = random_mv_complex::<A>();

        let wp = a.fft().pow(5);
        assert!(wp.ifft().approx_eq(
            &a.naive_mul_impl(&a)
                .naive_mul_impl(&a)
                .naive_mul_impl(&a)
                .naive_mul_impl(&a),
            1e-10
        ));
    }
}

#[test]
fn fft_repr_exp_test() {
    declare_algebra!(A, [+,+,+,+,0,0], ["w", "x", "y", "z", "e0", "e1"]);
    type MV = Multivector<Complex64, A>;

    for _ in 0..100 {
        let a = MV::from_indexed_iter(A::index_iter().map(|idx| {
            (
                idx,
                Complex64::new(rand::random::<f64>(), rand::random::<f64>()),
            )
        }))
        .unwrap()
            * Complex64::from(3.);

        let wa = a.fft();
        let we = wa.exp();

        assert!(we.ifft().approx_eq(&a.exp(), 1e-5));
    }
}

#[test]
fn fft_inv_test() {
    declare_algebra!(A, [+,+,+,+,-,-,+]);
    type MV = Multivector<f64, A>;

    let e = MV::basis();

    let mut a = MV::default();
    let mut a_inv_opt = None;
    // A random mv is likely to be invertible, but no guarantees.
    let mut i = 0;
    while a_inv_opt.is_none() {
        a = random_mv_real();
        a_inv_opt = a.fft().inv();
        i += 1;
        if i > 5 {
            assert!(false, "Couldn't invert any of the multivectors");
        }
    }
    let a_inv = a_inv_opt.unwrap();
    assert!((&a.fft() * &a_inv).approx_eq(&FFTRepr::<A>::one(), 1e-10));
    assert!((&a * &a_inv.ifft()).approx_eq(&MV::one(), 1e-10));

    assert_eq!((&MV::one() + &e[0]).fft().inv(), None);
    assert_eq!(
        (&MV::one() + &e[4]).fft().inv().unwrap().ifft(),
        (&MV::one() - &e[4]) / 2.
    );
}

#[test]
fn fmt_test() {
    declare_algebra!(Cl3, [+,+,+], ["x","y","z"]);
    let e = Multivector::<f64, Cl3>::basis();
    let [x, y, z] = e.each_ref();

    let a = x - y + x * y - x * z * 2. + x * y * z / 2000.;
    assert_eq!(
        format!("{}", a), //
        "1 x + (-1) y + (-1) y^x + 2 z^x + (-0.0005) z^y^x"
    );
    assert_eq!(
        format!("{:#}", a), //
        "1 x + (-1) y + 1 x^y + (-2) x^z + 0.0005 x^y^z"
    );
    assert_eq!(
        format!("{:+#.1}", a), //
        "(+1.0) x + (-1.0) y + (+1.0) x^y + (-2.0) x^z + (+0.0) x^y^z"
    );
}

#[test]
fn division_test() {
    declare_algebra!(H, [-,-], ["J", "I"]);
    type CQ = Multivector<f64, H>;
    let e = H::basis::<f64>();
    let one_val = CQ::one();
    let one = &one_val;
    let qj = &e[0];
    let qi = &e[1];
    let qk_val = qi * qj;
    let qk = &qk_val;

    assert_eq!(one / qi, -qi, "{} != {}", one / qi, -qi);
    assert_eq!(one / qj, -qj, "{} != {}", one / qj, -qj);
    assert_eq!(
        one / (qk * 300.),
        -(qk / 300.),
        "{} != {}",
        one / (qk * 300.),
        -(qk / 300.)
    );
}

#[test]
fn wedge_test() {
    declare_algebra!(Gr4, [0, 0, 0, 0], ["w", "x", "y", "z"]);

    let a = Multivector::<f64, Gr4>::from_vector_ref([1., 2., 3., 4.].iter()).unwrap();
    let b = Multivector::<f64, Gr4>::from_vector_ref([4., 3., 2., 1.].iter()).unwrap();
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
fn vee_test() {
    declare_algebra!(Cl3, [+,+,+], ["v1", "v2", "v3"]);
    let a = random_mv_complex::<Cl3>();
    let b = random_mv_complex::<Cl3>();

    assert!(a
        .naive_meet_impl(&b)
        .approx_eq(&a.dual().wedge(&b.dual()).undual(), 1e-12));

    assert!(a
        .meet(&b)
        .approx_eq(&a.dual().wedge(&b.dual()).undual(), 1e-12));

    assert!(a
        .meet(&b)
        .dual()
        .approx_eq(&a.dual().wedge(&b.dual()), 1e-12));

    let e = Multivector::<f64, Cl3>::basis();
    let [v1, v2, v3] = e.each_ref();
    let u1 = v3 + v2 - v1 * v2 * 3. - v1 * v3 * 2. + v2 * v3;
    let u2 = v1 * v2 * v3 - v1 * v2 * 7. - v1 * v3 * 2. + v2 * v3 * 3.;
    let t1 = u1.dual().wedge(&u2.dual());
    let t2 = u1.meet(&u2).dual();
    println!("t1 = {}", t1);
    println!("t2 = {}", t2);

    let expected =
        v1 + v2 * 2. - v3 * 3. - v1 * v2 * 3. + v1 * v3 - v2 * v3 * 8. - v1 * v2 * v3 * 5.;
    assert_eq!(t1, expected);
    assert_eq!(t2, expected);

    //////////////////////

    declare_algebra!(Cl8, [+,+,+,+,+,+,+,+]);
    let a = random_mv_complex::<Cl8>();
    let b = random_mv_complex::<Cl8>();

    assert!(a
        .naive_meet_impl(&b)
        .approx_eq(&a.dual().wedge(&b.dual()).undual(), 1e-12));

    assert!(a
        .meet(&b)
        .approx_eq(&a.dual().wedge(&b.dual()).undual(), 1e-12));

    assert!(a
        .meet(&b)
        .dual()
        .approx_eq(&a.dual().wedge(&b.dual()), 1e-12));
}

#[test]
fn dual_test() {
    declare_algebra!(Cl3, [+,+,+], ["x","y","z"]);
    let e = Cl3::basis::<f64>();
    let [x, y, z] = e.each_ref();
    // Check that the blade and it's dual rotate together.
    let mut theta = 0.;
    while theta <= PI {
        let rot = (x * y * theta).exp();

        // rotate vector
        let rx = rot.rev() * x * &rot;
        let rxd = rot.rev() * x.dual() * &rot;
        assert!(rx.dual().approx_eq(&rxd, 1e-12));
        assert!(rx.approx_eq(&rxd.dual(), 1e-12));

        // rotate 2-blade
        let v = (x + y).wedge(z);
        let rv = rot.rev() * &v * &rot;
        let rvd = rot.rev() * v.dual() * &rot;
        assert!(rv.dual().approx_eq(&rvd, 1e-12));
        assert!(rv.approx_eq(&rvd.dual(), 1e-12));

        theta += PI / 20.
    }

    declare_algebra!(Cl7, [-,-,-,-,-,-,-]);
    type MV = Multivector<f64, Cl7>;

    let mut blade = MV::one();
    for i in 1..=7 {
        blade = blade.wedge(&(MV::from_vector((0..7).map(|_| rand::random())).unwrap() * 2.));

        let p = blade.dual() * &blade;

        println!("{i}:: {:.80}...", format!("{:.4}", blade));
        println!("    {:.4}", p.filter(|_, c| *c > 1e-12));

        assert!(p.grade_extract(7).approx_eq(&p, 1e-12)); // There are no other coeffs besides pseudoscalar
        assert!(p.get_by_idx(0b1111111) > 0.); // And the coefficient in front of it is positive
        assert_eq!(blade.dual().dual(), blade);
    }

    let m = random_mv_real::<Cl7>();
    assert_eq!(m.dual().dual(), m);

    // By the Hodge star definition:
    // a.wedge(b.dual()) == a.dot(b) * I
    // (where dot ignores metrics, since Hodge star is defined for Grassmann algebras)

    // Check for metrics-agnostic dual
    let a = MV::from_vector((0..7).map(|_| rand::random())).unwrap();
    let b = MV::from_vector((0..7).map(|_| rand::random())).unwrap();
    assert!(
        (a.wedge(&b.dual()).get_by_idx(0b1111111) - a.extract_vector().dot(&b.extract_vector()))
            .abs()
            < 1e-12
    );
    assert!((a.wedge(&b.dual()).get_by_idx(0b1111111) - a.vsdot(&b)).abs() < 1e-12);

    // Check for metrics-dependent dual (i.e. just a multiplication by pseudoscalar)
    declare_algebra!(Cl7A, [+, +, +, -, 0, 0, 0, 0, 0]);
    type MVA = Multivector<f64, Cl7A>;

    let ps = MVA::zero().set_by_idx(0b111111111, 1.);

    let a = MVA::from_vector((0..9).map(|_| rand::random())).unwrap();
    let b = MVA::from_vector((0..9).map(|_| rand::random())).unwrap();

    let metric = Array2::from_diag(&Array1::from_iter(Cl7A::signaturef()));
    let vdot1 = metric.dot(&a.extract_vector()).t().dot(&b.extract_vector());
    let vdot2 = a.vdot(&b);
    let vdot3 = a.wedge(&(&b * &ps)).dual();
    let vdot4 = (&a * &b).scalar_part();

    assert!((&vdot1 - &vdot2).abs() < 1e-12);
    assert!((&vdot2 - &vdot3.scalar_part()).abs() < 1e-12);
    assert!((&vdot1 - &vdot4).abs() < 1e-12);

    declare_algebra!(Cl2, [+,+]);
    let [x, _y] = Cl2::basis::<f32>();
    println!("*x = {}; **x = {}", x.dual(), x.dual().undual());
    assert_eq!(x, x.dual().undual());

    declare_algebra!(ST, [-,+,+,+]);
    let [t, x, _y, _z] = ST::basis::<f32>();
    println!(
        "*x = {}; **x = {}",
        (&t + &x).dual(),
        (&t + &x).dual().undual()
    );
    assert_eq!((&t + &x), (&t + &x).dual().undual());
    assert_eq!((&t + &x), (&t + &x).undual().dual());

    // Check consistency with Grassmann.jl (up to a sign)
    declare_algebra!(TestA, [+,+,+,-,-,0,0]);
    let e = TestA::basis::<f32>();
    let [v1, v2, v3, v4, v5, v6, v7] = &e;
    let a = -v3 * 5. + v1 * v5 * v6 - v1 * v4 * v6 * v7
        + v2 * v3 * v6 * v7
        + v2 * v4 * v5 * v6 * v7 * 4.
        + v1 * v2 * v3 * v4 * v5 * v6 * 2.;
    println!("{}", a.dual());
    let expected = -(v7 * 2. - v1 * v3 * 4. + v1 * v4 * v5 - v2 * v3 * v5 + v2 * v3 * v4 * v7
        - v1 * v2 * v4 * v5 * v6 * v7 * 5.);
    println!("{}", expected);
    assert_eq!(a.dual(), expected);
    assert_eq!(a.undual().dual(), a);
    assert_eq!(a, expected.undual());
}

#[test]
fn vdot_test() {
    declare_algebra!(A, [+,+,+,-,-,-,0,0]);

    let a = random_mv_real::<A>();
    let b = random_mv_real::<A>();
    assert!((a.vdot(&b) - (&a * &b).scalar_part()).abs() < 1e-12);
    assert!((a.vdot(&b) - (a.fft() * b.fft()).ntrace()).abs() < 1e-12);
}

#[test]
fn magnitude_test() {
    declare_algebra!(A, [+,+,+,-,-,-,0,0]);
    let e = A::basis::<f64>();
    let [x, y, z, a, b, c, e0, e1] = e.each_ref();

    assert_eq!((x + y + e0).mag2(), 2.0);
    assert_eq!((x * z + e0 * a).mag2(), 1.0);
    assert_eq!((b * c * e1).mag2(), 0.0);

    let mv1 = random_mv_complex::<A>();
    let mv2 = random_mv_complex::<A>().grade_extract(1);

    assert!((mv1.revcm() * &mv1)
        .grade_extract(0)
        .approx_eq(&Multivector::<Complex64, A>::one(), 1e-12));

    assert!(mv2.cmag2().im.abs() < 1e-12);

    let dg = x * (y + a);
    assert_eq!(dg.mag2(), 0.0);
    println!("{}", dg.mag2());
    println!("{}", dg.revm());
    println!("{}", (x * y).mag2());

    let rot = (x * y + y * z + a * b).exp() * 5.;
    let m = rot.revm() * b * &rot;
    assert!((m.mag2() - b.mag2()).abs() < 1e-12);
}

/// This "test" applies random rotors to random normalized multivectors
/// in 2 ways: using naive multiplication and FFT multiplication.
/// It checks
/// 1. How un-normalized it became
/// 2. How much the multiplication implementations disagree with each other.
///
/// Last run results:
/// ```plaintext
/// Naive: avg: 5.182e-8, max: 4.432e-4
/// FFT  : avg: 5.182e-8, max: 4.432e-4
/// Diff : avg: 5.589e-12, max: 6.898e-8
/// ```
#[test]
#[ignore]
fn error_estimation() {
    declare_algebra!(A, [+,+,+,-,-,-,+,0]);

    let mut rot_err = 0.0f64;
    let mut rot_err_max = 0.0f64;
    let mut rot_fft_err = 0.0f64;
    let mut rot_fft_err_max = 0.0f64;
    let mut disagreement = 0.0f64;
    let mut disagreement_max = 0.0f64;

    const N: i32 = 100000;

    for _ in 0..N {
        // Make a normalized vector a_ n
        let mut mag2 = 0.;
        let mut a = Multivector::<f64, A>::zero();
        while mag2.abs() < 1e-12 {
            // avoid possible division by 0
            a = random_mv_real::<A>();
            mag2 = a.rev().vdot(&a);
        }
        let a_n = &a / mag2.abs().sqrt();
        assert!((a_n.rev().vdot(&a_n).abs() - 1.0).abs() < 1e-10);

        // Make a random rotor
        let rot = Multivector::<f64, A>::from_indexed_iter(
            A::grade_index_iter(2).map(|idx| (idx, rand::random())),
        )
        .unwrap()
        .exp();

        let rotated_a = rot.rev() * &a_n * &rot;
        let fft_rotated_a = (rot.rev().fft() * a_n.fft() * rot.fft()).ifft::<f64>();

        let rmag = rotated_a.rev().vdot(&rotated_a).abs();
        let fft_rmag = fft_rotated_a.rev().vdot(&fft_rotated_a).abs();

        rot_err += (rmag - 1.0).abs();
        rot_err_max = rot_err_max.max((rmag - 1.0).abs());
        rot_fft_err += (fft_rmag - 1.0).abs();
        rot_fft_err_max = rot_err_max.max((fft_rmag - 1.0).abs());
        disagreement += (fft_rmag - rmag).abs();
        disagreement_max = disagreement_max.max((fft_rmag - rmag).abs());
    }

    let avg_err = rot_err / (N as f64);
    let avg_fft_err = rot_fft_err / (N as f64);
    let avg_disagreement = disagreement / (N as f64);

    println!("Naive: avg: {:.3e}, max: {:.3e}", avg_err, rot_err_max);
    println!(
        "FFT  : avg: {:.3e}, max: {:.3e}",
        avg_fft_err, rot_fft_err_max
    );
    println!(
        "Diff : avg: {:.3e}, max: {:.3e}",
        avg_disagreement, disagreement_max
    );
}

#[test]
fn sparse_test() {
    declare_algebra!(Cl20, [+,+,+,+,+,+,+,+,+,+,+,+,+,+,+,+,+,+,+,+]);
    type MV = SparseMultivector<f64, Cl20>;

    let bivec = MV::from_indexed_iter(Cl20::grade_index_iter(2).map(|idx| (idx, 1.))).unwrap();
    assert_eq!(
        (&bivec.rev() * &bivec).grade_extract(0),
        MV::from_scalar(num_integer::binomial(20, 2) as f64)
    );
    let a = MV::from_vector((0..20).map(|_| rand::random::<f64>())).unwrap();
    let b = MV::from_vector((0..20).map(|_| rand::random::<f64>())).unwrap();

    let blade = a.wedge(&b);

    println!("{:.02e}", (&blade * &blade).grade_extract(0));
    println!(
        "{:.02e}",
        (&blade * &blade).filtered(|_, c| c.abs() > 1e-14)
    );

    assert_eq!(
        (&blade * &blade).grades_extract(&[0]),
        (&blade * &blade).filtered(|_, c| c.abs() > 1e-14)
    );
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
    let [qi, qj] = CH::basis().map(CHO::from_scalar);
    let qk = &qi * &qj;
    // Octonionic basis
    let mut e = CHO::basis().to_vec();
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

#[test]
fn trace_test() {
    declare_algebra!(Cl22, [+,-,+,-]);
    let a = Array2::from_diag(&arr1(&[
        Complex64::from(1.),
        Complex64::from(2.),
        Complex64::from(3.),
        Complex64::from(4.),
    ]));
    let aa = FFTRepr::<Cl22>::from_array2(a).unwrap();
    assert_eq!(aa.trace().re / 4., aa.ifft::<f64>().get_by_idx(0));

    let b = random_mv_complex::<Cl22>();
    assert!((b.fft().trace() / 4. - b.get_by_idx(0)).abs() < 1e-12);

    declare_algebra!(A, [+,+,+,+,+,0,0]);
    let c = random_mv_complex::<A>();
    let fc = c.fft();

    assert!(((fc.trace() / (fc.shape().1 as f64)) - c.get_by_idx(0)).abs() < 1e-12);
    assert!(SparseMultivector::<Complex64, A>::from_scalar(fc.ntrace())
        .approx_eq(&c.grade_extract(0).to_sparse(), 1e-12));
}

#[test]
fn irrep_test() {
    fn irrep_test_internal<const RDIM: usize, A: ClAlgebra + NonDegenerate>() {
        let blade = random_unitary_blade::<f64, A>(2);

        let angle = PI / 10.;
        let rot: FFTRepr<A> = (blade * angle).fft().exp();

        let (even_spin_repr, odd_spin_repr) = rot.into_half_spin_repr();
        // Check that the even representation has the expected order (A^10 == -I)
        let mut actual = Array2::eye(RDIM);
        for _ in 0..10 {
            actual = actual.dot(&even_spin_repr);
        }
        assert!((actual - (-Array2::<Complex64>::eye(RDIM)))
            .iter()
            .all(|c| c.abs() < 1e-12));

        // Check that the odd representation has the expected order (A^10 == -I)
        let mut actual = Array2::eye(RDIM);
        for _ in 0..10 {
            actual = actual.dot(&odd_spin_repr);
        }
        assert!((actual - (-Array2::<Complex64>::eye(RDIM)))
            .iter()
            .all(|c| c.abs() < 1e-12));
    }

    declare_algebra!(Cl8, [+,+,+,+,+,+,+,+]);
    irrep_test_internal::<8, Cl8>();
    declare_algebra!(Cl10, [+,+,+,+,+,+,+,+,+,+]);
    irrep_test_internal::<16, Cl10>();
}

/// Test for examples in README.md
#[test]
fn examples_test() {
    // Simple PGA Example

    // Create types for the algebra and multivectors in it
    declare_algebra!(PGA2, [+,+,0], ["x", "y", "n"]);
    type M = Multivector<f64, PGA2>;

    // Create PGA points
    let ax = M::pga_point([10., 0.]).unwrap();
    let ay = M::pga_point([0., 8.]).unwrap();

    // Line connecting the two points
    let line1 = ax.meet(&ay);

    // Line defined by the equation `-2 x + 5 y - 10 = 0`
    let line2 = M::pga_hyperplane([-2., 5., -10.]).unwrap();

    // Find the intersection point
    let intersection = line1.wedge(&line2);
    println!("{:?}", intersection.pga_extract_point()); // [5.0, 4.0]
    assert_eq!(intersection.pga_extract_point(), [5., 4.]);

    // Advanced Example

    // Create types for the algebra and multivectors in it
    declare_algebra!(Cl8, [+,+,+,+,+,+,+,+]);
    type MV = Multivector<f64, Cl8>;

    // Generate a random unitary blade.
    let a = MV::from_vector((0..8).map(|_| rand::random::<f64>() - 0.5)).unwrap();
    let b = MV::from_vector((0..8).map(|_| rand::random::<f64>() - 0.5)).unwrap();
    let mut blade = a.wedge(&b);
    blade = &blade / blade.mag2().sqrt(); // normalize it

    // Produce a rotor generated by rotation in this plane by a random angle.
    let angle = PI * rand::random::<f64>();
    // Or, rather, its full-spin representation, because exponents work much faster on them.
    let rot: FFTRepr<Cl8> = (blade * angle).fft().exp();

    // Compute the SO(8) representation from the rotor action on the basis vectors.
    let e: [MV; 8] = MV::basis();
    let mut so_repr = ndarray::Array2::default((8, 8));
    for (i, ei) in e.iter().enumerate() {
        // Apply the rotor to each basis vector
        let rotated_ei: MV = (rot.rev() * ei.fft() * &rot).ifft();
        // And collect the results into a matrix
        let ei_column = rotated_ei.extract_vector();
        so_repr
            .index_axis_mut(ndarray::Axis(1), i)
            .assign(&ei_column);
    }
    // Check that the resulting matrix indeed has a determinant of 1 (allowing some floating point error)
    //assert!((so_repr.det().unwrap() - 1.0).abs() < 1e-12);

    // Irreducible spin representations are built-in!
    let (even_spin_repr, odd_spin_repr) = rot.into_half_spin_repr();

    println!("The 3 irreducible representations of this Spin(8) element are:");
    println!("{:.2}", so_repr);
    println!("{:.2}", even_spin_repr);
    println!("{:.2}", odd_spin_repr);
}

#[cfg(not(debug_assertions))]
mod benchmarks {
    use super::*;
    use std::hint::black_box;
    use std::time;
    use tclifford::clifft;

    #[test]
    fn fft_bench() {
        declare_algebra!(Cl8, [+,+,+,+,+,+,+,+]);

        // Check overhead of FFTRepr compared to the raw clifft
        let b = random_mv_real::<Cl8>();
        let ts = time::Instant::now();
        for _ in 0..10000 {
            let _ = black_box(
                clifft::iclifft(
                    clifft::clifft(b.coeff_array_view()).unwrap().view(), //
                )
                .unwrap(),
            );
        }
        println!("raw_fft {:?}", ts.elapsed());

        let ts = time::Instant::now();
        for _ in 0..10000 {
            let _ = black_box(b.fft().ifft::<f64>());
        }
        println!("    fft {:?}", ts.elapsed());

        let x = random_mv_real::<Cl8>();
        let y = random_mv_real::<Cl8>();

        let mut fx = clifft::clifft(x.coeff_array_view()).unwrap();
        let fy = clifft::clifft(y.coeff_array_view()).unwrap();

        let mut gx = x.fft();
        let gy = y.fft();

        let ts = time::Instant::now();
        for _ in 0..100000 {
            fx = black_box(fx.dot(&fy));
        }
        println!("dot {:?}", ts.elapsed());

        let ts = time::Instant::now();
        for _ in 0..100000 {
            gx = black_box(&gx * &gy);
        }
        println!("mul {:?}", ts.elapsed());
    }

    #[test]
    fn low_dim_bench() {
        fn bench_mul<A: ClAlgebra>(count: u32) -> time::Duration {
            let arr1: Vec<_> = (0..count).map(|_| random_mv_real::<A>()).collect();
            let arr2: Vec<_> = (0..count).map(|_| random_mv_real::<A>()).collect();
            let ts = time::Instant::now();
            for a in &arr1 {
                for b in &arr2 {
                    let _ = black_box(a * b);
                }
            }
            ts.elapsed() / (count * count)
        }

        fn bench_rvr<A: ClAlgebra>(count: u32) -> time::Duration {
            let arr1: Vec<_> = (0..count).map(|_| random_rotor::<f32, A>()).collect();
            let arr2: Vec<_> = (0..count)
                .map(|_| random_unitary_blade::<f32, A>(1))
                .collect();
            let ts = time::Instant::now();
            for r in &arr1 {
                for v in &arr2 {
                    let _ = black_box(r.revm() * v * r);
                }
            }
            ts.elapsed() / (count * count)
        }

        declare_algebra!(Cl2, [-,-]);
        declare_algebra!(Cl3, [-,-,-]);
        declare_algebra!(Cl4, [-,-,-,-]);
        declare_algebra!(Cl5, [-,-,-,-,-]);

        println!("dim 2 avg mul time: {:?}", bench_mul::<Cl2>(1000));
        println!("dim 3 avg mul time: {:?}", bench_mul::<Cl3>(1000));
        println!("dim 4 avg mul time: {:?}", bench_mul::<Cl4>(1000));
        println!("dim 5 avg mul time: {:?}", bench_mul::<Cl5>(1000));
        println!("---");
        println!("dim 2 avg rvr time: {:?}", bench_rvr::<Cl2>(1000));
        println!("dim 3 avg rvr time: {:?}", bench_rvr::<Cl3>(1000));
        println!("dim 4 avg rvr time: {:?}", bench_rvr::<Cl4>(1000));
        println!("dim 5 avg rvr time: {:?}", bench_rvr::<Cl5>(1000));
    }
}
