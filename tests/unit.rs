use core::f64::consts::PI;
use ndarray::arr1;
use ndarray::arr2;
use ndarray::Array1;
use ndarray::Array2;
use num::complex::ComplexFloat;
use num::{One, Zero};
use std::hint::black_box;
use std::time;
use tclifford::algebra_ifft::InverseClifftRepr;
use tclifford::types::WedgeProduct;
use tclifford::ClError;

use tclifford::declare_algebra;
use tclifford::ClAlgebra;
use tclifford::FFTRepr;
use tclifford::Multivector;
use tclifford::SparseMultivector;

use num::complex::Complex64;

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

#[test]
fn basic_test() {
    // Test conversions between mv types and operations on them
    declare_algebra!(Cl3, [+,+,+], ["x","y","z"]);
    type MV = Multivector<f64, Cl3>;
    type SMV = SparseMultivector<f64, Cl3>;

    let e = MV::basis();
    let x = &e[0];
    let y = &e[1];
    let z = &e[2];

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
    declare_algebra!(A, [+,+,+,+,-,0,0,0], ["w", "x", "y", "z", "t", "e0", "e1", "e3"]);
    type MV = Multivector<Complex64, A>;

    // rev and flip test
    for _ in 0..10 {
        let a: MV = random_mv_complex();
        let b: MV = random_mv_complex();

        let a_repr = a.fft();
        let b_repr = b.fft();

        assert!(a_repr.rev().ifft().approx_eq(&a.rev(), 1e-10));
        assert!(b_repr.rev().rev().ifft().approx_eq(&b, 1e-10),);
        assert!((a_repr.rev() * b_repr.rev())
            .ifft()
            .approx_eq(&(b.naive_mul_impl(&a)).rev(), 1e-10));
        assert!(a_repr.flip().ifft().approx_eq(&a.flip(), 1e-10));
        assert!(b_repr.flip().flip().ifft().approx_eq(&b, 1e-10));
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
        MV::zero().set_by_mask(0b11111111, Complex64::one()).fft()
    );

    let a = random_mv_complex();
    let b = FFTRepr::<A>::from_array3(a.clone().fft().into_array3()).unwrap();
    assert_eq!(a.fft(), b);

    let mut arr = a.fft().into_array3();
    let mut arr2 = arr.clone();
    arr[(1, 1, 1)] += 1.0;
    assert_eq!(
        FFTRepr::<A>::from_array3(arr),
        Err(ClError::NotARepresentation)
    );
    arr2[(0, 2, 0)] += 1.0;
    assert_eq!(
        FFTRepr::<A>::from_array3(arr2),
        Err(ClError::NotARepresentation)
    );

    declare_algebra!(Cl2, [+,+], ["x","y"]);
    let e = Multivector::<f64, Cl2>::basis();
    assert_eq!(
        e[0].fft().into_array2(),
        ndarray::arr2(&[
            [Complex64::zero(), Complex64::one()],
            [Complex64::one(), Complex64::zero()]
        ])
    );
    assert_eq!(
        e[1].fft().into_array2(),
        ndarray::arr2(&[
            [Complex64::zero(), Complex64::i()],
            [-Complex64::i(), Complex64::zero()]
        ])
    );
    assert_eq!(
        FFTRepr::<Cl2>::from_array2(ndarray::arr2(&[
            [-Complex64::i(), Complex64::zero()],
            [Complex64::zero(), Complex64::i()]
        ]))
        .unwrap()
        .ifft(),
        &e[0] * &e[1]
    );
}

#[test]
fn fft_repr_mul_test() {
    declare_algebra!(A, [+,+,+,+,0,0], ["w", "x", "y", "z", "e0", "e1"]);
    type MV = Multivector<f64, A>;

    // Check multiplication of basis blades
    for idx in A::index_iter() {
        let ei = MV::zero().set_by_mask(idx, 1.);
        let wfi = ei.fft();
        //println!("{ei}");
        assert_eq!(wfi.ifft(), ei.clone());

        for jdx in A::index_iter() {
            let ej = MV::zero().set_by_mask(jdx, 1.);
            let wfj = ej.fft();
            let wfij = &wfi * &wfj;

            let actual = wfij.ifft();
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
        let ra = a.fft();
        let rb = b.fft();
        let actual = (&ra * &rb).ifft::<f64>();
        assert!(actual.approx_eq(&expected, 1e-10));
    }
}

#[test]
fn fft_repr_odd_dim_test() {
    declare_algebra!(Cl3, [+,+,+], ["x", "y", "z"]);
    let one = Multivector::<f64, Cl3>::from_scalar(1.);
    let fone = one.raw_fft().unwrap();
    assert_eq!(one, Cl3::raw_ifft::<f64>(fone.view()).unwrap());

    declare_algebra!(Cl5, [+,+,+,+,+], ["x", "y", "z", "a", "b"]);
    let v = Multivector::<f64, Cl5>::from_vector([1., 2., 3., 4., 5.].into_iter()).unwrap();
    let fv = v.raw_fft().unwrap();
    assert_eq!(v, Cl5::raw_ifft::<f64>(fv.view()).unwrap());

    declare_algebra!(Cl502, [+,+,+,+,+,0,0], ["x", "y", "z", "a", "b", "E", "G"]);
    type MV = Multivector<f64, Cl502>;
    for _ in 0..100 {
        let mv = MV::from_indexed_iter(Cl502::index_iter().map(|idx| (idx, rand::random::<f64>())))
            .unwrap();
        let m = mv.fft();
        let actual = m.ifft();
        assert!(mv.approx_eq(&actual, 1e-15));
    }

    declare_algebra!(Cl303, [+,+,+,0,0,0], ["x", "y", "z", "E", "F", "G"]);
    type MV303 = Multivector<f64, Cl303>;
    for _ in 0..100 {
        let mv =
            MV303::from_indexed_iter(Cl303::index_iter().map(|idx| (idx, rand::random::<f64>())))
                .unwrap();
        let m = mv.fft();
        let actual = m.ifft();
        assert!(mv.approx_eq(&actual, 1e-15));
    }
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

        assert!(we.ifft().approx_eq(&a.exp(), 2.));
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
    let x = &e[0];
    let y = &e[1];
    let z = &e[2];

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
    declare_algebra!(Cl3, [+,+,+]);
    let a = random_mv_complex::<Cl3>();
    let b = random_mv_complex::<Cl3>();

    assert!(a
        .naive_vee_impl(&b)
        .approx_eq(&a.dual().wedge(&b.dual()).dual(), 1e-12));

    assert!(a
        .regressive_product(&b)
        .approx_eq(&a.dual().wedge(&b.dual()).dual(), 1e-12));

    //////////////////////

    declare_algebra!(Cl8, [+,+,+,+,+,+,+,+]);
    let a = random_mv_complex::<Cl8>();
    let b = random_mv_complex::<Cl8>();

    assert!(a
        .naive_vee_impl(&b)
        .approx_eq(&a.dual().wedge(&b.dual()).dual(), 1e-12));

    assert!(a
        .regressive_product(&b)
        .approx_eq(&a.dual().wedge(&b.dual()).dual(), 1e-12));
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
    assert_eq!(aa.trace().re / 4., aa.ifft::<f64>().get_by_mask(0));

    let b = random_mv_complex::<Cl22>();
    assert!((b.fft().trace() / 4. - b.get_by_mask(0)).abs() < 1e-12);

    declare_algebra!(A, [+,+,+,+,+,0,0]);
    let c = random_mv_complex::<A>();
    let fc = c.fft();

    assert!(((fc.trace() / (fc.shape().1 as f64)) - c.get_by_mask(0)).abs() < 1e-12);
    assert!(SparseMultivector::<Complex64, A>::from_scalar(fc.ntrace())
        .approx_eq(&c.grade_extract(0).to_sparse(), 1e-12));
}

#[test]
#[cfg(not(debug_assertions))]
fn fft_bench() {
    declare_algebra!(Cl8, [+,+,+,+,+,+,+,+]);

    let b = random_mv_real::<Cl8>();

    let ts = time::Instant::now();
    for _ in 0..10000 {
        let _ = black_box(Cl8::raw_ifft::<f64>(b.raw_fft().unwrap().view()));
    }
    println!("raw_fft {:?}", ts.elapsed());

    let ts = time::Instant::now();
    for _ in 0..10000 {
        let _ = black_box(b.fft().ifft::<f64>());
    }
    println!("    fft {:?}", ts.elapsed());

    let x = random_mv_real::<Cl8>();
    let y = random_mv_real::<Cl8>();

    let mut fx = x.raw_fft().unwrap();
    let fy = y.raw_fft().unwrap();

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
#[cfg(not(debug_assertions))]
fn wfft_bench() {
    declare_algebra!(A, [+,+,+,+,+,-,0,0,0], ["e0","e1","e2","e3","e4","e5","n0","n1","n2"]);
    //type MV = Multivector<f64, A>;

    let a = random_mv_real::<A>().fft();
    let b = random_mv_real::<A>().fft();

    let ts = time::Instant::now();
    for _ in 0..10000 {
        let _ = black_box(&a * &b);
    }
    println!("mul: {:?}", ts.elapsed());
}
/*
Reference:
mul: 225.503855ms
mul: 227.737987ms
mul: 228.743096ms
mul: 229.495629ms
mul: 229.821396ms
mul: 230.190217ms
mul: 230.469068ms
mul: 232.822678ms
mul: 232.865549ms
mul: 234.022987ms
mul: 234.948563ms
mul: 235.350182ms
mul: 236.916854ms
mul: 239.372875ms
mul: 242.979197ms
mul: 248.881104ms
mul: 251.172618ms
mul: 254.597744ms
mul: 258.026662ms
mul: 261.127349ms
mul: 264.031057ms

*/
