use tclifford::algebra::TAlgebra;
use tclifford::declare_algebra;

use tclifford::Multivector;

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
    declare_algebra!(A, [+,+,+,+,0,0,0], ["w", "x", "y", "z", "e0", "e1", "e3"]);
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
}
