use crate::clifft::iclifft;
use crate::clifft::iclifft_into;
use crate::types::FromComplex;
use crate::types::IndexType;
use crate::ClError;
use crate::Multivector;
use crate::Ring;
use crate::TAlgebra;
use ndarray::s;
use ndarray::Axis;
use ndarray::{Array1, Array3, ArrayView2, ArrayView3, ArrayViewMut3};
use num::complex::Complex64;
use num::Integer;
use num::One;

#[cfg(test)]
use {crate::declare_algebra, num::Zero, std::hint::black_box, std::time};

pub trait InverseWfftRepr: TAlgebra {
    fn iwfft<T>(wm: ArrayView3<Complex64>) -> Result<Multivector<T, Self>, ClError>
    where
        T: Ring + FromComplex + Clone,
        Self: Sized;
}

impl<A> InverseWfftRepr for A
where
    A: TAlgebra,
{
    fn iwfft<T>(wm: ArrayView3<Complex64>) -> Result<Multivector<T, Self>, ClError>
    where
        T: Ring + FromComplex + Clone,
        Self: Sized,
    {
        if !A::normalized_for_wfft() {
            return Err(ClError::FFTConditionsNotMet);
        }
        let non_degen_dim = (A::real_mask() | A::imag_mask()).count_ones();
        let repr_non_degen_dim = ((non_degen_dim + 1) / 2) * 2;
        let degen_dim = A::proj_mask().count_ones();
        let repr_dim = degen_dim + repr_non_degen_dim;

        let mut ret_arr = Array1::<Complex64>::zeros(1 << repr_dim);
        let step = 1 << repr_non_degen_dim;
        for (i, m) in wm.axis_iter(Axis(0)).enumerate() {
            iclifft_into(
                m,
                ret_arr.view_mut().slice_mut(s![i * step..(i + 1) * step]),
            )?;
        }

        if non_degen_dim.is_even() {
            Multivector::<T, A>::from_indexed_iter(A::decomplexified_iter(ret_arr.indexed_iter()))
        } else {
            // Drop half of coefficients for odd-non-degen-dimensional algebras
            // This is probably better done by a custom strided view of array
            // (which might not be possible unless there is some reshaping involved
            //  https://stackoverflow.com/questions/65491179/how-to-create-ndarrayarrayview-with-custom-strides)
            let effective_proj_mask = A::proj_mask() << 1;
            let effective_full_mask = effective_proj_mask | A::real_mask() | A::imag_mask();

            Multivector::<T, A>::from_indexed_iter(A::decomplexified_iter(
                ret_arr.indexed_iter().filter_map(|(idx, c)| {
                    if (idx & !effective_full_mask) != 0 {
                        None
                    } else {
                        let effective_idx = ((idx & effective_proj_mask) >> 1)
                            | (idx & (A::real_mask() | A::imag_mask()));
                        Some((effective_idx, c))
                    }
                }),
            ))
        }
    }
}

pub trait InverseClifftRepr: TAlgebra {
    fn decomplexified_iter<'a, T, It>(iter: It) -> impl Iterator<Item = (IndexType, T)>
    where
        T: Ring + Clone + FromComplex + 'a,
        It: Iterator<Item = (IndexType, &'a Complex64)>;

    /// Restore the multivector from its matrix representation ([`Multivector::fft`]).
    ///
    /// Applicable only when the multivector has complex coefficients.
    fn ifft<T>(m: ArrayView2<Complex64>) -> Result<Multivector<T, Self>, ClError>
    where
        T: Ring + Clone + FromComplex,
        Self: Sized;
}

impl<A> InverseClifftRepr for A
where
    A: TAlgebra,
{
    fn decomplexified_iter<'a, T, It>(iter: It) -> impl Iterator<Item = (IndexType, T)>
    where
        T: Ring + Clone + FromComplex + 'a,
        It: Iterator<Item = (IndexType, &'a Complex64)>,
    {
        const I_REV_POWERS: [Complex64; 4] = [
            Complex64 { re: 1., im: 0. },
            Complex64 { re: 0., im: 1. },
            Complex64 { re: -1., im: 0. },
            Complex64 { re: 0., im: -1. },
        ];
        iter.map(|(idx, c)| {
            (
                idx,
                T::from_complex(
                    I_REV_POWERS[((idx & A::imag_mask()).count_ones() as usize) % 4] * c,
                ),
            )
        })
    }

    fn ifft<T>(m: ArrayView2<Complex64>) -> Result<Multivector<T, Self>, ClError>
    where
        T: Ring + Clone + FromComplex,
        Self: Sized,
    {
        if A::proj_mask() != 0 {
            return Err(ClError::FFTConditionsNotMet);
        }

        let coeffs = iclifft(m)?;
        // This drops upper half for odd-dimensional algebras
        let cview = coeffs.slice(s![0..(1 << A::dim())]);
        let ret_coeff_iter = Self::decomplexified_iter(cview.indexed_iter());
        Multivector::<T, Self>::from_indexed_iter(ret_coeff_iter)
    }
}

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

pub fn wmul(
    a: ArrayView3<Complex64>,
    b: ArrayView3<Complex64>,
) -> Result<Array3<Complex64>, ClError> {
    if a.shape() != b.shape() {
        return Err(ClError::InvalidShape);
    }
    if a.dim().0 == 1 {
        // Optimization for non-degen signatures that avoids extra memory allocations
        return Ok(a
            .index_axis(Axis(0), 0)
            .dot(&b.index_axis(Axis(0), 0))
            .into_shape_clone(a.dim())
            .unwrap());
    }
    let mut ret = Array3::zeros(a.dim());
    let mut ca = a.into_owned();
    wmul_impl(ca.view_mut(), b, ret.view_mut());
    Ok(ret)
}

pub fn wpow(m: ArrayView3<Complex64>, n: usize) -> Result<Array3<Complex64>, ClError> {
    if n == 0 {
        let mut res = Array3::zeros(m.dim());
        for i in 0..m.dim().1 {
            res[(0, i, i)] = Complex64::one();
        }
        return Ok(res);
    }
    if n == 1 {
        return Ok(m.into_owned());
    }
    let x = wpow(m, n >> 1)?;
    wmul(wmul(x.view(), x.view())?.view(), wpow(m, n & 1)?.view())
}

/// Exponent of wfft representation of a multivector computed using Taylor series.
pub fn wexp(m: ArrayView3<Complex64>) -> Result<Array3<Complex64>, ClError> {
    if m.dim().1 != m.dim().2 {
        return Err(ClError::InvalidShape);
    }
    let mut res = Array3::zeros(m.dim());
    for i in 0..m.dim().1 {
        res[(0, i, i)] = Complex64::one();
    }

    let mut coeff = 1.0;
    let mut i = 1;

    // Ensure that all coefficients are below 1 so that the series doesn't blow up
    let mut max_c: f64 = 1.0;
    for (_, ci) in m.indexed_iter() {
        max_c = f64::max(ci.norm(), max_c);
    }
    max_c = max_c.ceil();
    let int_pow = max_c as usize;
    let normalized_m = m.map(|c| c / max_c);

    // Taylor series for exp(self / max_c)
    let mut p = normalized_m.clone();
    while coeff > f64::EPSILON {
        coeff = coeff / (i as f64);
        res = res + &p * coeff;
        i += 1;
        p = wmul(p.view(), normalized_m.view())?;
    }
    // exp(a) = pow(exp(a/b), b)
    wpow(res.view(), int_pow)
}

#[test]
fn odd_dim_test() {
    declare_algebra!(Cl3, [+,+,+], ["x", "y", "z"]);
    let one = Multivector::<f64, Cl3>::from_scalar(1.);
    let fone = one.fft().unwrap();
    assert_eq!(one, Cl3::ifft::<f64>(fone.view()).unwrap());

    declare_algebra!(Cl5, [+,+,+,+,+], ["x", "y", "z", "a", "b"]);
    let v = Multivector::<f64, Cl5>::from_vector(vec![1., 2., 3., 4., 5.].iter()).unwrap();
    let fv = v.fft().unwrap();
    assert_eq!(v, Cl5::ifft::<f64>(fv.view()).unwrap());

    declare_algebra!(Cl502, [+,+,+,+,+,0,0], ["x", "y", "z", "a", "b", "E", "G"]);
    type MV = Multivector<f64, Cl502>;
    for _ in 0..100 {
        let mv = MV::from_indexed_iter(Cl502::index_iter().map(|idx| (idx, rand::random::<f64>())))
            .unwrap();
        let m = mv.wfft().unwrap();
        let actual = Cl502::iwfft::<f64>(m.view()).unwrap();
        assert!(mv.approx_eq(&actual, 1e-15));
    }

    declare_algebra!(Cl303, [+,+,+,0,0,0], ["x", "y", "z", "E", "F", "G"]);
    type MV303 = Multivector<f64, Cl303>;
    for _ in 0..100 {
        let mv =
            MV303::from_indexed_iter(Cl303::index_iter().map(|idx| (idx, rand::random::<f64>())))
                .unwrap();
        let m = mv.wfft().unwrap();
        let actual = Cl303::iwfft::<f64>(m.view()).unwrap();
        assert!(mv.approx_eq(&actual, 1e-15));
    }

    // declare_algebra!(BadCl, [+,+,0,-], ["a","b","c","d"]);
    // assert!(Multivector::<f64, BadCl>::zero().wfft() == Err(ClError::FFTConditionsNotMet))
}

#[test]
fn wfft_test() {
    declare_algebra!(A, [+,+,+,+,0,0], ["w", "x", "y", "z", "e0", "e1"]);
    type MV = Multivector<f64, A>;

    // Check multiplication of basis blades
    for idx in A::index_iter() {
        let ei = MV::zero().set_by_mask(idx, 1.);
        let wfi = ei.wfft().unwrap();
        //println!("{ei}");
        assert_eq!(A::iwfft::<f64>(wfi.view()).unwrap(), ei.clone());

        for jdx in A::index_iter() {
            let ej = MV::zero().set_by_mask(jdx, 1.);
            let wfj = ej.wfft().unwrap();
            let wfij = wmul(wfi.view(), wfj.view()).unwrap();

            let actual = A::iwfft::<f64>(wfij.view()).unwrap();
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
        let wa = a.wfft().unwrap();
        let wb = b.wfft().unwrap();

        assert!(A::iwfft::<f64>(wmul(wa.view(), wb.view()).unwrap().view())
            .unwrap()
            .approx_eq(&expected, 1e-10));
    }

    // declare_algebra!(BadCl, [+,+,0,-,-], ["a","b","c","d","e"]);
    // assert!(Multivector::<f64, BadCl>::zero().wfft() == Err(ClError::FFTConditionsNotMet));

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
fn wfft_perf_test() {
    declare_algebra!(A, [+,+,+,+,0,0,0], ["w", "x", "y", "z", "e0", "e1", "e3"]);
    type MV = Multivector<f64, A>;
    let a = MV::from_indexed_iter(A::index_iter().map(|idx| (idx, rand::random::<f64>()))).unwrap();
    let b = MV::from_indexed_iter(A::index_iter().map(|idx| (idx, rand::random::<f64>()))).unwrap();

    let mut expected = MV::zero();
    let mut actual = MV::zero();

    let start = time::Instant::now();
    for _ in 0..1000 {
        let wa = a.wfft().unwrap();
        let wb = b.wfft().unwrap();
        let _ = black_box(A::iwfft::<f64>(wmul(wa.view(), wb.view()).unwrap().view()).unwrap());
    }
    println!("WFFT(full)     {:?}", start.elapsed());

    let start = time::Instant::now();
    let wa = a.wfft().unwrap();
    let wb = b.wfft().unwrap();
    for _ in 0..1000 {
        actual = black_box(A::iwfft::<f64>(wmul(wa.view(), wb.view()).unwrap().view()).unwrap());
    }
    println!("WFFT(iwfftmul) {:?}", start.elapsed());

    let start = time::Instant::now();
    let wa = a.wfft().unwrap();
    let wb = b.wfft().unwrap();
    for _ in 0..1000 {
        let _ = black_box(wmul(wa.view(), wb.view()).unwrap().view());
    }
    println!("WFFT(mul only) {:?}", start.elapsed());

    let start = time::Instant::now();
    for _ in 0..1000 {
        expected = black_box(a.naive_mul_impl(&b));
    }
    println!("Reference      {:?}", start.elapsed());
    assert!(actual.approx_eq(&expected, 1e-10));
}

#[test]
fn wpow_test() {
    declare_algebra!(A, [+,+,+,+,0,0,0], ["w", "x", "y", "z", "e0", "e1", "e3"]);
    type MV = Multivector<Complex64, A>;
    for _ in 0..100 {
        let a = MV::from_indexed_iter(A::index_iter().map(|idx| {
            (
                idx,
                Complex64::new(rand::random::<f64>(), rand::random::<f64>()),
            )
        }))
        .unwrap();

        let wp = wpow(a.wfft().unwrap().view(), 5).unwrap();
        assert!(A::iwfft::<Complex64>(wp.view()).unwrap().approx_eq(
            &a.naive_mul_impl(&a)
                .naive_mul_impl(&a)
                .naive_mul_impl(&a)
                .naive_mul_impl(&a),
            1e-10
        ));
    }
}

#[test]
fn wexp_test() {
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

        let wa = a.wfft().unwrap();
        let we = wexp(wa.view()).unwrap();

        assert!(A::iwfft::<Complex64>(we.view())
            .unwrap()
            .approx_eq(&a.exp(), 1.));
    }
}
