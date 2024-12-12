use clifft::iclifft;
use ndarray::s;
use ndarray::ArrayView3;
use ndarray::ArrayViewMut3;
use std::hint::black_box;
use std::time;

use crate::coeff_storage::ArrayStorage;
use crate::declare_algebra;
use crate::types::FromComplex;
use crate::types::GeometricProduct;
use crate::types::WedgeProduct;
use crate::ClError;
use crate::InverseClifftRepr;
use crate::MultivectorBase;
use crate::Ring;
use crate::SparseMultivector;
use crate::TAlgebra;
use clifft::clifft;
use ndarray::Array1;
use ndarray::Array3;
use ndarray::ArrayViewMut1;
use ndarray::Axis;
use ndarray::{Array2, ArrayView1};
use num::complex::Complex64;
use num::Zero;

pub type Multivector<T, A> = MultivectorBase<T, A, ArrayStorage<T>>;

impl<T, A> Multivector<T, A>
where
    T: Ring + Clone,
    A: TAlgebra,
{
    pub fn to_sparse(&self) -> SparseMultivector<T, A> {
        self.to_storage_type()
    }

    pub fn coeff_array_view(&self) -> ArrayView1<T> {
        self.coeffs.array_view()
    }

    /// Produce fast matrix representation of a multivector.
    ///
    /// For the inverse transform see [`InverseClifftRepr::ifft`
    pub fn fft(&self) -> Result<Array2<Complex64>, ClError>
    where
        T: Into<Complex64>,
    {
        if A::proj_mask() != 0 {
            return Err(ClError::FFTConditionsNotMet);
        }
        if A::imag_mask() == 0 {
            return clifft(self.coeffs.array_view()).or(Err(ClError::FFTConditionsNotMet));
        }
        // FIXME: This comes with a completely unnecessary memory allocation
        // WARNING: This is NOT a valid multivector. Only contents of its storage are valid coefficients
        let tmp =
            Multivector::<Complex64, A>::from_indexed_iter(self.complexified_coeff_enumerate())
                .unwrap();
        clifft(tmp.coeffs.array_view()).or(Err(ClError::FFTConditionsNotMet))
    }

    pub fn wfft(&self) -> Result<Array3<Complex64>, ClError>
    where
        T: Into<Complex64>,
    {
        if !A::normalized_for_wfft() {
            return Err(ClError::FFTConditionsNotMet);
        }
        // proj_mask             == 0b1..10..0
        // real_mask | imag_mask == 0b0..01..1

        let matrix_side = 1 << (((A::real_mask() | A::imag_mask()).count_ones() + 1) / 2) as usize;
        let wcount = 1 << A::proj_mask().count_ones();
        let mut ret = Array3::zeros([wcount, matrix_side, matrix_side]);

        // FIXME: This comes with a completely unnecessary memory allocation
        // WARNING: This is NOT a valid multivector. Only contents of its storage are valid coefficients
        let tmp =
            Multivector::<Complex64, A>::from_indexed_iter(self.complexified_coeff_enumerate())
                .unwrap();
        let step = (A::real_mask() | A::imag_mask()) + 1;
        let coeffs = tmp.coeffs.array_view();
        for i in 0..wcount {
            let v = coeffs.slice(s![i * step..(i + 1) * step]);
            // TODO: clifft_into
            ret.slice_mut(s![i, .., ..]).assign(&clifft(v).unwrap());
        }
        Ok(ret)
    }
}

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
        let mut ret_arr = Array1::<Complex64>::zeros(1 << A::dim());
        let step = (A::real_mask() | A::imag_mask()) + 1;
        for (i, m) in wm.axis_iter(Axis(0)).enumerate() {
            // TODO: iclifft_into
            let chunk = iclifft(m).or(Err(ClError::FFTConditionsNotMet))?;
            ret_arr
                .view_mut()
                .slice_mut(s![i * step..(i + 1) * step])
                .assign(&chunk);
        }

        Multivector::<T, A>::from_indexed_iter(A::decomplexified_iter(ret_arr.indexed_iter()))
    }
}

impl<T, A> WedgeProduct for Multivector<T, A>
where
    T: Ring + Clone,
    A: TAlgebra,
{
    fn wedge(&self, rhs: &Self) -> Self {
        let mut ret = Self::zero();
        let mut sc = self.clone();
        wedge_impl(
            sc.coeffs.array_view_mut(),
            rhs.coeff_array_view(),
            ret.coeffs.array_view_mut(),
        );
        ret
    }
}

impl<T, A> GeometricProduct for Multivector<T, A>
where
    T: Ring + Clone, // + FromComplex + Into<Complex64>,
    A: TAlgebra,     // + InverseClifftRepr,
{
    fn geo_mul(&self, rhs: &Self) -> Self {
        // optimization for Grassmann algebras
        if A::real_mask() == 0 && A::imag_mask() == 0 {
            return self.wedge(rhs);
        }
        if A::proj_mask() != 0 {
            // This case is not optimized yet
            return self.naive_mul_impl(rhs);
        }
        //A::ifft(self.fft().unwrap().dot(&rhs.fft().unwrap()).view()).unwrap()
        self.naive_mul_impl(rhs)
    }
}

// This is actually slower ??????
// This doesn't do any size checks. Sizes of arrays should be enforced on the Storage level.
fn wedge_impl<T>(a: ArrayViewMut1<T>, b: ArrayView1<T>, mut dest: ArrayViewMut1<T>)
where
    T: Ring + Clone,
{
    let size = a.len();
    if size == 1 {
        dest[0] = a[0].ref_mul(&b[0]);
        return;
    }

    // (a0 + e ^ a1) ^ (b0 + e ^ b1) =
    // a0 ^ b0 + e ^ (a1 ^ b0 + ~a0 ^ b1)
    let (mut a_bottom, a_top) = a.split_at(Axis(0), size / 2);
    let (b_bottom, b_top) = b.split_at(Axis(0), size / 2);
    let (mut bottom, mut top) = dest.split_at(Axis(0), size / 2);

    // Use bottom as the temp storage
    let mut tmp = bottom.view_mut();
    wedge_impl(a_top, b_bottom, top.view_mut());
    for (idx, c) in a_bottom.indexed_iter_mut() {
        if (idx.count_ones() & 1) == 1 {
            *c = c.ref_neg();
        }
    }
    wedge_impl(a_bottom.view_mut(), b_top, tmp.view_mut());
    // fill the top and return a_bottom back to it's original state
    for (idx, c) in tmp.indexed_iter() {
        top[idx] = top[idx].ref_add(c);
        if (idx.count_ones() & 1) == 1 {
            a_bottom[idx] = a_bottom[idx].ref_neg();
        }
    }
    // fill the bottom
    wedge_impl(a_bottom, b_bottom, bottom);
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
    let size = a.shape()[0];
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
    let mut ret = Array3::zeros(a.dim());
    let mut ca = a.into_owned();
    wmul_impl(ca.view_mut(), b, ret.view_mut());
    Ok(ret)
}

#[test]
fn fast_wedge_test() {
    declare_algebra!(
        Gr6,
        [0, 0, 0, 0, 0, 0],
        ["e0", "e1", "e2", "e3", "e4", "e5"]
    );
    let e = Gr6::basis::<f64>();
    let mut a = &e[0] + &e[1] * &e[2] + &e[1] * &e[3];
    let b = &e[0] + &e[1] * &e[2] + &e[5] * &e[4];

    let mut w = Multivector::<f64, Gr6>::zero();
    // wedge_impl(
    //     a.coeff_array_view(),
    //     b.coeff_array_view(),
    //     w.coeffs.array_view_mut(),
    // );

    // println!("w = {w}");

    let st = time::Instant::now();
    for _ in 0..1000 {
        _ = black_box(wedge_impl(
            a.coeffs.array_view_mut(),
            b.coeff_array_view(),
            w.coeffs.array_view_mut(),
        ));
    }
    println!("w = {w} in {:?}", st.elapsed());

    let st = time::Instant::now();
    for _ in 0..1000 {
        w = black_box(a.naive_wedge_impl(&b));
    }
    println!("w = {w} in {:?}", st.elapsed());

    // declare_algebra!(
    //     Cl6,
    //     [+, +, +, +, +, +],
    //     ["e0", "e1", "e2", "e3", "e4", "e5"]
    // );
    // let mut m = Multivector::<f64, Cl6>::zero();

    // let st = time::Instant::now();
    // for _ in 0..100 {
    //     black_box(mul_somewhat_better_impl::<f64, Cl6>(
    //         a.coeff_array_view(),
    //         b.coeff_array_view(),
    //         m.coeffs.array_view_mut(),
    //     ));
    // }
    // println!("m = {m} in {:?}", st.elapsed());

    // let a = a.into_algebra::<Cl6>();
    // let b = b.into_algebra::<Cl6>();
    // let st = time::Instant::now();
    // for _ in 0..100 {
    //     m = black_box(a.naive_mul_impl(&b));
    // }
    // println!("m = {m} in {:?}", st.elapsed());

    // let st = time::Instant::now();
    // for _ in 0..100 {
    //     m = black_box(Cl6::ifft::<f64>(a.fft().unwrap().dot(&b.fft().unwrap()).view()).unwrap());
    // }
    // println!("m = {m} in {:?}", st.elapsed());

    // let a = Multivector::<i32, Gr6>::zero();
    // let b = Multivector::<i32, Gr6>::one(); // .......
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

    declare_algebra!(BadCl, [+,+,0,-,-], ["a","b","c","d","e"]);
    assert!(Multivector::<f64, BadCl>::zero().wfft() == Err(ClError::FFTConditionsNotMet))
}

#[test]
fn wfft_perf_test() {
    declare_algebra!(A, [+,+,+,+,0,0,0], ["w", "x", "y", "z", "e0", "e1", "e3"]);
    type MV = Multivector<f64, A>;
    let a = MV::from_indexed_iter(A::index_iter().map(|idx| (idx, rand::random::<f64>()))).unwrap();
    let b = MV::from_indexed_iter(A::index_iter().map(|idx| (idx, rand::random::<f64>()))).unwrap();

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
        let _ = black_box(A::iwfft::<f64>(wmul(wa.view(), wb.view()).unwrap().view()).unwrap());
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
        let _ = black_box(a.naive_mul_impl(&b));
    }
    println!("Reference      {:?}", start.elapsed());
}
