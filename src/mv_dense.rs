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
use ndarray::ArrayViewMut1;
use ndarray::Axis;
use ndarray::Ix1;
use ndarray::Shape;
use ndarray::{Array2, ArrayView1};
use num::complex::Complex64;
use num::One;
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
    /// For the inverse transform see [`InverseClifftRepr::ifft`] and [`InverseClifftRepr::ifft_re`]
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
}

impl<T, A> WedgeProduct for Multivector<T, A>
where
    T: Ring + Clone,
    A: TAlgebra,
{
    fn wedge(&self, rhs: &Self) -> Self {
        let mut ret = Self::zero();
        wedge_impl(
            self.coeff_array_view(),
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

// It's actually 10 times worse than naive_mul_impl lol
fn mul_somewhat_better_impl<T, A>(a: ArrayView1<T>, b: ArrayView1<T>, mut dest: ArrayViewMut1<T>)
where
    T: Ring + Clone,
    A: TAlgebra,
{
    let size = a.len();
    if size == 1 {
        dest[0] = a[0].clone() * b[0].clone();
        return;
    }

    // (a0 + e ^ a1) * (b0 + e ^ b1) =
    // (a0 * b0 + e*e ^ ~a1 ^ b1) + e ^ (~a0 * b1 + a1 * b0)

    let (a_bottom, a_top) = a.split_at(Axis(0), size / 2);
    let (b_bottom, b_top) = b.split_at(Axis(0), size / 2);
    let (mut bottom, mut top) = dest.split_at(Axis(0), size / 2);

    let mut tmp = Array1::<T>::zeros(size / 2);
    mul_somewhat_better_impl::<T, A>(a_bottom, b_bottom, bottom.view_mut());
    if size & A::real_mask() != 0 {
        mul_somewhat_better_impl::<T, A>(alpha_1d(a_top).view(), b_top, tmp.view_mut());
        for (i, c) in tmp.indexed_iter() {
            bottom[i] = bottom[i].ref_add(c);
        }
    } else if size & A::imag_mask() != 0 {
        mul_somewhat_better_impl::<T, A>(alpha_1d(a_top).view(), b_top, tmp.view_mut());
        for (i, c) in tmp.indexed_iter() {
            bottom[i] = bottom[i].ref_add(c);
        }
    } else if size & A::proj_mask() != 0 {
        // don't do anything
    }

    mul_somewhat_better_impl::<T, A>(a_top, b_bottom, top.view_mut());
    mul_somewhat_better_impl::<T, A>(alpha_1d(a_bottom).view(), b_top, tmp.view_mut());
    for (i, c) in tmp.indexed_iter() {
        top[i] = top[i].ref_add(c);
    }
}

// This is actually slower ??????
// This doesn't do any size checks. Sizes of arrays should be enforced on the Storage level.
fn wedge_impl<T>(a: ArrayView1<T>, b: ArrayView1<T>, mut dest: ArrayViewMut1<T>)
where
    T: Ring + Clone,
{
    let size = a.len();
    if size == 1 {
        dest[0] = a[0].clone() * b[0].clone();
        return;
    }

    // (a0 + e ^ a1) ^ (b0 + e ^ b1) =
    // a0 ^ b0 + e ^ (a1 ^ b0 + ~a0 ^ b1)
    let (a_bottom, a_top) = a.split_at(Axis(0), size / 2);
    let (b_bottom, b_top) = b.split_at(Axis(0), size / 2);
    let (mut bottom, mut top) = dest.split_at(Axis(0), size / 2);

    // Use bottom as the temp storage
    let mut tmp = bottom.view_mut();
    wedge_impl(a_top, b_bottom, top.view_mut());
    wedge_impl(alpha_1d(a_bottom).view(), b_top, tmp.view_mut());
    // fill the top
    for (i, c) in tmp.indexed_iter() {
        top[i] = top[i].ref_add(c);
    }
    // fill the bottom
    wedge_impl(a_bottom, b_bottom, bottom);
}

fn alpha_1d<T>(mv: ArrayView1<T>) -> Array1<T>
where
    T: Ring + Clone,
{
    let mut ret = Array1::zeros(mv.len());
    for idx in 0..mv.len() {
        ret[idx] = if (idx.count_ones() & 1) == 1 {
            mv[idx].ref_neg()
        } else {
            mv[idx].clone()
        }
    }
    ret
}

#[test]
fn fast_wedge_test() {
    declare_algebra!(
        Gr6,
        [0, 0, 0, 0, 0, 0],
        ["e0", "e1", "e2", "e3", "e4", "e5"]
    );
    let e = Gr6::basis::<f64>();
    let a = &e[0] + &e[1] * &e[2] + &e[1] * &e[3];
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
            a.coeff_array_view(),
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
