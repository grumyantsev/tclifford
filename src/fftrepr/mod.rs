use crate::algebra::{ClAlgebraBase, ClBasis, NonDegenerate};
use crate::clifft::iclifft_into;
use crate::complexification::{Complexification, Even};
use crate::fftrepr::wmul::wmul;
use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

use ndarray::{s, Array2, Array3, ArrayView3, Axis};
use ndarray_linalg::Inverse;
use num::complex::Complex64;
use num::{Integer, One, Zero};
use std::iter::zip;

use crate::{clifft, ClError};
use crate::{
    types::{FromComplex, Ring},
    ClAlgebra, Multivector, Norm,
};

mod wmul;

/// `FFTRepr` is a representation of a multivector that provides the fastest multiplication for densely-packed multivectors.
///
/// For Cl(p,q,s) and m = p+q rounded up to an even number, every element can be viewed as
///
/// `A + n_{s_1} B + n_{s-2} C + n_{s_1} n_{s-2} D + ...`
/// where n_{i} are null basis elements, and A,B,C,D,... are m/2 by m/2 matrices produced by
/// the [fast matrix representation algorithm](https://arxiv.org/abs/2410.06103).
///
/// So, the representation is a 3-dimensional array of shape [2^s, m/2, m/2], where the first index is
/// a binary number with each bit corresponding to inclusion of `n_{i}` basis vector into the basis blade,
/// and the latter 2 indices are the spinor indices for the Cl(p,q) subalgebra representation.
///
/// It's important to note that this is NOT a Grassman algebra over matrices, due to `A,B,C,D,...`
/// being non-commutative with `n_0,...,n_{s-1}`.
#[derive(Debug)]
pub struct FFTRepr<A: ClAlgebra> {
    arr: Array3<Complex64>,
    a: PhantomData<A>,
}

impl<A> FFTRepr<A>
where
    A: ClAlgebra,
{
    pub(crate) fn from_array3_unchecked(arr: Array3<Complex64>) -> Self {
        Self {
            arr,
            a: PhantomData,
        }
    }

    /// Create FFTRepr form a 3-dimensional array
    ///
    /// It is assumed the array provided is a valid representation for this algebra.
    /// If it's not, then getting a multivector back from it results in an indefinite value.
    pub fn from_array3(arr: Array3<Complex64>) -> Result<Self, ClError> {
        let nonnull_dim = (A::real_mask() | A::imag_mask()).count_ones();
        let repr_nonnull_dim = ((nonnull_dim + 1) / 2) * 2;
        let martix_side = 1 << (repr_nonnull_dim / 2);
        let null_size = (A::proj_mask() >> nonnull_dim) + 1;

        if arr.dim() != (null_size, martix_side, martix_side) {
            return Err(ClError::InvalidShape);
        }

        Ok(Self::from_array3_unchecked(arr))
    }

    pub fn basis<const DIM: usize>() -> [Self; DIM]
    where
        A: ClBasis<DIM>,
    {
        A::basis::<Complex64>().map(|ei| ei.fft())
    }

    pub fn ifft<T>(&self) -> Multivector<T, A>
    where
        T: Ring + FromComplex + Clone,
    {
        // The signature order and array sizes are enforced by the declare_algebra macro

        let mut ret_c = Multivector::<Complex64, Even<Complexification<A>>>::zero();
        let repr_nonnull_dim =
            Even::<Complexification<A>>::real_dim() + Even::<Complexification<A>>::imag_dim();
        let step = 1 << repr_nonnull_dim;
        for (i, m) in self.arr.axis_iter(Axis(0)).enumerate() {
            iclifft_into(
                m,
                ret_c
                    .coeffs
                    .array_view_mut()
                    .slice_mut(s![i * step..(i + 1) * step]),
            )
            .unwrap();
        }

        ret_c.into_original().decomplexify()
    }

    pub fn exp(&self) -> FFTRepr<A> {
        //Self::from_array3(wexp(self.arr.view()).unwrap())
        let mut res = Self::one();

        let mut coeff = 1.0;
        let mut i = 1;

        // Ensure that all coefficients are below 1 so that the series doesn't blow up
        let mut max_c: f64 = 1.0;
        for (_, ci) in self.arr.indexed_iter() {
            max_c = f64::max(ci.norm(), max_c);
        }
        max_c = max_c.ceil();
        let int_pow = max_c as usize;
        let normalized_self = self / max_c;

        // Taylor series for exp(self / max_c)
        let mut p = normalized_self.clone();
        while coeff > f64::EPSILON {
            coeff = coeff / (i as f64);
            res = &res + &p * coeff;
            i += 1;
            p = p * &normalized_self;
        }
        // exp(a) = pow(exp(a/b), b)
        res.pow(int_pow)
    }

    pub fn pow(&self, n: usize) -> FFTRepr<A> {
        if n == 0 {
            return Self::one();
        }
        if n == 1 {
            return self.clone();
        }
        let x = self.pow(n >> 1);
        &x * &x * self.pow(n & 1)
    }

    pub fn into_array3(self) -> Array3<Complex64> {
        self.arr
    }

    pub fn from_array2(arr: Array2<Complex64>) -> Result<FFTRepr<A>, ClError>
    where
        A: NonDegenerate,
    {
        let (r, c) = arr.dim();
        let arr3 = arr
            .into_shape_clone([1, r, c])
            .or(Err(ClError::InvalidShape))?;
        Self::from_array3(arr3)
    }

    pub fn into_array2(self) -> Array2<Complex64>
    where
        A: NonDegenerate,
    {
        self.arr.index_axis(Axis(0), 0).into_owned()
    }

    pub fn inv(&self) -> Option<Self>
    where
        A: NonDegenerate,
    {
        let ret_arr = self.arr.index_axis(Axis(0), 0).inv().ok()?;
        let (m, n) = ret_arr.dim();
        Some(Self::from_array3_unchecked(
            ret_arr.into_shape_clone([1, m, n]).ok()?,
        ))
    }

    /// Reversal for the representation.
    pub fn rev(&self) -> Self {
        let mut ret = Self::zero();

        for (i, m) in self.arr.axis_iter(Axis(0)).enumerate() {
            let mut view = ret.arr.index_axis_mut(Axis(0), i);
            view.assign(&clifft::reversal(m).unwrap());
            match i.count_ones() % 4 {
                0 => {}
                1 => {
                    // alpha
                    view.indexed_iter_mut().for_each(|((j, k), p)| {
                        if (j ^ k).count_ones().is_odd() {
                            *p = -*p;
                        }
                    })
                }
                2 => view.map_inplace(|c| *c = -*c), // negate
                3 => {
                    // negate + alpha
                    view.indexed_iter_mut().for_each(|((j, k), p)| {
                        if (j ^ k).count_ones().is_even() {
                            *p = -*p;
                        }
                    })
                }
                _ => unreachable!(),
            };
        }
        ret
    }

    /// Parity flip for the representation.
    pub fn flip(&self) -> Self {
        let mut ret = self.clone();
        for ((idx, i, j), c) in ret.arr.indexed_iter_mut() {
            if (idx.count_ones() + (i ^ j).count_ones()) & 1 == 1 {
                *c = -*c;
            }
        }
        ret
    }

    /// Trace of the representation.
    pub fn trace(&self) -> Complex64 {
        let mut s = Complex64::zero();
        for i in 0..self.arr.dim().1 {
            s += self.arr[(0, i, i)]
        }
        s
    }

    /// Normalized trace. Equals to the scalar part of the represented multivector.
    pub fn ntrace(&self) -> Complex64 {
        self.trace() / (self.shape().1 as f64)
    }

    /// View of the representation as a 3-dimensional array.
    ///
    /// The axes are:
    /// * `Axis(0)` - index by the null basis vectors
    /// * `Axis(1)` and `Axis(2)` - the spinor indices of the fast matrix representation for each non-degenerate part.
    ///
    /// Mutable view is not provided because not all 3-arrays are valid representations of multivectors
    /// in case of algebras with odd number of non-null dimensions.
    /// Use [`FFTRepr::into_array3`] (or [`FFTRepr::into_array2`] for non-degenerate algebras) instead.
    pub fn view(&self) -> ArrayView3<Complex64> {
        self.arr.view()
    }

    pub fn approx_eq(&self, rhs: &Self, precision: f64) -> bool {
        zip(self.arr.iter(), rhs.arr.iter()).all(|(a, b)| (a - b).norm() < precision)
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        self.view().dim()
    }
}

#[opimps::impl_ops(Add)]
fn add<A>(self: FFTRepr<A>, rhs: FFTRepr<A>) -> FFTRepr<A>
where
    A: ClAlgebra,
{
    FFTRepr::<A>::from_array3_unchecked(&self.arr + &rhs.arr)
}

#[opimps::impl_ops(Sub)]
fn sub<A>(self: FFTRepr<A>, rhs: FFTRepr<A>) -> FFTRepr<A>
where
    A: ClAlgebra,
{
    FFTRepr::<A>::from_array3_unchecked(&self.arr - &rhs.arr)
}

#[opimps::impl_ops(Mul)]
fn mul<A>(self: FFTRepr<A>, rhs: FFTRepr<A>) -> FFTRepr<A>
where
    A: ClAlgebra,
{
    FFTRepr::<A>::from_array3_unchecked(wmul(self.arr.view(), rhs.arr.view()))
}

#[opimps::impl_ops(Mul)]
fn mul<A>(self: FFTRepr<A>, rhs: Complex64) -> FFTRepr<A>
where
    A: ClAlgebra,
{
    FFTRepr::<A>::from_array3_unchecked(self.arr.map(|c| c * rhs))
}

#[opimps::impl_ops(Mul)]
fn mul<A>(self: FFTRepr<A>, rhs: f64) -> FFTRepr<A>
where
    A: ClAlgebra,
{
    FFTRepr::<A>::from_array3_unchecked(self.arr.map(|c| c * rhs))
}

#[opimps::impl_ops(Div)]
fn div<A>(self: FFTRepr<A>, rhs: Complex64) -> FFTRepr<A>
where
    A: ClAlgebra,
{
    FFTRepr::<A>::from_array3_unchecked(self.arr.map(|c| c / rhs))
}

#[opimps::impl_ops(Div)]
fn div<A>(self: FFTRepr<A>, rhs: f64) -> FFTRepr<A>
where
    A: ClAlgebra,
{
    FFTRepr::<A>::from_array3_unchecked(self.arr.map(|c| c / rhs))
}

#[opimps::impl_uni_ops(Neg)]
fn neg<A>(self: FFTRepr<A>) -> FFTRepr<A>
where
    A: ClAlgebra,
{
    FFTRepr::<A>::from_array3_unchecked(-&self.arr)
}

impl<A> Zero for FFTRepr<A>
where
    A: ClAlgebra,
{
    fn zero() -> Self {
        let matrix_side = 1 << (((A::real_mask() | A::imag_mask()).count_ones() + 1) / 2) as usize;
        let wcount = 1 << A::proj_mask().count_ones();
        Self::from_array3_unchecked(Array3::zeros([wcount, matrix_side, matrix_side]))
    }

    fn is_zero(&self) -> bool {
        self.arr.iter().all(Complex64::is_zero)
    }
}

impl<A> One for FFTRepr<A>
where
    A: ClAlgebra,
{
    fn one() -> Self {
        let mut ret = Self::zero();
        for i in 0..ret.arr.dim().1 {
            ret.arr[(0, i, i)] = Complex64::one();
        }
        ret
    }
}

impl<A> Clone for FFTRepr<A>
where
    A: ClAlgebra,
{
    fn clone(&self) -> Self {
        Self {
            arr: self.arr.clone(),
            a: PhantomData {},
        }
    }
}

impl<A> PartialEq for FFTRepr<A>
where
    A: ClAlgebra,
{
    fn eq(&self, other: &Self) -> bool {
        self.arr == other.arr
    }
}

impl<A> Display for FFTRepr<A>
where
    A: ClAlgebra,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", self.arr))
    }
}
