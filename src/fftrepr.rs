use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};

use ndarray::{Array3, Axis};
use num::complex::Complex64;
use num::{Integer, One, Zero};

use crate::algebra_ifft::wexp;
use crate::clifft;
use crate::{
    algebra::TAlgebra,
    algebra_ifft::{wmul, InverseWfftRepr},
    mv_dense::Multivector,
    types::{FromComplex, Ring},
};

#[derive(Debug, PartialEq)]
pub struct FFTRepr<A: TAlgebra + InverseWfftRepr> {
    arr: Array3<Complex64>,
    a: PhantomData<A>,
}

impl<A> FFTRepr<A>
where
    A: TAlgebra + InverseWfftRepr,
{
    pub(crate) fn from_array3(arr: Array3<Complex64>) -> Self {
        Self {
            arr,
            a: PhantomData,
        }
    }

    pub fn igfft<T>(&self) -> Multivector<T, A>
    where
        T: Ring + FromComplex + Clone,
    {
        // The signature order and array sizes are enforced by the declare_algebra macro
        A::iwfft(self.arr.view()).unwrap()
    }

    pub fn exp(&self) -> FFTRepr<A> {
        Self::from_array3(wexp(self.arr.view()).unwrap())
    }

    pub fn into_array(self) -> Array3<Complex64> {
        self.arr
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
}

#[opimps::impl_ops(Add)]
fn add<A>(self: FFTRepr<A>, rhs: FFTRepr<A>) -> FFTRepr<A>
where
    A: TAlgebra + InverseWfftRepr,
{
    FFTRepr::<A>::from_array3(&self.arr + &rhs.arr)
}

#[opimps::impl_ops(Sub)]
fn sub<A>(self: FFTRepr<A>, rhs: FFTRepr<A>) -> FFTRepr<A>
where
    A: TAlgebra + InverseWfftRepr,
{
    FFTRepr::<A>::from_array3(&self.arr - &rhs.arr)
}

#[opimps::impl_ops(Mul)]
fn mul<A>(self: FFTRepr<A>, rhs: FFTRepr<A>) -> FFTRepr<A>
where
    A: TAlgebra + InverseWfftRepr,
{
    FFTRepr::<A>::from_array3(wmul(self.arr.view(), rhs.arr.view()).unwrap())
}

impl<A> Zero for FFTRepr<A>
where
    A: TAlgebra + InverseWfftRepr,
{
    fn zero() -> Self {
        let matrix_side = 1 << (((A::real_mask() | A::imag_mask()).count_ones() + 1) / 2) as usize;
        let wcount = 1 << A::proj_mask().count_ones();
        Self::from_array3(Array3::zeros([wcount, matrix_side, matrix_side]))
    }

    fn is_zero(&self) -> bool {
        self.arr.iter().all(Complex64::is_zero)
    }
}

impl<A> One for FFTRepr<A>
where
    A: TAlgebra + InverseWfftRepr,
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
    A: TAlgebra + InverseWfftRepr,
{
    fn clone(&self) -> Self {
        Self {
            arr: self.arr.clone(),
            a: PhantomData {},
        }
    }
}
