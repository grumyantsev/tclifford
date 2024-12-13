use std::fmt::{Debug, Display};

use mv_dense::Multivector;
use mv_sparse::SparseMultivector;
use num::complex::{Complex64, ComplexFloat};
use num::{pow, One, Zero};
use types::GeometricProduct;

use crate::algebra::TAlgebra;
use crate::coeff_storage::CoeffStorage;
use crate::types::{IndexType, Ring, Sign};
use std::marker::PhantomData;

pub mod algebra;
pub mod algebra_ifft;
pub mod clifft;
pub mod coeff_storage;
pub mod index_utils;
pub mod mv_dense;
pub mod mv_sparse;
pub mod ops;
pub mod types;

mod test;

#[derive(Debug, PartialEq, Eq)]
pub enum ClError {
    IndexOutOfBounds,
    FFTConditionsNotMet,
    InvalidShape,
}

#[derive(Debug)]
pub struct MultivectorBase<T: Ring, A: TAlgebra, Storage>
where
    Storage: CoeffStorage<T>,
{
    t: PhantomData<T>,
    a: PhantomData<A>,

    coeffs: Storage,
}

impl<T, A, Storage> MultivectorBase<T, A, Storage>
where
    T: Ring + Clone,
    A: TAlgebra,
    Storage: CoeffStorage<T>,
{
    pub fn from_scalar(scalar: T) -> Self {
        let mut ret = Self::default();
        ret.coeffs.set_by_mask(0, scalar);
        ret
    }

    pub fn from_vector<'a, It>(vec_coeffs: It) -> Result<Self, ()>
    where
        It: Iterator<Item = &'a T>,
        T: 'a,
    {
        let mut ret = Self::default();
        let mut n = 0;
        for c in vec_coeffs {
            if n >= A::dim() {
                return Err(()); // FIXME: actual error
            }
            ret.coeffs.set_by_mask(1 << n, c.clone());
            n += 1;
        }
        if n != A::dim() {
            return Err(());
        }
        Ok(ret)
    }

    pub fn from_indexed_iter_ref<'a, It>(iter: It) -> Result<Self, ClError>
    where
        It: Iterator<Item = (IndexType, &'a T)>,
        T: 'a,
    {
        let mut ret = Self::default();
        for (idx, c) in iter {
            if idx >= (1 << A::dim()) {
                return Err(ClError::IndexOutOfBounds);
            }
            ret.coeffs.set_by_mask(idx, c.clone());
        }
        Ok(ret)
    }

    pub fn from_indexed_iter<It>(iter: It) -> Result<Self, ClError>
    where
        It: Iterator<Item = (IndexType, T)>,
    {
        let mut ret = Self::default();
        for (idx, c) in iter {
            if idx >= (1 << A::dim()) {
                return Err(ClError::IndexOutOfBounds);
            }
            ret.coeffs.set_by_mask(idx, c);
        }
        Ok(ret)
    }

    pub fn coeff_enumerate(&self) -> impl Iterator<Item = (IndexType, &T)> {
        self.coeffs.coeff_enumerate()
    }

    /// Enumerate coefficients, but imaginary basis elements is treated as real ones multiplied by complex i
    pub fn complexified_coeff_enumerate<'a>(
        &'a self,
    ) -> impl Iterator<Item = (IndexType, Complex64)> + use<'a, T, A, Storage>
    where
        T: Into<Complex64>,
    {
        const I_POWERS: [Complex64; 4] = [
            Complex64 { re: 1., im: 0. },
            Complex64 { re: 0., im: -1. },
            Complex64 { re: -1., im: 0. },
            Complex64 { re: 0., im: 1. },
        ];

        self.coeff_enumerate().map(|(idx, c)| {
            (
                idx,
                I_POWERS[((idx & A::imag_mask()).count_ones() as usize) % 4] * c.clone().into(),
            )
        })
    }

    pub fn grade_enumerate(&self, grade: usize) -> impl Iterator<Item = (IndexType, &T)> {
        self.coeffs.grade_enumerate(grade)
    }

    pub fn grade_extract(&self, grade: usize) -> Self {
        let mut ret = Self::default();
        for (idx, c) in self.grade_enumerate(grade.into()) {
            ret.coeffs.set_by_mask(idx, c.clone());
        }
        ret
    }

    // This seems to be more harmful than useful.
    // pub fn into_algebra<OutA: TAlgebra>(self) -> MultivectorBase<T, OutA, Storage> {
    //     MultivectorBase::<T, OutA, Storage> {
    //         a: PhantomData,
    //         t: PhantomData,
    //         coeffs: self.coeffs,
    //     }
    // }

    pub fn to_storage_type<OutS>(&self) -> MultivectorBase<T, A, OutS>
    where
        OutS: CoeffStorage<T>,
    {
        let mut ret = MultivectorBase::<T, A, OutS>::default();
        for (idx, c) in self.coeff_enumerate() {
            if !c.is_zero() {
                ret.coeffs.set_by_mask(idx, c.clone());
            }
        }
        ret
    }

    pub fn add_impl(&self, rhs: &Self) -> Self {
        Self {
            t: PhantomData,
            a: PhantomData,
            coeffs: self.coeffs.add(&rhs.coeffs),
        }
    }

    pub fn sub_impl(&self, rhs: &Self) -> Self {
        Self {
            t: PhantomData,
            a: PhantomData,
            coeffs: self.coeffs.sub(&rhs.coeffs),
        }
    }

    pub fn neg_impl(&self) -> Self {
        Self::from_indexed_iter(self.coeff_enumerate().map(|(idx, c)| (idx, c.ref_neg()))).unwrap()
    }

    pub fn naive_wedge_impl(&self, rhs: &Self) -> Self {
        let mut ret = Self::default();
        for (self_idx, self_c) in self.coeff_enumerate() {
            for (rhs_idx, rhs_c) in rhs.coeff_enumerate() {
                let idx = self_idx ^ rhs_idx;
                let c = ret.coeffs.get_by_mask(idx);
                match A::blade_wedge_product_sign(self_idx, rhs_idx) {
                    Sign::Null => {}
                    Sign::Plus => ret.coeffs.set_by_mask(idx, c + self_c.ref_mul(rhs_c)),
                    Sign::Minus => ret.coeffs.set_by_mask(idx, c - self_c.ref_mul(rhs_c)),
                }
            }
        }
        ret
    }

    pub fn naive_mul_impl(&self, rhs: &Self) -> Self {
        let mut ret = Self::default();
        for (self_idx, self_c) in self.coeff_enumerate() {
            for (rhs_idx, rhs_c) in rhs.coeff_enumerate() {
                let idx = self_idx ^ rhs_idx;
                let c = ret.coeffs.get_by_mask(idx);
                match A::blade_geo_product_sign(self_idx, rhs_idx) {
                    Sign::Null => {}
                    Sign::Plus => ret.coeffs.set_by_mask(idx, c + self_c.ref_mul(rhs_c)),
                    Sign::Minus => ret.coeffs.set_by_mask(idx, c - self_c.ref_mul(rhs_c)),
                }
            }
        }
        ret
    }

    pub fn set_by_mask(mut self, idx: IndexType, value: T) -> Self {
        self.coeffs.set_by_mask(idx, value);
        self
    }

    pub fn get_by_mask(&self, idx: IndexType) -> T {
        self.coeffs.get_by_mask(idx)
    }

    pub fn approx_eq(&self, rhs: &Self, precision: f64) -> bool
    where
        T: Norm,
    {
        let diff = self - rhs;
        let ret = diff.coeff_enumerate().all(|(_, c)| c.norm() < precision);
        ret
    }

    ///Exponent of a multivector computed using Taylor series.
    pub fn exp(&self) -> Self
    where
        T: Norm + From<f64>,
        Self: GeometricProduct,
    {
        let mut res = Self::one();
        let mut coeff = 1.0;
        let mut i = 1;

        // Ensure that all coefficients are below 1 so that the series doesn't blow up
        let mut max_c: f64 = 1.0;
        for (_, ci) in self.coeff_enumerate() {
            max_c = f64::max(ci.norm(), max_c);
        }
        max_c = max_c.ceil();
        let int_pow = max_c as usize;
        let inv_max_c = T::from(1.0 / max_c);
        let normalized_mv = self * &inv_max_c;

        // Taylor series for exp(self / max_c)
        let mut p = normalized_mv.clone();
        while coeff > f64::EPSILON {
            coeff = coeff / (i as f64);
            res = res + &p * T::from(coeff);
            i += 1;
            p = &p * &normalized_mv;
        }
        // exp(a) = pow(exp(a/b), b)
        pow(res, int_pow)
    }

    /**
     * Reflection / "grade involution" of a Multivector through the origin.
     * Every basis vector `e` is turned into `-e`.
     */
    pub fn grade_involution(&self) -> Self {
        let mut ret = Self::zero();
        for (idx, coeff) in self.coeff_enumerate() {
            ret = ret.set_by_mask(idx, {
                match idx.count_ones() % 2 {
                    0 => coeff.clone(),
                    1 => coeff.ref_neg(),
                    _ => unreachable!(),
                }
            });
        }
        ret
    }

    /// Reversal operation. k-vectors e1^e2^...^ek are reversed into ek^...^e2^e1.
    pub fn reversal(&self) -> Self {
        let mut ret = Self::zero();
        for (idx, coeff) in self.coeff_enumerate() {
            ret = ret.set_by_mask(idx, {
                match idx.count_ones() % 4 {
                    0 | 1 => coeff.clone(),
                    2 | 3 => coeff.ref_neg(),
                    _ => unreachable!(),
                }
            });
        }
        ret
    }
}

impl<T, A, Storage> Default for MultivectorBase<T, A, Storage>
where
    T: Ring,
    A: TAlgebra,
    Storage: CoeffStorage<T>,
{
    fn default() -> Self {
        MultivectorBase::<T, A, Storage> {
            t: PhantomData,
            a: PhantomData,
            coeffs: Storage::new(A::dim()),
        }
    }
}

impl<T, A, Storage> PartialEq for MultivectorBase<T, A, Storage>
where
    T: Ring,
    A: TAlgebra,
    Storage: CoeffStorage<T>,
{
    fn eq(&self, other: &Self) -> bool {
        self.coeffs == other.coeffs
    }
}

impl<T, A, Storage> Display for MultivectorBase<T, A, Storage>
where
    T: Ring + Display,
    A: TAlgebra,
    Storage: CoeffStorage<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for (idx, coeff) in self.coeffs.coeff_enumerate() {
            if coeff.is_zero() {
                continue;
            }
            if !first {
                f.write_str(" + ")?;
            } else {
                first = false;
            }
            let coeff_str = format!("{}", coeff);
            if coeff_str.contains(['+', '-']) {
                f.write_fmt(format_args!("({:}) {}", coeff_str, A::blade_label(idx)))?;
            } else {
                f.write_fmt(format_args!("{:} {}", coeff_str, A::blade_label(idx)))?;
            }
        }
        if first {
            f.write_str("0")?;
        }
        Ok(())
    }
}

impl<T, A, Storage> Zero for MultivectorBase<T, A, Storage>
where
    T: Ring + Clone,
    A: TAlgebra,
    Storage: CoeffStorage<T>,
{
    fn zero() -> Self {
        Self::default()
    }

    fn is_zero(&self) -> bool {
        self.coeff_enumerate().all(|(_, c)| c.is_zero())
    }
}

impl<T, A, Storage> One for MultivectorBase<T, A, Storage>
where
    T: Ring + Clone,
    A: TAlgebra,
    Storage: CoeffStorage<T>,
    MultivectorBase<T, A, Storage>: GeometricProduct,
{
    fn one() -> Self {
        Self::from_scalar(T::one())
    }
}

pub trait Norm {
    fn norm(&self) -> f64;
}
impl<T> Norm for T
where
    T: ComplexFloat,
    T::Real: Into<f64>,
{
    fn norm(&self) -> f64 {
        self.abs().into()
    }
}

impl<T, A, S> Clone for MultivectorBase<T, A, S>
where
    T: Ring + Clone,
    A: TAlgebra,
    S: CoeffStorage<T>,
{
    fn clone(&self) -> Self {
        Self {
            t: PhantomData,
            a: PhantomData,
            coeffs: self.coeffs.clone(),
        }
    }
}
