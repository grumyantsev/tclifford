use std::fmt::{Debug, Display, Write};

use num::complex::{Complex64, ComplexFloat};
use num::{pow, One, Zero};
use types::GeometricProduct;

use crate::coeff_storage::CoeffStorage;
use crate::types::{IndexType, Ring, Sign};
use std::marker::PhantomData;

pub mod algebra;
pub mod clifft;
pub mod quaternion;
pub mod types;

pub mod algebra_ifft;
mod coeff_storage;
mod fftrepr;
mod index_utils;
mod mv_dense;
mod mv_sparse;
mod ops;

mod test;

// Exported types
pub use crate::fftrepr::FFTRepr;
pub use crate::mv_dense::Multivector;
pub use crate::mv_sparse::SparseMultivector;

// Exported traits
pub use crate::algebra::ClAlgebra;

#[derive(Debug, PartialEq, Eq)]
pub enum ClError {
    IndexOutOfBounds,
    FFTConditionsNotMet,
    InvalidShape,
    NotARepresentation,
}

/// Base type for multivectors with a generic storage. See [`Multivector`], [`SparseMultivector`].
#[derive(Debug)]
pub struct MultivectorBase<T: Ring, A: ClAlgebra, Storage>
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
    A: ClAlgebra,
    Storage: CoeffStorage<T>,
{
    pub fn from_scalar(scalar: T) -> Self {
        let mut ret = Self::default();
        ret.coeffs.set_by_mask(0, scalar);
        ret
    }

    pub fn from_vector<'a, It>(vec_coeffs: It) -> Result<Self, ClError>
    where
        It: Iterator<Item = &'a T>,
        T: 'a,
    {
        let mut ret = Self::default();
        let mut n = 0;
        for c in vec_coeffs {
            if n >= A::dim() {
                return Err(ClError::InvalidShape);
            }
            ret.coeffs.set_by_mask(1 << n, c.clone());
            n += 1;
        }
        if n != A::dim() {
            return Err(ClError::InvalidShape);
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
        self.grade_extract_as(grade)
    }

    pub fn grade_extract_as<OutStorageType>(
        &self,
        grade: usize,
    ) -> MultivectorBase<T, A, OutStorageType>
    where
        OutStorageType: CoeffStorage<T>,
    {
        let mut ret = MultivectorBase::<T, A, OutStorageType>::default();
        for (idx, c) in self.grade_enumerate(grade.into()) {
            ret.coeffs.set_by_mask(idx, c.clone());
        }
        ret
    }

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

    /// Parity flip / "grade involution" of a Multivector.
    /// Every basis vector `e` is turned into `-e`.
    pub fn flip(&self) -> Self {
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

    #[inline(always)]
    fn rev_c(idx: IndexType, coeff: &T) -> T {
        match idx.count_ones() % 4 {
            0 | 1 => coeff.clone(),
            2 | 3 => coeff.ref_neg(),
            _ => unreachable!(),
        }
    }

    /// Reversal operation. k-vectors e1^e2^...^ek are reversed into ek^...^e2^e1.
    pub fn rev(&self) -> Self {
        let mut ret = Self::zero();
        for (idx, coeff) in self.coeff_enumerate() {
            ret = ret.set_by_mask(idx, Self::rev_c(idx, coeff));
        }
        ret
    }

    pub fn dual(&self) -> Self {
        // This might need some sign adjustments.
        //
        let mut ret = Self::default();
        for (idx, c) in self.coeff_enumerate() {
            let dual_idx = !idx & (A::proj_mask() | A::imag_mask() | A::real_mask());
            ret.coeffs.set_by_mask(dual_idx, c.clone());
        }
        ret
    }
}

impl<T, A, Storage> Default for MultivectorBase<T, A, Storage>
where
    T: Ring,
    A: ClAlgebra,
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
    A: ClAlgebra,
    Storage: CoeffStorage<T>,
{
    fn eq(&self, other: &Self) -> bool {
        self.coeffs == other.coeffs
    }
}

impl<T, A, Storage> Display for MultivectorBase<T, A, Storage>
where
    T: Ring + Clone + Display,
    A: ClAlgebra,
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
            let displayed_coeff = if f.alternate() {
                Self::rev_c(idx, coeff)
            } else {
                coeff.clone()
            };
            let inner_sign = f.sign_plus() || {
                // This formats any coeff twice,
                // but it seems that there is no easy way to pass the flags down the line
                let coeff_str = format!("{}", displayed_coeff);
                coeff_str.contains(['+', '-'])
            };
            if inner_sign {
                f.write_char('(')?;
                displayed_coeff.fmt(f)?;
                f.write_char(')')?;
            } else {
                displayed_coeff.fmt(f)?;
            }
            let lbl = if f.alternate() {
                A::blade_label_rev(idx)
            } else {
                A::blade_label(idx)
            };
            if lbl.len() > 0 {
                f.write_char(' ')?;
                f.write_str(lbl.as_str())?;
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
    A: ClAlgebra,
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
    A: ClAlgebra,
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
    A: ClAlgebra,
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
