use std::fmt::{Debug, Display, LowerExp, UpperExp, Write};

use ndarray::Array1;
use num::complex::ComplexFloat;
use types::DivRing;

use crate::coeff_storage::CoeffStorage;
use crate::types::{IndexType, Ring, Sign};
use std::marker::PhantomData;

pub mod algebra;
pub mod clifft;
pub mod pga;
pub mod quaternion;
pub mod types;

mod coeff_storage;
mod complexification;
mod fftrepr;
mod index_utils;
mod mv_dense;
mod mv_sparse;
mod ops;
mod precomputed;

mod test;

// Exported types
pub use crate::fftrepr::FFTRepr;
pub use crate::mv_dense::Multivector;
pub use crate::mv_sparse::SparseMultivector;

// Exported traits
pub use crate::algebra::ClAlgebra;
pub use crate::algebra::ClBasis;

// Re-export num traits to avoid explicit dependency on num by the caller
pub use num::{One, Zero};

#[derive(Debug, PartialEq, Eq)]
pub enum ClError {
    IndexOutOfBounds,
    FFTConditionsNotMet,
    InvalidShape,
}

/// Base type for multivectors with a generic coefficient storage. See [`Multivector`], [`SparseMultivector`].
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

    pub fn from_vector(vec_coeffs: impl IntoIterator<Item = T>) -> Result<Self, ClError> {
        let mut ret = Self::default();
        let mut n = 0;
        for c in vec_coeffs.into_iter() {
            if n >= A::dim() {
                return Err(ClError::InvalidShape);
            }
            ret.coeffs.set_by_mask(1 << n, c);
            n += 1;
        }
        if n != A::dim() {
            return Err(ClError::InvalidShape);
        }
        Ok(ret)
    }

    pub fn from_vector_ref<'a>(vec_coeffs: impl Iterator<Item = &'a T>) -> Result<Self, ClError>
    where
        T: 'a,
    {
        Self::from_vector(vec_coeffs.cloned())
    }

    pub fn from_indexed_iter(iter: impl Iterator<Item = (IndexType, T)>) -> Result<Self, ClError> {
        let mut ret = Self::default();
        for (idx, c) in iter {
            if idx >= (1 << A::dim()) {
                return Err(ClError::IndexOutOfBounds);
            }
            ret.coeffs.set_by_mask(idx, c);
        }
        Ok(ret)
    }

    pub fn from_indexed_iter_ref<'a>(
        iter: impl Iterator<Item = (IndexType, &'a T)>,
    ) -> Result<Self, ClError>
    where
        T: 'a,
    {
        Self::from_indexed_iter(iter.map(|(idx, c)| (idx, c.clone())))
    }

    /// Iterator through of all _non-zero_ coefficients of the multivector.
    ///
    /// It may or may not skip the zero coefficients, depending on the storage type.
    /// For iterating over all possible indices use [`ClAlgebra::index_iter`].
    pub fn coeff_enumerate(&self) -> impl Iterator<Item = (IndexType, &T)> {
        self.coeffs.coeff_enumerate()
    }

    /// Iterator through of all _non-zero_ coefficients of the given grade.
    ///
    /// It may or may not skip the zero coefficients, depending on the storage type.
    /// For iterating over all possible indices of the grade use [`ClAlgebra::grade_index_iter`].
    pub fn grade_enumerate(&self, grade: usize) -> impl Iterator<Item = (IndexType, &T)> {
        self.coeffs.grade_enumerate(grade)
    }

    /// Take a part of the multivector of the given grade.
    pub fn grade_extract(&self, grade: usize) -> Self {
        self.grade_extract_as(grade)
    }

    // TODO: a different return type?
    pub fn extract_vector(&self) -> Array1<T> {
        let mut ret = Array1::zeros(A::dim());
        for i in 0..A::dim() {
            ret[i] = self.get_by_mask(1 << i);
        }
        ret
    }

    /// Take a part of the multivector of the given grade, converted to a given multivector type
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

    /// Convert from one coefficient storage type to another
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

    pub(crate) fn add_impl(&self, rhs: &Self) -> Self {
        Self {
            t: PhantomData,
            a: PhantomData,
            coeffs: self.coeffs.add(&rhs.coeffs),
        }
    }

    pub(crate) fn sub_impl(&self, rhs: &Self) -> Self {
        Self {
            t: PhantomData,
            a: PhantomData,
            coeffs: self.coeffs.sub(&rhs.coeffs),
        }
    }

    pub(crate) fn neg_impl(&self) -> Self {
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

    /// Regressive product. Equivalent to `self.dual().wedge(&rhs.dual).dual()`
    pub fn naive_meet_impl(&self, rhs: &Self) -> Self {
        let mut ret = Self::default();
        let mask = A::real_mask() | A::imag_mask() | A::proj_mask();
        for (self_idx, self_c) in self.coeff_enumerate() {
            for (rhs_idx, rhs_c) in rhs.coeff_enumerate() {
                let self_dual_idx = mask & !self_idx;
                let rhs_dual_idx = mask & !rhs_idx;

                let dual_idx = mask & !(self_idx ^ rhs_idx);

                let sign = A::blade_wedge_product_sign(self_dual_idx, rhs_dual_idx)
                    * A::blade_wedge_product_sign(self_dual_idx, self_idx)
                    * A::blade_wedge_product_sign(rhs_dual_idx, rhs_idx)
                    * A::blade_wedge_product_sign(dual_idx, self_idx ^ rhs_idx);

                let c = ret.coeffs.get_by_mask(dual_idx);
                match sign {
                    Sign::Null => {}
                    Sign::Plus => ret.coeffs.set_by_mask(dual_idx, c + self_c.ref_mul(&rhs_c)),
                    Sign::Minus => ret.coeffs.set_by_mask(dual_idx, c - self_c.ref_mul(&rhs_c)),
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

    /// Set a coefficient before the basis blade of a given index.
    pub fn set_by_mask(mut self, idx: IndexType, value: T) -> Self {
        self.coeffs.set_by_mask(idx, value);
        self
    }

    /// Get a coefficient before the basis blade of a given index.
    pub fn get_by_mask(&self, idx: IndexType) -> T {
        self.coeffs.get_by_mask(idx)
    }

    /// Scalar part of a multivector.
    pub fn scalar_part(&self) -> T {
        self.get_by_mask(0)
    }

    /// Approximate equality between two multivectors with a given precision.
    // TODO: implementations for the approx crate
    pub fn approx_eq(&self, rhs: &Self, precision: f64) -> bool
    where
        T: Norm,
    {
        let diff = self - rhs;
        let ret = diff.coeff_enumerate().all(|(_, c)| c.norm() < precision);
        ret
    }

    /// Integer power of a multivector.
    // num::pow does some unnecessary cloning, let's just implement our own
    pub fn pow(&self, n: usize) -> Self {
        if n == 0 {
            return Self::one();
        }
        if n == 1 {
            return self.clone();
        }
        let x = self.pow(n >> 1);
        if n & 1 == 0 {
            &x * &x
        } else {
            &x * &x * self
        }
    }

    ///Exponent of a multivector computed using Taylor series.
    pub fn exp(&self) -> Self
    where
        T: Norm + From<f64>,
    {
        let mut res = Self::one();
        let mut coeff = 1.0;
        let mut i = 1;

        // Ensure that the magnitude and all the coefficients are below 1
        // so that the series doesn't blow up
        let mut max_c: f64 = 1.0;
        for (_, ci) in self.coeff_enumerate() {
            max_c = f64::max(ci.norm(), max_c);
        }
        max_c = max_c.max(self.mag2().norm().sqrt());
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
        res.pow(int_pow)
    }

    /// Parity flip / "grade involution" of a Multivector.
    /// Every generator `e` is turned into `-e`.
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

    /// Reversal operation. k-blades e1^e2^...^ek are reversed into ek^...^e2^e1.
    pub fn rev(&self) -> Self {
        let mut ret = Self::zero();
        for (idx, coeff) in self.coeff_enumerate() {
            ret = ret.set_by_mask(idx, Self::rev_c(idx, coeff));
        }
        ret
    }

    /// Hodge dual of a multivector, signature-agnostic.
    ///
    /// For any k-blade, `b.dual() * b` is a pseudoscalar.
    pub fn dual(&self) -> Self {
        let mask = A::proj_mask() | A::imag_mask() | A::real_mask();
        let mut ret = Self::default();
        for (idx, c) in self.coeff_enumerate() {
            let dual_idx = !idx & mask;
            let val = match A::blade_wedge_product_sign(dual_idx, idx) {
                Sign::Null => unreachable!(),
                Sign::Plus => c.clone(),
                Sign::Minus => c.ref_neg(),
            };
            ret.coeffs.set_by_mask(dual_idx, val);
        }
        ret
    }

    /// Dot product treating the multivectors as elements of vector space R^(2^DIM) with "all +" signature.
    pub fn vsdot(&self, rhs: &Self) -> T {
        let mut ret = T::zero();
        for (idx, c) in self.coeff_enumerate() {
            ret = ret + c.ref_mul(&rhs.get_by_mask(idx));
        }
        ret
    }

    /// Dot product treating the multivectors as elements of vector space R^(2^DIM) with the signature based on the squares of the blades.
    ///
    /// Equivalent to `(&a * &b).scalar_part()`, but runs in linear time.
    pub fn vdot(&self, rhs: &Self) -> T {
        let mut ret = T::zero();
        for (idx, c) in self.coeff_enumerate() {
            match A::blade_geo_product_sign(idx, idx) {
                Sign::Null => {}
                Sign::Plus => {
                    ret = ret + c.ref_mul(&rhs.get_by_mask(idx));
                }
                Sign::Minus => {
                    ret = ret - c.ref_mul(&rhs.get_by_mask(idx));
                }
            };
        }
        ret
    }

    /// Squared magnitude of a multivector.
    /// Equivalent to `(a.rev() * &a).scalar_part()`, but faster.
    pub fn mag2(&self) -> T {
        // Optimized version of self.rev().vdot(&self)
        let mut ret = T::zero();
        for (idx, c) in self.coeff_enumerate() {
            match A::blade_geo_product_sign(idx, idx) {
                Sign::Null => {}
                Sign::Plus => {
                    ret = ret + Self::rev_c(idx, c).ref_mul(&self.get_by_mask(idx));
                }
                Sign::Minus => {
                    ret = ret - Self::rev_c(idx, c).ref_mul(&self.get_by_mask(idx));
                }
            };
        }
        ret
    }

    /// Squared magnitude of a complex multivector.
    pub fn cmag2(&self) -> T
    where
        T: ComplexFloat,
    {
        let conjugate = Self::from_indexed_iter(
            self.coeff_enumerate()
                .map(|(idx, c)| (idx, Self::rev_c(idx, &c.conj()))),
        )
        .unwrap();
        conjugate.vdot(&self)
    }

    /// Revert the multivector and invert its magnitude. Equivalent to `self.rev() / self.mag2()`.
    ///
    /// Magnitude of `s.revm() * s` is one.
    pub fn revm(&self) -> Self
    where
        T: DivRing,
    {
        self.rev() / self.mag2()
    }

    /// Revert and complex conjugate the coefficients.
    pub fn revc(&self) -> Self
    where
        T: ComplexFloat,
    {
        Self::from_indexed_iter(
            self.coeff_enumerate()
                .map(|(idx, c)| (idx, Self::rev_c(idx, &c.conj()))),
        )
        .unwrap()
    }

    /// Revert, complex conjugate, and invert the magnitude. Equivalent to `self.revc() / self.mag2()`.
    ///
    /// Magnitude of `s.revcm() * s` is one.
    pub fn revcm(&self) -> Self
    where
        T: DivRing + ComplexFloat,
    {
        self.revc() / self.cmag2()
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

macro_rules! impl_formatting {
    ($fmt_trait:ident, $sign_checker_fmt:literal) => {
        impl<T, A, Storage> $fmt_trait for MultivectorBase<T, A, Storage>
        where
            T: Ring + Clone + $fmt_trait,
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
                        let coeff_str = format!($sign_checker_fmt, displayed_coeff);
                        coeff_str.contains(['+', '-'])
                    };
                    if inner_sign {
                        f.write_char('(')?;
                        $fmt_trait::fmt(&displayed_coeff, f)?;
                        f.write_char(')')?;
                    } else {
                        $fmt_trait::fmt(&displayed_coeff, f)?;
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
                    $fmt_trait::fmt(&T::zero(), f)?;
                }
                Ok(())
            }
        }
    };
}

impl_formatting!(Display, "{}");
impl_formatting!(LowerExp, "{:e}");
impl_formatting!(UpperExp, "{:E}");

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
