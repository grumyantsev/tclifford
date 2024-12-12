use core::ops::{Add, Mul, Neg, Sub};
use num::{
    complex::{Complex32, Complex64},
    One, Zero,
};
use std::ops;

pub type IndexType = usize;

pub trait Ring
where
    Self: RefAdd
        + Add
        + RefSub
        + Sub<Output = Self>
        + RefNeg
        + Neg<Output = Self>
        + Zero
        + RefMul
        + Mul
        + One
        + PartialEq,
{
}
impl<T> Ring for T where
    T: RefAdd
        + Add
        + RefSub
        + Sub<Output = Self>
        + RefNeg
        + Neg<Output = Self>
        + Zero
        + RefMul
        + Mul
        + One
        + PartialEq
{
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Sign {
    Null = 0,
    Plus = 1,
    Minus = -1,
}

impl ops::Mul for Sign {
    type Output = Sign;

    fn mul(self, rhs: Sign) -> Self::Output {
        match self as i8 * rhs as i8 {
            -1 => Sign::Minus,
            1 => Sign::Plus,
            _ => Sign::Null,
        }
    }
}

/// Optimized wedge product implementation
pub trait WedgeProduct {
    fn wedge(&self, rhs: &Self) -> Self;
}

pub trait GeometricProduct {
    fn geo_mul(&self, rhs: &Self) -> Self;
}

pub trait FromComplex {
    fn from_complex(c: Complex64) -> Self;
}

impl FromComplex for Complex32 {
    fn from_complex(c: Complex64) -> Self {
        Complex32 {
            re: c.re as f32,
            im: c.im as f32,
        }
    }
}

impl FromComplex for Complex64 {
    fn from_complex(c: Complex64) -> Self {
        c
    }
}

impl FromComplex for f32 {
    fn from_complex(c: Complex64) -> Self {
        c.re as f32
    }
}

impl FromComplex for f64 {
    fn from_complex(c: Complex64) -> Self {
        c.re
    }
}

pub trait RefAdd {
    fn ref_add(&self, rhs: &Self) -> Self;
}

pub trait RefSub {
    fn ref_sub(&self, rhs: &Self) -> Self;
}

pub trait RefMul {
    fn ref_mul(&self, rhs: &Self) -> Self;
}

pub trait RefNeg {
    fn ref_neg(&self) -> Self;
}

impl<T> RefAdd for T
where
    for<'a> &'a T: Add<Output = T>,
{
    fn ref_add(&self, rhs: &Self) -> Self {
        self + rhs
    }
}

impl<T> RefSub for T
where
    for<'a> &'a T: Sub<Output = T>,
{
    fn ref_sub(&self, rhs: &Self) -> Self {
        self - rhs
    }
}

impl<T> RefMul for T
where
    for<'a> &'a T: Mul<Output = T>,
{
    fn ref_mul(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

impl<T> RefNeg for T
where
    for<'a> &'a T: Neg<Output = T>,
{
    fn ref_neg(&self) -> Self {
        self.neg()
    }
}
