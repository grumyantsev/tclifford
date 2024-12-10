use core::ops::{Add, Mul, Neg, Sub};
use num::{
    complex::{Complex32, Complex64},
    One, Zero,
};
use std::ops;

pub type IndexType = usize;

pub trait Ring
where
    Self: Add + Sub<Output = Self> + Neg<Output = Self> + Zero + Mul + One + PartialEq,
{
}
impl<T> Ring for T where
    T: Add + Sub<Output = Self> + Neg<Output = Self> + Zero + Mul + One + PartialEq
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
