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

pub trait ComplexProbe {
    fn type_is_real() -> bool;
    fn type_is_complex() -> bool;
}

macro_rules! impl_complex_probe {
    ($type: ident, $is_real: ident, $is_complex: ident) => {
        impl ComplexProbe for $type {
            fn type_is_real() -> bool {
                $is_real
            }

            fn type_is_complex() -> bool {
                $is_complex
            }
        }
    };
}

impl_complex_probe!(Complex32, false, true);
impl_complex_probe!(Complex64, false, true);
impl_complex_probe!(f32, true, false);
impl_complex_probe!(f64, true, false);
