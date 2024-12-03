use core::ops::{Add, Mul, Neg, Sub};
use num::{One, Zero};
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

/// Assign this trait to Multivector types that don't have fast wedge implementation.
/// The default implementation will be used
pub trait UseNaiveWedgeImpl {}

/// Alternative to [UseNaiveWedgeImpl]
/// Implement this trait for Multivector types that can utilize some trick to wedge faster
pub trait FastWedge {
    fn fast_wedge(&self, rhs: &Self) -> Self;
}

/// Assign this trait to Multivector types that don't have fast multiplication implementation
/// The default implementation will be used
pub trait UseNaiveMulImpl {}

/// Alternative to [UseNaiveWedgeImpl]
/// Implement this trait for Multivector types that can utilize some trick to multiply faster
pub trait FastMul {
    fn fast_mul(&self, rhs: &Self) -> Self;
}
