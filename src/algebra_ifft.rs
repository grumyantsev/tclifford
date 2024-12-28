use crate::clifft::iclifft;
use crate::complexification::DecomplexifiedIter;
use crate::types::FromComplex;
use crate::ClAlgebra;
use crate::ClError;
use crate::Multivector;
use crate::Ring;
use ndarray::s;
use ndarray::ArrayView2;
use num::complex::Complex64;

pub trait InverseClifftRepr: ClAlgebra + DecomplexifiedIter {
    /// Restore the multivector from its matrix representation ([`Multivector::fft`]).
    ///
    /// Applicable only when the multivector has complex coefficients.
    fn ifft<T>(m: ArrayView2<Complex64>) -> Result<Multivector<T, Self>, ClError>
    where
        T: Ring + Clone + FromComplex,
        Self: Sized;
}

impl<A> InverseClifftRepr for A
where
    A: ClAlgebra + DecomplexifiedIter,
{
    fn ifft<T>(m: ArrayView2<Complex64>) -> Result<Multivector<T, Self>, ClError>
    where
        T: Ring + Clone + FromComplex,
        Self: Sized,
    {
        if A::proj_mask() != 0 {
            return Err(ClError::FFTConditionsNotMet);
        }

        let coeffs = iclifft(m)?;
        // This drops upper half for odd-dimensional algebras
        let cview = coeffs.slice(s![0..(1 << A::dim())]);
        let ret_coeff_iter = Self::decomplexified_iter(cview.indexed_iter());
        Multivector::<T, Self>::from_indexed_iter(ret_coeff_iter)
    }
}
