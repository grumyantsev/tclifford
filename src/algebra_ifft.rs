use crate::clifft::iclifft;
use crate::types::FromComplex;
use crate::types::IndexType;
use crate::ClError;
use crate::Multivector;
use crate::Ring;
use crate::TAlgebra;
use ndarray::s;
use ndarray::ArrayView2;
use num::complex::Complex64;

pub trait InverseClifftRepr: TAlgebra {
    fn decomplexified_iter<'a, T, It>(iter: It) -> impl Iterator<Item = (IndexType, T)>
    where
        T: Ring + Clone + FromComplex + 'a,
        It: Iterator<Item = (IndexType, &'a Complex64)>;

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
    A: TAlgebra,
{
    fn decomplexified_iter<'a, T, It>(iter: It) -> impl Iterator<Item = (IndexType, T)>
    where
        T: Ring + Clone + FromComplex + 'a,
        It: Iterator<Item = (IndexType, &'a Complex64)>,
    {
        const I_REV_POWERS: [Complex64; 4] = [
            Complex64 { re: 1., im: 0. },
            Complex64 { re: 0., im: 1. },
            Complex64 { re: -1., im: 0. },
            Complex64 { re: 0., im: -1. },
        ];
        iter.map(|(idx, c)| {
            (
                idx,
                T::from_complex(
                    I_REV_POWERS[((idx & A::imag_mask()).count_ones() as usize) % 4] * c,
                ),
            )
        })
    }

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
