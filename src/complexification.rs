use crate::algebra::{ClAlgebraBase, NonDegenerate};
use crate::types::FromComplex;
use crate::{ClAlgebra, CoeffStorage, IndexType, MultivectorBase, Ring};
use num::complex::Complex64;
use num::Integer;
use std::marker::PhantomData;

/// Given a type for `Cl(p,q,s)` produces a type for algebra `Cl(p+q,0,s)`.
/// The generators of the new algebra are supposed to be the same for real axes
/// or multiplied by `i` for imaginary axes.
///
/// Provides type multivector complexification conversion for [`FFTRepr`]
pub struct Complexification<A: ClAlgebraBase> {
    a: PhantomData<A>,
}
impl<A: ClAlgebraBase> ClAlgebraBase for Complexification<A> {
    fn dim() -> usize {
        A::dim()
    }
    fn real_mask() -> IndexType {
        A::real_mask() | A::imag_mask()
    }
    fn imag_mask() -> IndexType {
        0
    }
    fn proj_mask() -> IndexType {
        A::proj_mask()
    }
    fn axis_name(n: usize) -> String {
        if (1 << n) & A::imag_mask() != 0 {
            format!("(i*{})", A::axis_name(n))
        } else {
            A::axis_name(n)
        }
    }
}
impl<A> NonDegenerate for Complexification<A> where A: NonDegenerate {}

/// Given a type for `Cl(p,q,s)` produces a type for algebra `Cl(P,q,s)`,
/// where `P == p` if `p+q` is even, or `P == p+1` if `p+q` is odd
///
/// Provides type checking and conversion for [`FFTRepr`]
pub struct Even<A: ClAlgebraBase> {
    a: PhantomData<A>,
}
impl<A: ClAlgebraBase> Even<A> {
    #[inline(always)]
    fn base_non_null_dim() -> usize {
        A::real_dim() + A::imag_dim()
    }
}

impl<A: ClAlgebraBase> ClAlgebraBase for Even<A> {
    fn dim() -> usize {
        if (A::real_dim() + A::imag_dim()).is_even() {
            A::dim()
        } else {
            A::dim() + 1
        }
    }
    fn real_mask() -> IndexType {
        if Self::base_non_null_dim().is_even() {
            A::real_mask()
        } else {
            A::real_mask() | (1 << Self::base_non_null_dim())
        }
    }
    fn imag_mask() -> IndexType {
        A::imag_mask()
    }
    fn proj_mask() -> IndexType {
        if Self::base_non_null_dim().is_even() {
            A::proj_mask()
        } else {
            A::proj_mask() << 1
        }
    }
    fn axis_name(n: usize) -> String {
        if Self::base_non_null_dim().is_even() {
            A::axis_name(n)
        } else {
            if n == Self::base_non_null_dim() {
                "?".to_string()
            } else if n > Self::base_non_null_dim() {
                A::axis_name(n - 1)
            } else {
                A::axis_name(n)
            }
        }
    }

    fn real_dim() -> usize {
        if (A::real_dim() + A::imag_dim()).is_even() {
            A::real_dim()
        } else {
            A::real_dim() + 1
        }
    }

    fn imag_dim() -> usize {
        A::imag_dim()
    }

    fn proj_dim() -> usize {
        A::proj_dim()
    }
}
impl<A> NonDegenerate for Even<A> where A: NonDegenerate {}

impl<T, A, Storage> MultivectorBase<T, A, Storage>
where
    T: Ring + Clone + Into<Complex64>,
    A: ClAlgebra,
    Storage: CoeffStorage<T>,
{
    /// Enumerate coefficients, but imaginary basis elements is treated as real ones multiplied by complex i
    pub(crate) fn complexified_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = (IndexType, Complex64)> + use<'a, T, A, Storage> {
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

    pub fn complexify<OutStorage>(
        &self,
    ) -> MultivectorBase<Complex64, Complexification<A>, OutStorage>
    where
        OutStorage: CoeffStorage<Complex64>,
    {
        MultivectorBase::<Complex64, Complexification<A>, OutStorage>::from_indexed_iter(
            self.complexified_iter(),
        )
        .unwrap()
    }
}

impl<A, Storage> MultivectorBase<Complex64, Complexification<A>, Storage>
where
    A: ClAlgebra,
    Storage: CoeffStorage<Complex64>,
{
    pub fn decomplexify<T, OutStorage>(&self) -> MultivectorBase<T, A, OutStorage>
    where
        T: Ring + Clone + FromComplex,
        OutStorage: CoeffStorage<T>,
    {
        const I_REV_POWERS: [Complex64; 4] = [
            Complex64 { re: 1., im: 0. },
            Complex64 { re: 0., im: 1. },
            Complex64 { re: -1., im: 0. },
            Complex64 { re: 0., im: -1. },
        ];
        MultivectorBase::<T, A, OutStorage>::from_indexed_iter(self.coeff_enumerate().map(
            |(idx, c)| {
                (
                    idx,
                    T::from_complex(
                        I_REV_POWERS[((idx & A::imag_mask()).count_ones() as usize) % 4] * c,
                    ),
                )
            },
        ))
        .unwrap()
    }
}

impl<T, A, Storage> MultivectorBase<T, Even<A>, Storage>
where
    T: Ring + Clone,
    A: ClAlgebra,
    Storage: CoeffStorage<T>,
{
    pub(crate) fn into_original(self) -> MultivectorBase<T, A, Storage> {
        if A::dim() == Even::<A>::dim() {
            MultivectorBase::<T, A, Storage> {
                t: PhantomData,
                a: PhantomData,
                coeffs: self.coeffs,
            }
        } else {
            let effective_proj_mask = A::proj_mask() << 1;
            let effective_full_mask = effective_proj_mask | A::real_mask() | A::imag_mask();

            MultivectorBase::<T, A, Storage>::from_indexed_iter(self.coeff_enumerate().filter_map(
                |(idx, c)| {
                    if (idx & !effective_full_mask) != 0 {
                        None
                    } else {
                        let effective_idx = ((idx & effective_proj_mask) >> 1)
                            | (idx & (A::real_mask() | A::imag_mask()));
                        Some((effective_idx, c.clone()))
                    }
                },
            ))
            .unwrap()
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Complexification, Even};
    use crate::algebra::ClAlgebraBase;
    use crate::{declare_algebra, ClAlgebra, Multivector};
    use num::Zero;

    #[test]
    fn dc_test() {
        declare_algebra!(A, [+,-,+,-,+,+,0,0,0]);
        type MV = Multivector<f64, A>;

        let e = MV::basis();
        let a: MV = e.iter().fold(MV::zero(), |acc, ei| &acc + ei);
        let b = MV::from_indexed_iter(A::grade_index_iter(2).map(|idx| (idx, 1.))).unwrap();

        let ca: Multivector<_, _> = a.complexify();
        let cb: Multivector<_, _> = b.complexify();
        println!("ca = {}", ca);
        println!("cb = {}", cb);

        let dca: MV = ca.decomplexify();
        let dcb: MV = cb.decomplexify();
        println!("dca = {}", dca);
        println!("dcb = {}", dcb);

        assert_eq!(a, dca);
        assert_eq!(b, dcb);
    }

    #[test]
    fn even_test() {
        declare_algebra!(A, [+,+,-,0,0], ["a", "b", "c", "p", "q"]);
        assert_eq!(Even::<A>::real_dim(), 3);
        assert_eq!(Even::<A>::imag_dim(), 1);
        assert_eq!(Even::<A>::proj_dim(), 2);
        assert_eq!(Even::<A>::real_mask(), 0b001011);
        assert_eq!(Even::<A>::imag_mask(), 0b000100);
        assert_eq!(Even::<A>::proj_mask(), 0b110000);
        assert_eq!(Even::<Complexification<A>>::real_dim(), 4);

        declare_algebra!(B, [+,+,-,-,0,0]);
        assert_eq!(Even::<B>::real_dim(), B::real_dim());
        assert_eq!(Even::<B>::imag_dim(), B::imag_dim());
        assert_eq!(Even::<B>::proj_dim(), B::proj_dim());
        assert_eq!(Even::<B>::real_mask(), B::real_mask());
        assert_eq!(Even::<B>::imag_mask(), B::imag_mask());
        assert_eq!(Even::<B>::proj_mask(), B::proj_mask());

        assert_eq!(Even::<A>::axis_name(0), "a");
        assert_eq!(Even::<A>::axis_name(1), "b");
        assert_eq!(Even::<A>::axis_name(2), "c");
        assert_eq!(Even::<A>::axis_name(3), "?");
        assert_eq!(Even::<A>::axis_name(4), "p");
        assert_eq!(Even::<A>::axis_name(5), "q");

        assert_eq!(Even::<Complexification<A>>::axis_name(2), "(i*c)");
    }
}
