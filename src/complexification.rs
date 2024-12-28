use crate::algebra::ClAlgebraBase;
use crate::types::FromComplex;
use crate::Multivector;
use crate::{declare_algebra, ClAlgebra, CoeffStorage, IndexType, MultivectorBase, Ring};
use num::complex::Complex64;
use num::Zero;
use std::marker::PhantomData;

// FIXME: Either go with a full-blown EvenComplexification, or drop it completely.
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

pub trait DecomplexifiedIter: ClAlgebra {
    fn decomplexified_iter<'a, T, It>(iter: It) -> impl Iterator<Item = (IndexType, T)>
    where
        T: Ring + Clone + FromComplex + 'a,
        It: Iterator<Item = (IndexType, &'a Complex64)>;
}

impl<A> DecomplexifiedIter for A
where
    A: ClAlgebra,
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
}

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
