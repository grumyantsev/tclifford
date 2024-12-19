use crate::algebra::DivisionAlgebra;
use crate::types::{DivRing, GeometricProduct, RefMul, Ring};
use crate::{ClAlgebra, CoeffStorage, Norm};
use std::fmt::Display;
use std::ops::{Add, Div, Mul, Neg, Not, Sub};

use crate::MultivectorBase;

#[opimps::impl_ops(Add)]
fn add<T, A, Storage>(
    self: MultivectorBase<T, A, Storage>,
    rhs: MultivectorBase<T, A, Storage>,
) -> MultivectorBase<T, A, Storage>
where
    T: Ring + Clone,
    A: ClAlgebra,
    Storage: CoeffStorage<T>,
{
    self.add_impl(&rhs)
}

#[opimps::impl_ops(Sub)]
fn sub<T, A, Storage>(
    self: MultivectorBase<T, A, Storage>,
    rhs: MultivectorBase<T, A, Storage>,
) -> MultivectorBase<T, A, Storage>
where
    T: Ring + Clone,
    A: ClAlgebra,
    Storage: CoeffStorage<T>,
{
    self.sub_impl(&rhs)
}

#[opimps::impl_ops(Mul)]
fn mul<T, A, Storage>(
    self: MultivectorBase<T, A, Storage>,
    rhs: MultivectorBase<T, A, Storage>,
) -> MultivectorBase<T, A, Storage>
where
    T: Ring + Clone,
    A: ClAlgebra,
    Storage: CoeffStorage<T>,
    MultivectorBase<T, A, Storage>: GeometricProduct,
{
    self.naive_mul_impl(&rhs)
}

#[opimps::impl_ops(Div)]
fn div<T, A, Storage>(
    self: MultivectorBase<T, A, Storage>,
    rhs: MultivectorBase<T, A, Storage>,
) -> MultivectorBase<T, A, Storage>
where
    T: DivRing + Clone + Display + Norm,
    A: ClAlgebra + DivisionAlgebra,
    Storage: CoeffStorage<T>,
    MultivectorBase<T, A, Storage>: GeometricProduct,
{
    let mut rhs_conj = rhs.rev().flip();
    let rhs_norm = rhs.ref_mul(&rhs_conj).get_by_mask(0);

    for (_, c) in rhs_conj.coeffs.coeff_enumerate_mut() {
        *c = c.ref_div(&rhs_norm);
    }

    self.naive_mul_impl(&rhs_conj)
}

// #[opimps::impl_ops(Mul)]
// fn mul<T, A>(self: Multivector<T, A>, rhs: Multivector<T, A>) -> Multivector<T, A>
// where
//     T: Ring + Clone,
//     A: TAlgebra,
//     Multivector<T, A>: FastMul,
// {
//     self.fast_mul(&rhs)
// }

#[opimps::impl_ops(Mul)]
fn mul<T, A, Storage>(
    self: MultivectorBase<T, A, Storage>,
    rhs: T,
) -> MultivectorBase<T, A, Storage>
where
    T: Ring + Clone,
    A: ClAlgebra,
    Storage: CoeffStorage<T>,
{
    let mut ret = MultivectorBase::<T, A, Storage>::default();
    for (idx, c) in self.coeffs.coeff_enumerate() {
        ret = ret.set_by_mask(idx, c.ref_mul(&rhs));
    }
    ret
}

#[opimps::impl_ops(Div)]
fn div<T, A, Storage>(
    self: MultivectorBase<T, A, Storage>,
    rhs: T,
) -> MultivectorBase<T, A, Storage>
where
    T: Ring + Clone,
    for<'a> &'a T: Div<Output = T>,
    A: ClAlgebra,
    Storage: CoeffStorage<T>,
{
    let mut ret = MultivectorBase::<T, A, Storage>::default();
    for (idx, c) in self.coeffs.coeff_enumerate() {
        ret = ret.set_by_mask(idx, c / &rhs);
    }
    ret
}

#[opimps::impl_uni_ops(Neg)]
fn neg<T, A, Storage>(self: MultivectorBase<T, A, Storage>) -> MultivectorBase<T, A, Storage>
where
    T: Ring + Clone,
    A: ClAlgebra,
    Storage: CoeffStorage<T>,
{
    self.neg_impl()
}

#[opimps::impl_uni_ops(Not)]
fn not<T, A, Storage>(self: MultivectorBase<T, A, Storage>) -> MultivectorBase<T, A, Storage>
where
    T: Ring + Clone,
    A: ClAlgebra,
    Storage: CoeffStorage<T>,
{
    self.rev()
}
