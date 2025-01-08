use crate::algebra::DivisionAlgebra;
use crate::types::{DivRing, RefMul, Ring};
use crate::{ClAlgebra, CoeffStorage};
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
{
    self.naive_mul_impl(&rhs)
}

// FIXME: This approach would fail for Quaternion<Quaternion>
// Or complex Quaternions for that matter.
// In case of non-invertible complex quaternions it gives a bunch of NaNs though, so maybe that's fine.
#[opimps::impl_ops(Div)]
fn div<T, A, Storage>(
    self: MultivectorBase<T, A, Storage>,
    rhs: MultivectorBase<T, A, Storage>,
) -> MultivectorBase<T, A, Storage>
where
    T: DivRing + Clone,
    A: ClAlgebra + DivisionAlgebra,
    Storage: CoeffStorage<T>,
{
    let mut rhs_conj = rhs.rev().flip();
    let rhs_norm = rhs.ref_mul(&rhs_conj).get_by_mask(0);

    for (_, c) in rhs_conj.coeffs.coeff_enumerate_mut() {
        *c = c.ref_div(&rhs_norm);
    }

    self.naive_mul_impl(&rhs_conj)
}

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
    T: DivRing + Clone,
    A: ClAlgebra,
    Storage: CoeffStorage<T>,
{
    let mut ret = MultivectorBase::<T, A, Storage>::default();
    for (idx, c) in self.coeffs.coeff_enumerate() {
        ret = ret.set_by_mask(idx, c.ref_div(&rhs));
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
