use crate::coeff_storage::SparseStorage;
use crate::types::GeometricProduct;
use crate::types::WedgeProduct;
use crate::Multivector;
use crate::MultivectorBase;
use crate::Ring;
use crate::TAlgebra;

/// The multivector type where coefficients are stored in a HashMap.
///
/// Ideal for storing high-dimensional multivectors that only have a few non-zero coefficients.
/// For example, a rotor in Cl(6) only has 16 coefficients out of total 64 of the algebra,
/// so it should be stored as `SparseMultivector`.
pub type SparseMultivector<T, A> = MultivectorBase<T, A, SparseStorage<T>>;

impl<T, A> SparseMultivector<T, A>
where
    T: Ring + Clone,
    A: TAlgebra,
{
    pub fn basis() -> Vec<Self> {
        return A::basis_sparse::<T>();
    }

    pub fn to_dense(&self) -> Multivector<T, A> {
        self.to_storage_type()
    }
}

impl<T, A> WedgeProduct for SparseMultivector<T, A>
where
    T: Ring + Clone,
    A: TAlgebra,
{
    fn wedge(&self, rhs: &Self) -> Self {
        self.naive_wedge_impl(rhs)
    }
}

impl<T, A> GeometricProduct for SparseMultivector<T, A>
where
    T: Ring + Clone,
    A: TAlgebra,
{
    fn geo_mul(&self, rhs: &Self) -> Self {
        self.naive_mul_impl(rhs)
    }
}
