use crate::coeff_storage::SparseStorage;
use crate::types::GeometricProduct;
use crate::types::WedgeProduct;
use crate::Multivector;
use crate::MultivectorBase;
use crate::Ring;
use crate::TAlgebra;

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
