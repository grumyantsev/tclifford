use crate::algebra::ClBasis;
use crate::coeff_storage::SparseStorage;
use crate::types::WedgeProduct;
use crate::ClAlgebra;
use crate::IndexType;
use crate::Multivector;
use crate::MultivectorBase;
use crate::Ring;

/// The multivector type where coefficients are stored in a HashMap.
///
/// Ideal for storing high-dimensional multivectors that only have a few non-zero coefficients.
/// For example, a blade in Cl(16) might be stored as `SparseMultivector`.
///
/// The [`Multivector`] type is faster than `SparseMultivector` at 10 dimensions and below.
pub type SparseMultivector<T, A> = MultivectorBase<T, A, SparseStorage<T>>;

impl<T, A> SparseMultivector<T, A>
where
    T: Ring + Clone,
    A: ClAlgebra,
{
    pub fn basis<const DIM: usize>() -> [Self; DIM]
    where
        A: ClBasis<DIM>,
    {
        return A::basis_sparse::<T>();
    }

    pub fn to_dense(&self) -> Multivector<T, A> {
        self.to_storage_type()
    }

    pub fn grades_extract(&self, grades: &[usize]) -> Self {
        Self::from_indexed_iter_ref(
            self.coeff_enumerate()
                .filter(|(idx, _)| grades.contains(&(idx.count_ones() as usize))),
        )
        .unwrap()
    }

    // Drop all the coefficients of the other grades,
    pub fn retain_grades(&mut self, grades: &[usize]) {
        self.filter_inplace(|idx, _| grades.contains(&(idx.count_ones() as usize)));
    }

    /// Drop all the coefficients for which the predicate is false.
    pub fn filter_inplace<F>(&mut self, f: F)
    where
        F: FnMut(&IndexType, &mut T) -> bool,
    {
        self.coeffs.coeffs.retain(f);
    }

    /// Consume self and return it without all the coefficients for which the predicate is false.
    pub fn filtered<F>(self, f: F) -> Self
    where
        F: FnMut(&IndexType, &mut T) -> bool,
    {
        let mut ret = self;
        ret.filter_inplace(f);
        ret
    }

    /// Amount of non-zero coefficients in the sparse multivector
    pub fn coeff_count(&self) -> usize {
        self.coeffs.coeffs.len()
    }
}

impl<T, A> WedgeProduct for SparseMultivector<T, A>
where
    T: Ring + Clone,
    A: ClAlgebra,
{
    fn wedge(&self, rhs: &Self) -> Self {
        self.naive_wedge_impl(rhs)
    }

    fn meet(&self, rhs: &Self) -> Self {
        self.naive_meet_impl(rhs)
    }
}
