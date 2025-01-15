use crate::coeff_storage::CoeffStorage;
use crate::types::DivRing;
use crate::{declare_algebra, MultivectorBase};

// A separate type of storage is needed for implementation of the Copy trait
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QStorage<T: DivRing + Copy + Clone> {
    coeff: [T; 4],
}

impl<T> CoeffStorage<T> for QStorage<T>
where
    T: DivRing + Copy + Clone,
{
    fn new(dim: usize) -> Self {
        if dim != 2 {
            panic!("Quaternions must have dimension 2")
        }
        Self {
            coeff: [T::zero(), T::zero(), T::zero(), T::zero()],
        }
    }

    fn get_by_mask(&self, idx: crate::types::IndexType) -> T {
        if idx < 4 {
            self.coeff[idx]
        } else {
            T::zero()
        }
    }

    fn set_by_mask(&mut self, idx: crate::types::IndexType, value: T) {
        if idx < 4 {
            self.coeff[idx] = value
        }
    }

    fn coeff_enumerate<'a>(&'a self) -> impl Iterator<Item = (crate::types::IndexType, &'a T)>
    where
        T: 'a,
    {
        self.coeff.iter().enumerate()
    }

    fn coeff_enumerate_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (crate::types::IndexType, &'a mut T)>
    where
        T: 'a,
    {
        self.coeff.iter_mut().enumerate()
    }

    fn grade_enumerate<'a>(
        &'a self,
        grade: usize,
    ) -> impl Iterator<Item = (crate::types::IndexType, &'a T)>
    where
        T: 'a,
    {
        self.coeff
            .iter()
            .enumerate()
            .filter(move |(idx, _)| (idx.count_ones() as usize) == grade)
    }

    fn add(&self, rhs: &Self) -> Self {
        Self {
            coeff: [
                self.coeff[0] + rhs.coeff[0],
                self.coeff[1] + rhs.coeff[1],
                self.coeff[2] + rhs.coeff[2],
                self.coeff[3] + rhs.coeff[3],
            ],
        }
    }

    fn sub(&self, rhs: &Self) -> Self {
        Self {
            coeff: [
                self.coeff[0] - rhs.coeff[0],
                self.coeff[1] - rhs.coeff[1],
                self.coeff[2] - rhs.coeff[2],
                self.coeff[3] - rhs.coeff[3],
            ],
        }
    }
}

declare_algebra!(pub QuaternionAlgebra, [-,-], ["I", "J"]);

/// Quaternion type is a multivector in Cl(0,2) with Copy trait implemented.
/// With the Copy trait, `ndarray::linalg::LinalgScalar` is automatically implemented too.
pub type Quaternion<T> = MultivectorBase<T, QuaternionAlgebra, QStorage<T>>;
impl<T> Copy for Quaternion<T> where T: DivRing + Clone + Copy {}

impl<T> Quaternion<T>
where
    T: DivRing + Clone + Copy,
{
    pub fn i() -> Self {
        Self {
            a: std::marker::PhantomData,
            t: std::marker::PhantomData,
            coeffs: QStorage {
                coeff: [T::zero(), T::one(), T::zero(), T::zero()],
            },
        }
    }
    pub fn j() -> Self {
        Self {
            a: std::marker::PhantomData,
            t: std::marker::PhantomData,
            coeffs: QStorage {
                coeff: [T::zero(), T::zero(), T::one(), T::zero()],
            },
        }
    }
    pub fn k() -> Self {
        Self {
            a: std::marker::PhantomData,
            t: std::marker::PhantomData,
            coeffs: QStorage {
                coeff: [T::zero(), T::zero(), T::zero(), -T::one()],
            },
        }
    }
    /// Quaternionic conjugate
    pub fn conj(&self) -> Self {
        Self {
            a: std::marker::PhantomData,
            t: std::marker::PhantomData,
            coeffs: QStorage {
                coeff: [
                    self.coeffs.coeff[0],
                    -self.coeffs.coeff[1],
                    -self.coeffs.coeff[2],
                    -self.coeffs.coeff[3],
                ],
            },
        }
    }
}

// TODO: Into and From nalgebra::Quaternion for real-valued quaternions

#[cfg(test)]
mod test {
    use crate::coeff_storage::ArrayStorage;
    use crate::quaternion::Quaternion;
    use ndarray::arr2;
    use num::complex::Complex64;
    use num::{One, Zero};

    #[test]
    fn quat_test() {
        type Q = Quaternion<f64>;
        type QC = Quaternion<Complex64>;
        let a = Q::i() + Q::j() * 2. + Q::k() * 3. + Q::one();
        let b = QC::i() + QC::j() * Complex64::from(2.) + QC::k() * Complex64::from(3.) + QC::one();

        assert_eq!(a.conj(), a.rev().flip());
        assert_eq!(b.conj(), b.rev().flip());

        let p0 = arr2(&[[QC::zero(), QC::one()], [QC::one(), QC::zero()]]);
        let p1 = arr2(&[[QC::zero(), QC::k()], [-QC::k(), QC::zero()]]);
        let p2 = arr2(&[[-QC::k(), QC::zero()], [QC::zero(), QC::k()]]);

        assert_eq!(p0.dot(&p1), p2);
        assert_eq!(p1.dot(&p0), -p2);

        println!(
            "{}", // FIXME: FFT should be implemented not just for Multivector<_>...
            QC::j().to_storage_type::<ArrayStorage<Complex64>>().fft()
        );

        // Check division
        let x = Quaternion::<Quaternion<f64>>::one();
        assert_eq!(&x / &x, x);
        assert_eq!(&QC::i() / &QC::j(), -QC::k());

        // Non-invertible cases. Check that we don't panic
        // 1 / (1 + i I) in H x H
        let r = &x
            / (Quaternion::<Quaternion<f64>>::i() * Quaternion::<f64>::i()
                + Quaternion::<Quaternion<f64>>::one());
        println!("{}", r);
        assert!(r.get_by_mask(0) != r.get_by_mask(0)); // The result is a bunch of NaNs.

        // 1 / (1 + i I) in H x C
        let r = QC::one() / (QC::one() + QC::i() * Complex64::i());
        println!("{}", r);
        assert!(r.get_by_mask(0) != r.get_by_mask(0)); // The result is a bunch of NaNs.
    }
}
