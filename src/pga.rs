use crate::algebra::{make_array, PGAlgebra};
use crate::coeff_storage::CoeffStorage;
use crate::{ClAlgebra, ClError, MultivectorBase, One};

pub trait PGAMV<const SPC_DIM: usize>: Sized {
    /// Multivector corresponding to a point.
    fn pga_point(coords: impl IntoIterator<Item = f64>) -> Result<Self, ClError>;

    /// Hyperplane defined by the equation `a0 x0 + a1 x1 + ... + a{n-1} x{n-1} + an = 0`.
    fn pga_hyperplane(equation_coeffs: impl IntoIterator<Item = f64>) -> Result<Self, ClError>;

    /// Create a translation motor from a direction vector.
    fn pga_translation(direction: impl IntoIterator<Item = f64>) -> Result<Self, ClError>;

    /// Get coordinates of the point described by the PGA multivector (if it corresponds to one).
    fn pga_extract_point(&self) -> [f64; SPC_DIM];
}

impl<A, S, const SPC_DIM: usize> PGAMV<SPC_DIM> for MultivectorBase<f64, A, S>
where
    A: ClAlgebra + PGAlgebra<SPC_DIM>,
    S: CoeffStorage<f64>,
{
    fn pga_point(coords: impl IntoIterator<Item = f64>) -> Result<Self, ClError> {
        let mut coords: Vec<f64> = coords.into_iter().collect();
        let mag = coords.iter().map(|c| c * c).sum::<f64>().sqrt();
        coords.push(f64::one());
        let p = Self::from_vector(coords)?.dual();
        Ok(p / mag)
    }

    fn pga_hyperplane(equation_coeffs: impl IntoIterator<Item = f64>) -> Result<Self, ClError> {
        let ec: Vec<f64> = equation_coeffs.into_iter().collect();
        let mag = ec.iter().map(|c| c * c).sum::<f64>().sqrt();
        let p = Self::from_vector(ec)?;
        Ok(p / mag)
    }

    fn pga_translation(direction: impl IntoIterator<Item = f64>) -> Result<Self, ClError> {
        let e_idx = A::proj_mask();
        let bivec = Self::from_indexed_iter(
            direction
                .into_iter()
                .enumerate()
                .map(|(n, c)| (e_idx | (1 << n), c / 2.)),
        )?;
        // A faster exp
        Ok(bivec.set_by_mask(0, 1.))
    }

    fn pga_extract_point(&self) -> [f64; SPC_DIM] {
        let sd = self.dual();
        let scale = sd.get_by_mask(A::proj_mask());
        make_array(|n| sd.get_by_mask(1 << n) / scale)
    }
}

#[cfg(test)]
mod test {
    use num::{One, Zero};

    use crate::declare_algebra;
    use crate::pga::PGAMV;
    use crate::types::WedgeProduct;
    use crate::Multivector;
    use std::f64::consts::PI;

    #[test]
    fn pga_test() {
        declare_algebra!(PGA3, [+,+,+,0], ["x", "y", "z", "e"]);
        type MV = Multivector<f64, PGA3>;
        let [x, _y, z, _e] = MV::basis();

        let pt = MV::pga_point([3., 3., 3.]).unwrap();
        let r8 = x.wedge(&z) * (PI / 8.).sin() + MV::from_scalar((PI / 8.).cos());

        let tr = MV::pga_translation([4., 0., 3.]).unwrap();
        // Rotor around the point [4,0,4]
        let rot8 = !&tr * r8 * &tr;
        assert!(
            rot8.pow(8).approx_eq(&(-MV::one()), 1e-12),
            "{}",
            rot8.pow(8)
        );
        // Check 90 degree rotations of the point around the center.
        let check1 = !rot8.pow(2) * &pt * rot8.pow(2);
        assert!(check1
            .pga_extract_point()
            .into_iter()
            .zip([4., 3., 2.])
            .all(|(actual, expected)| (expected - actual).abs() < 1e-12));

        let check2 = !rot8.pow(4) * &pt * rot8.pow(4);
        assert!(check2
            .pga_extract_point()
            .into_iter()
            .zip([5., 3., 3.])
            .all(|(actual, expected)| (expected - actual).abs() < 1e-12));
        let check3 = !rot8.pow(6) * &pt * rot8.pow(6);
        assert!(check3
            .pga_extract_point()
            .into_iter()
            .zip([4., 3., 4.])
            .all(|(actual, expected)| (expected - actual).abs() < 1e-12));
        let check4 = !rot8.pow(8) * &pt * rot8.pow(8);
        assert!(check4
            .pga_extract_point()
            .into_iter()
            .zip([3., 3., 3.])
            .all(|(actual, expected)| (expected - actual).abs() < 1e-12));

        let plane = MV::pga_hyperplane([1., 1., 1., -9.]).unwrap();
        assert!(plane.wedge(&pt).approx_eq(&MV::zero(), 1e-12)); // the poing lies on the plane
    }
}
