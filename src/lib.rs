use std::fmt::{Debug, Display};
use std::hint::black_box;
use std::time::{self, Duration};

use clifft::{clifft, iclifft};
use ndarray::{Array2, ArrayView1, ArrayView2};
use num::complex::Complex64;
use num::{One, Zero};
//use types::{FastMul, UseNaiveMulImpl, UseNaiveWedgeImpl};

use crate::algebra::TAlgebra;
use crate::coeff_storage::{ArrayStorage, CoeffStorage, SparseStorage};
use crate::types::{IndexType, Ring, Sign};
use std::marker::PhantomData;

pub mod algebra;
pub mod coeff_storage;
pub mod index_utils;
pub mod ops;
pub mod types;

#[derive(Debug, PartialEq, Eq)]
pub enum ClError {
    IndexOutOfBounds,
    FFTConditionsNotMet,
}

#[derive(Debug)]
pub struct MultivectorBase<T: Ring, A: TAlgebra, Storage>
where
    Storage: CoeffStorage<T>,
{
    t: PhantomData<T>,
    a: PhantomData<A>,

    coeffs: Storage,
}

pub type Multivector<T, A> = MultivectorBase<T, A, ArrayStorage<T>>;
pub type SparseMultivector<T, A> = MultivectorBase<T, A, SparseStorage<T>>;

impl<T, A, Storage> MultivectorBase<T, A, Storage>
where
    T: Ring + Clone,
    A: TAlgebra,
    Storage: CoeffStorage<T>,
{
    pub fn from_scalar(scalar: T) -> Self {
        let mut ret = Self::default();
        ret.coeffs.set_by_mask(0, scalar);
        ret
    }

    pub fn from_vector<'a, It>(vec_coeffs: It) -> Result<Self, ()>
    where
        It: Iterator<Item = &'a T>,
        T: 'a,
    {
        let mut ret = Self::default();
        let mut n = 0;
        for c in vec_coeffs {
            if n >= A::dim() {
                return Err(()); // FIXME: actual error
            }
            ret.coeffs.set_by_mask(1 << n, c.clone());
            n += 1;
        }
        if n != A::dim() {
            return Err(());
        }
        Ok(ret)
    }

    pub fn from_indexed_iter_ref<'a, It>(iter: It) -> Result<Self, ClError>
    where
        It: Iterator<Item = (IndexType, &'a T)>,
        T: 'a,
    {
        let mut ret = Self::default();
        for (idx, c) in iter {
            if idx >= (1 << A::dim()) {
                return Err(ClError::IndexOutOfBounds);
            }
            ret.coeffs.set_by_mask(idx, c.clone());
        }
        Ok(ret)
    }

    pub fn from_indexed_iter<It>(iter: It) -> Result<Self, ClError>
    where
        It: Iterator<Item = (IndexType, T)>,
    {
        let mut ret = Self::default();
        for (idx, c) in iter {
            if idx >= (1 << A::dim()) {
                return Err(ClError::IndexOutOfBounds);
            }
            ret.coeffs.set_by_mask(idx, c);
        }
        Ok(ret)
    }

    pub fn coeff_enumerate(&self) -> impl Iterator<Item = (IndexType, &T)> {
        self.coeffs.coeff_enumerate()
    }

    /// Enumerate coefficients, but imaginary basis elements is treated as real ones multiplied by complex i
    pub fn complexified_coeff_enumerate<'a>(
        &'a self,
    ) -> impl Iterator<Item = (IndexType, Complex64)> + use<'a, T, A, Storage>
    where
        T: Into<Complex64>,
    {
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

    pub fn grade_enumerate(&self, grade: usize) -> impl Iterator<Item = (IndexType, &T)> {
        self.coeffs.grade_enumerate(grade)
    }

    pub fn grade_extract(&self, grade: usize) -> Self {
        let mut ret = Self::default();
        for (idx, c) in self.grade_enumerate(grade.into()) {
            ret.coeffs.set_by_mask(idx, c.clone());
        }
        ret
    }

    pub fn into_algebra<OutA: TAlgebra>(self) -> MultivectorBase<T, OutA, Storage> {
        MultivectorBase::<T, OutA, Storage> {
            a: PhantomData,
            t: PhantomData,
            coeffs: self.coeffs,
        }
    }

    pub fn to_storage_type<OutS>(&self) -> MultivectorBase<T, A, OutS>
    where
        OutS: CoeffStorage<T>,
    {
        let mut ret = MultivectorBase::<T, A, OutS>::default();
        for (idx, c) in self.coeff_enumerate() {
            if !c.is_zero() {
                ret.coeffs.set_by_mask(idx, c.clone());
            }
        }
        ret
    }

    pub fn add_impl(&self, rhs: &Self) -> Self {
        Self {
            t: PhantomData,
            a: PhantomData,
            coeffs: self.coeffs.add(&rhs.coeffs),
        }
    }

    pub fn sub_impl(&self, rhs: &Self) -> Self {
        Self {
            t: PhantomData,
            a: PhantomData,
            coeffs: self.coeffs.sub(&rhs.coeffs),
        }
    }

    pub fn naive_wedge_impl(&self, rhs: &Self) -> Self {
        let mut ret = Self::default();
        for (self_idx, self_c) in self.coeff_enumerate() {
            for (rhs_idx, rhs_c) in rhs.coeff_enumerate() {
                let idx = self_idx ^ rhs_idx;
                let c = ret.coeffs.get_by_mask(idx);
                match A::blade_wedge_product_sign(self_idx, rhs_idx) {
                    Sign::Null => {}
                    Sign::Plus => ret
                        .coeffs
                        .set_by_mask(idx, c + self_c.clone() * rhs_c.clone()),
                    Sign::Minus => ret
                        .coeffs
                        .set_by_mask(idx, c - self_c.clone() * rhs_c.clone()),
                }
            }
        }
        ret
    }

    pub fn naive_mul_impl(&self, rhs: &Self) -> Self {
        let mut ret = Self::default();
        for (self_idx, self_c) in self.coeff_enumerate() {
            for (rhs_idx, rhs_c) in rhs.coeff_enumerate() {
                let idx = self_idx ^ rhs_idx;
                let c = ret.coeffs.get_by_mask(idx);
                match A::blade_geo_product_sign(self_idx, rhs_idx) {
                    Sign::Null => {}
                    Sign::Plus => ret
                        .coeffs
                        .set_by_mask(idx, c + self_c.clone() * rhs_c.clone()),
                    Sign::Minus => ret
                        .coeffs
                        .set_by_mask(idx, c - self_c.clone() * rhs_c.clone()),
                }
            }
        }
        ret
    }

    pub fn set_by_mask(mut self, idx: IndexType, value: T) -> Self {
        self.coeffs.set_by_mask(idx, value);
        self
    }

    pub fn get_by_mask(&self, idx: IndexType) -> T {
        self.coeffs.get_by_mask(idx)
    }
}

impl<T, A, Storage> Default for MultivectorBase<T, A, Storage>
where
    T: Ring,
    A: TAlgebra,
    Storage: CoeffStorage<T>,
{
    fn default() -> Self {
        MultivectorBase::<T, A, Storage> {
            t: PhantomData,
            a: PhantomData,
            coeffs: Storage::new(A::dim()),
        }
    }
}

impl<T, A, Storage> PartialEq for MultivectorBase<T, A, Storage>
where
    T: Ring,
    A: TAlgebra,
    Storage: CoeffStorage<T>,
{
    fn eq(&self, other: &Self) -> bool {
        self.coeffs == other.coeffs
    }
}

impl<T, A, Storage> Display for MultivectorBase<T, A, Storage>
where
    T: Ring + Display,
    A: TAlgebra,
    Storage: CoeffStorage<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for (idx, coeff) in self.coeffs.coeff_enumerate() {
            if coeff.is_zero() {
                continue;
            }
            if !first {
                f.write_str(" + ")?;
            } else {
                first = false;
            }
            let coeff_str = format!("{}", coeff);
            if coeff_str.contains(['+', '-']) {
                f.write_fmt(format_args!("({:}) {}", coeff_str, A::blade_label(idx)))?;
            } else {
                f.write_fmt(format_args!("{:} {}", coeff_str, A::blade_label(idx)))?;
            }
        }
        if first {
            f.write_str("0")?;
        }
        Ok(())
    }
}

impl<T, A, Storage> Zero for MultivectorBase<T, A, Storage>
where
    T: Ring + Clone,
    A: TAlgebra,
    Storage: CoeffStorage<T>,
{
    fn zero() -> Self {
        Self::default()
    }

    fn is_zero(&self) -> bool {
        self.coeff_enumerate().all(|(_, c)| c.is_zero())
    }
}

impl<T, A, Storage> One for MultivectorBase<T, A, Storage>
where
    T: Ring + Clone,
    A: TAlgebra,
    Storage: CoeffStorage<T>,
{
    fn one() -> Self {
        Self::from_scalar(T::one())
    }
}

impl<T, A> Multivector<T, A>
where
    T: Ring + Clone,
    A: TAlgebra,
{
    pub fn to_sparse(&self) -> SparseMultivector<T, A> {
        self.to_storage_type()
    }

    pub fn coeff_array_view(&self) -> ArrayView1<T> {
        self.coeffs.array_view()
    }

    /// Produce fast matrix representation of a multivector.
    ///
    /// For the inverse transform see [`InverseClifftRepr::ifft`] and [`InverseClifftRepr::ifft_re`]
    pub fn fft(&self) -> Result<Array2<Complex64>, ClError>
    where
        T: Into<Complex64>,
    {
        if A::proj_mask() != 0 {
            return Err(ClError::FFTConditionsNotMet);
        }
        if A::imag_mask() == 0 {
            return clifft(self.coeffs.array_view()).or(Err(ClError::FFTConditionsNotMet));
        }
        // FIXME: This comes with a completely unnecessary memory allocation
        // WARNING: This is NOT a valid multivector. Only contents of its storage are valid coefficients
        let tmp =
            Multivector::<Complex64, A>::from_indexed_iter(self.complexified_coeff_enumerate())
                .unwrap();
        clifft(tmp.coeffs.array_view()).or(Err(ClError::FFTConditionsNotMet))
    }
}

impl<T, A> SparseMultivector<T, A>
where
    T: Ring + Clone,
    A: TAlgebra,
{
    pub fn to_dense(&self) -> Multivector<T, A> {
        self.to_storage_type()
    }
}

pub trait InverseClifftRepr: TAlgebra {
    fn decomplexified_iter<'a, T, It>(iter: It) -> impl Iterator<Item = (IndexType, T)>
    where
        T: Ring + Clone + From<Complex64> + 'a,
        It: Iterator<Item = (IndexType, &'a Complex64)>;

    fn decomplexified_iter_re<'a, T, It>(iter: It) -> impl Iterator<Item = (IndexType, T)>
    where
        T: Ring + Clone + From<f64> + 'a,
        It: Iterator<Item = (IndexType, &'a Complex64)>;

    /// Restore the multivector from its matrix representation ([`Multivector::fft`]).
    ///
    /// Applicable only when the multivector has complex coefficients.
    ///
    /// See also [`InverseClifftRepr::ifft_re`]
    fn ifft<T>(m: ArrayView2<Complex64>) -> Result<Multivector<T, Self>, ClError>
    where
        T: Ring + Clone + From<Complex64>,
        Self: Sized;

    /// Restore the multivector from its matrix representation ([`Multivector::fft`]).
    ///
    /// Applicable only when the multivector has real coefficients.
    ///
    /// See also [`InverseClifftRepr::ifft`]
    fn ifft_re<T>(m: ArrayView2<Complex64>) -> Result<Multivector<T, Self>, ClError>
    where
        T: Ring + Clone + From<f64>,
        Self: Sized;
}

impl<A> InverseClifftRepr for A
where
    A: TAlgebra,
{
    fn decomplexified_iter<'a, T, It>(iter: It) -> impl Iterator<Item = (IndexType, T)>
    where
        T: Ring + Clone + From<Complex64> + 'a,
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
                (I_REV_POWERS[((idx & A::imag_mask()).count_ones() as usize) % 4] * c.clone())
                    .into(),
            )
        })
    }

    fn decomplexified_iter_re<'a, T, It>(iter: It) -> impl Iterator<Item = (IndexType, T)>
    where
        T: Ring + Clone + From<f64> + 'a,
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
                (I_REV_POWERS[((idx & A::imag_mask()).count_ones() as usize) % 4] * c.clone())
                    .re
                    .into(),
            )
        })
    }

    fn ifft<T>(m: ArrayView2<Complex64>) -> Result<Multivector<T, Self>, ClError>
    where
        T: Ring + Clone + From<Complex64>,
        Self: Sized,
    {
        if A::proj_mask() != 0 {
            return Err(ClError::FFTConditionsNotMet);
        }

        let coeffs = iclifft(m).or(Err(ClError::FFTConditionsNotMet))?;
        let ret_coeff_iter = Self::decomplexified_iter(coeffs.indexed_iter());
        Multivector::<T, Self>::from_indexed_iter(ret_coeff_iter)
    }

    fn ifft_re<T>(m: ArrayView2<Complex64>) -> Result<Multivector<T, Self>, ClError>
    where
        T: Ring + Clone + From<f64>,
        Self: Sized,
    {
        if A::proj_mask() != 0 {
            return Err(ClError::FFTConditionsNotMet);
        }

        let coeffs = iclifft(m).or(Err(ClError::FFTConditionsNotMet))?;
        let ret_coeff_iter =
            Self::decomplexified_iter_re(coeffs.indexed_iter()).map(|(idx, c)| (idx, c));
        Multivector::<T, Self>::from_indexed_iter(ret_coeff_iter)
    }
}

// impl<T, A> UseNaiveWedgeImpl for Multivector<T, A>
// where
//     T: Ring + Clone,
//     A: TAlgebra,
// {
// }

// impl<T, A> UseNaiveWedgeImpl for SparseMultivector<T, A>
// where
//     T: Ring + Clone,
//     A: TAlgebra,
// {
// }

// impl<T, A> UseNaiveMulImpl for SparseMultivector<T, A>
// where
//     T: Ring + Clone,
//     A: TAlgebra,
// {
// }

// impl<T, A> FastMul for Multivector<T, A>
// where
//     T: Ring + Clone,
//     A: TAlgebra,
// {
//     fn fast_mul(&self, rhs: &Self) -> Self {
//         if A::proj_mask() != 0 {
//             return self.naive_mul_impl(rhs);
//         }
//         let fp = self.fft().unwrap().dot(&rhs.fft().unwrap());
//         let p = A::ifft::<Complex64>(fp).unwrap();
//     }
// }

#[test]
fn test1() {
    declare_algebra!(Cl3, [+,+,+], ["x", "y", "z"]);
    type MV = Multivector<f64, Cl3>;

    let mut m = MV::default();
    m = m.set_by_mask(1, 2.0).set_by_mask(3, 1.0);
    if let [x, y, z] = Cl3::basis().as_slice() {
        assert!(m == x * 2. + y * x);
        assert!(z * z == MV::from_scalar(1.))
    } else {
        assert!(false);
    }

    let v = MV::from_vector([1., 2., 3.].iter()).unwrap();
    println!("v = {v}");
    assert!(MV::from_vector([1., 2.].iter()).is_err());
    assert!(MV::from_vector([1., 2., 3., 4.].iter()).is_err());
}

#[test]
fn wedge_test() {
    declare_algebra!(Gr4, [0, 0, 0, 0], ["w", "x", "y", "z"]);

    let a = Multivector::<f64, Gr4>::from_vector([1., 2., 3., 4.].iter()).unwrap();
    let b = Multivector::<f64, Gr4>::from_vector([4., 3., 2., 1.].iter()).unwrap();
    let c = a.naive_wedge_impl(&b);
    println!("c = {c}");

    let expected_c = Multivector::<f64, Gr4>::from_indexed_iter(
        [
            (0b0011, 5.),
            (0b0101, 10.),
            (0b0110, 5.),
            (0b1001, 15.),
            (0b1010, 10.),
            (0b1100, 5.),
        ]
        .into_iter(),
    )
    .unwrap();

    assert_eq!(&expected_c, &c);
    assert_eq!(
        expected_c.to_sparse(),
        a.to_sparse().naive_wedge_impl(&b.to_sparse())
    );
    assert_eq!(&expected_c, &a.naive_mul_impl(&b));
}

#[test]
fn complexify_test() {
    // "Real" algebra
    declare_algebra!(Cl04, [-,-,-,-], ["g1", "g2", "g3", "g4"]);
    // And it's complexification
    declare_algebra!(CCl4, [+,+,+,+], ["e1", "e2", "e3", "e4"]);

    let a = Multivector::<f64, Cl04>::from_vector([1., 2., 3., 4.].iter()).unwrap();
    let ca = Multivector::<Complex64, CCl4>::from_indexed_iter(a.complexified_coeff_enumerate())
        .unwrap();

    println!("ca = {}", ca);

    let restored = Multivector::<f64, Cl04>::from_indexed_iter(Cl04::decomplexified_iter_re(
        ca.coeff_enumerate(),
    ))
    .unwrap();
    assert_eq!(restored, a);

    let fa = a.fft().unwrap();
    let fft_restored = Cl04::ifft_re(fa.view()).unwrap();
    println!("{} == {}", fft_restored, a);
    assert_eq!(fft_restored, a);
}

#[test]
fn basis_test() {
    declare_algebra!(Cl44, [+,+,+,+,-,-,-,-], ["e1", "e2", "e3", "e4", "g1", "g2", "g3", "g4"]);
    if let [e1, e2, e3, e4, g1, g2, g3, g4] = Cl44::basis::<f64>().as_slice() {
        println!("{}", e1 + e2 + e3 + e4 + g1 + g2 + g3 + g4)
    } else {
        assert!(false)
    }

    let b = Cl44::basis_sparse::<Complex64>();
    println!("{}", &b[1] * &b[2]);
    for i in 0..b.len() {
        for j in 0..i {
            assert_eq!(&b[i] * &b[j], -(&b[j] * &b[i]))
        }
        //println!("{}", b[i].to_dense().fft().unwrap());
    }

    declare_algebra!(Cl4, [+,+,+,+], ["a", "b", "c", "d"]);
    let e = Cl4::basis::<f64>();
    for ei in e {
        println!("{}", ei.fft().unwrap());
    }
}

#[test]
fn fft_test() {
    declare_algebra!(Oct, [-,-,-,-,-,-], ["e1", "e2", "e3", "e4", "e5", "e6"]);
    // Associative map of real octonions
    let e = Oct::basis::<f64>();
    for i in 0..e.len() {
        let fei = e[i].fft().unwrap();
        // Check the the square of fft square is negative identity
        assert_eq!(fei.dot(&fei), Array2::from_diag_elem(8, -Complex64::one()));
        for j in 0..i {
            let eij = fei.dot(&e[j].fft().unwrap());
            let eji = e[j].fft().unwrap().dot(&fei);
            // Check anticommutativity
            assert_eq!(eij, -&eji);

            let prod = Oct::ifft_re::<f64>(eij.view()).unwrap();
            // Check that naive and fft products agree
            assert_eq!(prod, e[i].naive_wedge_impl(&e[j]));
            // And that the fft product is correct at all
            assert!(prod.get_by_mask((1 << i) | (1 << j)).is_one());
            assert!(prod.set_by_mask((1 << i) | (1 << j), 0.).is_zero());
        }
    }
    // Associative map of complex octonions
    let e = Oct::basis::<Complex64>();
    for i in 0..e.len() {
        let fei = e[i].fft().unwrap();
        // Check the the square of fft square is negative identity
        assert_eq!(fei.dot(&fei), Array2::from_diag_elem(8, -Complex64::one()));
        for j in 0..i {
            let eij = fei.dot(&e[j].fft().unwrap());
            let eji = e[j].fft().unwrap().dot(&fei);
            // Check anticommutativity
            assert_eq!(eij, -&eji);

            let prod = Oct::ifft::<Complex64>(eij.view()).unwrap();
            // Check that naive and fft products agree
            assert_eq!(prod, e[i].naive_wedge_impl(&e[j]));
            // And that the fft product is correct at all
            assert!(prod.get_by_mask((1 << i) | (1 << j)).is_one());
            assert!(prod
                .set_by_mask((1 << i) | (1 << j), Complex64::zero())
                .is_zero());
        }
    }
}

#[test]
fn fft_bench() {
    declare_algebra!(Cl08, [-,-,-,-,-,-,-,-], ["e1","e2","e3","e4","e5","e6","e7","e8"]);

    let e = Cl08::basis::<f64>();

    let start_time = time::Instant::now();
    for _ in 0..100 {
        for i in 0..e.len() {
            for j in 0..e.len() {
                let _ = black_box(e[i].naive_mul_impl(&e[j]));
            }
        }
    }
    let duration = start_time.elapsed();
    println!("naive mul duration = {:?}", duration);

    let start_time = time::Instant::now();
    for _ in 0..100 {
        for i in 0..e.len() {
            for j in 0..e.len() {
                let ei = e[i].fft().unwrap();
                let ej = e[j].fft().unwrap();
                let _ = black_box(Cl08::ifft_re::<f64>(ei.dot(&ej).view()).unwrap());
            }
        }
    }
    let duration = start_time.elapsed();
    println!("fft mul duration (each) = {:?}", duration);

    let start_time = time::Instant::now();
    let fe: Vec<_> = e.iter().map(|ei| ei.fft().unwrap()).collect();
    for _ in 0..100 {
        for i in 0..e.len() {
            for j in 0..e.len() {
                let _ = black_box(Cl08::ifft_re::<f64>(fe[i].dot(&fe[j]).view()).unwrap());
            }
        }
    }
    let duration = start_time.elapsed();
    println!("fft mul duration (once) = {:?}", duration);

    let se = Cl08::basis_sparse::<f64>();
    let start_time = time::Instant::now();
    for _ in 0..100 {
        for i in 0..se.len() {
            for j in 0..se.len() {
                let _ = black_box(se[i].naive_mul_impl(&se[j]));
            }
        }
    }
    let duration = start_time.elapsed();
    println!("sparse mul duration = {:?}", duration);

    // check validity
    for i in 0..e.len() {
        for j in 0..e.len() {
            assert_eq!(
                e[i].naive_mul_impl(&e[j]),
                Cl08::ifft_re::<f64>(fe[i].dot(&fe[j]).view()).unwrap()
            );
        }
    }
}
