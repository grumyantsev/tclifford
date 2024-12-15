use crate::algebra_ifft::InverseClifftRepr;
use crate::clifft::clifft;
use crate::clifft::clifft_into;
use crate::coeff_storage::ArrayStorage;
use crate::fftrepr::FFTRepr;
use crate::types::GeometricProduct;
use crate::types::WedgeProduct;
use crate::ClError;
use crate::MultivectorBase;
use crate::Ring;
use crate::SparseMultivector;
use crate::TAlgebra;
use ndarray::s;
use ndarray::Array1;
use ndarray::{Array2, Array3, ArrayView1, ArrayViewMut1, Axis};
use num::complex::Complex64;
use num::Zero;

#[cfg(test)]
use {crate::declare_algebra, std::hint::black_box, std::time};

/// The multivector type where coefficients are stored in a ndarray.
///
/// Should be used for "dense" multivectors that have non-zero values for most of coefficients of multiple grades.
/// For "sparse" multivectors with only a few non-zero coefficients or multivectors that only have values of a certain grade,
/// [`SparseMultivector`] should be used.
///
/// WARNING: Geometric multiplication of Multivectors is slow (on the order of N^2 where N is 2^(p+q+s) for Cl(p,q,s)). \
/// In order to have fast multiplication, especially for algebras of high dimensions, it's better to use the representation [`FFTRepr`].
///
/// For example,
/// ```
/// use num::One;
///
/// tclifford::declare_algebra!(Cl8, [+,+,+,+,+,+,+,+], ["e1","e2","e3","e4","e5","e6","e7","e8"]);
/// type MV = tclifford::Multivector::<f64, Cl8>;
///
/// let a = MV::one();
/// let b = MV::one();
///
/// let c: MV = &a * &b;                       // SLOW
/// let c: MV = (a.gfft() * b.gfft()).igfft(); // FAST
///
/// ```
pub type Multivector<T, A> = MultivectorBase<T, A, ArrayStorage<T>>;

impl<T, A> Multivector<T, A>
where
    T: Ring + Clone,
    A: TAlgebra,
{
    pub fn basis() -> Vec<Self> {
        return A::basis::<T>();
    }

    pub fn to_sparse(&self) -> SparseMultivector<T, A> {
        self.to_storage_type()
    }

    pub fn coeff_array_view(&self) -> ArrayView1<T> {
        self.coeffs.array_view()
    }

    /// Produce fast matrix representation of a multivector.
    ///
    /// For the inverse transform see [`InverseClifftRepr::ifft`]
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
        let mut complexified_coeffs = Array1::zeros(1 << A::dim());
        for (idx, c) in self.complexified_coeff_enumerate() {
            complexified_coeffs[idx] = c
        }

        clifft(complexified_coeffs.view()).or(Err(ClError::FFTConditionsNotMet))
    }

    pub fn gfft(&self) -> FFTRepr<A>
    where
        T: Into<Complex64>,
    {
        // Masks of the algebra are supposed to have this format:
        // proj_mask             == 0b1..10..0
        // real_mask | imag_mask == 0b0..01..1
        // This is enforced inside the declare_algebra! macro.

        let matrix_side = 1 << (((A::real_mask() | A::imag_mask()).count_ones() + 1) / 2) as usize;
        let wcount = 1 << A::proj_mask().count_ones();
        let mut ret = Array3::zeros([wcount, matrix_side, matrix_side]);

        let mut complexified_coeffs = Array1::zeros(1 << A::dim());
        for (idx, c) in self.complexified_coeff_enumerate() {
            complexified_coeffs[idx] = c
        }
        let step = (A::real_mask() | A::imag_mask()) + 1;
        let complexified_coeffs_view = complexified_coeffs.view();
        for i in 0..wcount {
            let v = complexified_coeffs_view.slice(s![i * step..(i + 1) * step]);
            clifft_into(v, ret.index_axis_mut(Axis(0), i)).unwrap();
        }
        FFTRepr::from_array3_unchecked(ret)
    }
}

impl<T, A> WedgeProduct for Multivector<T, A>
where
    T: Ring + Clone,
    A: TAlgebra,
{
    fn wedge(&self, rhs: &Self) -> Self {
        if A::dim() <= 4 {
            // benchmarking shows that at low dimensions this is faster
            return self.naive_wedge_impl(rhs);
        }
        // For >=5 dim use the asymptotically better algorithm
        let mut ret = Self::zero();
        let mut sc = self.clone();
        wedge_impl(
            sc.coeffs.array_view_mut(),
            rhs.coeff_array_view(),
            ret.coeffs.array_view_mut(),
        );
        ret
    }
}

// Ideally, there should be a default implementation, and specialized ones.
// But it's highly unstable...
// https://github.com/rust-lang/rust/issues/37653
impl<T, A> GeometricProduct for Multivector<T, A>
where
    T: Ring + Clone, // + FromComplex + Into<Complex64>,
    A: TAlgebra,     // + InverseClifftRepr,
{
    fn geo_mul(&self, rhs: &Self) -> Self {
        // optimization for Grassmann algebras
        if A::real_mask() == 0 && A::imag_mask() == 0 {
            return self.wedge(rhs);
        }
        //A::ifft(self.fft().unwrap().dot(&rhs.fft().unwrap()).view()).unwrap()
        self.naive_mul_impl(rhs)
    }
}

// This doesn't do any size checks. Sizes of arrays should be enforced on the Storage level.
fn wedge_impl<T>(a: ArrayViewMut1<T>, b: ArrayView1<T>, mut dest: ArrayViewMut1<T>)
where
    T: Ring + Clone,
{
    let size = a.len();
    if size == 1 {
        dest[0] = a[0].ref_mul(&b[0]);
        return;
    }

    // (a0 + e ^ a1) ^ (b0 + e ^ b1) =
    // = a0 ^ b0 + e ^ (a1 ^ b0 + ~a0 ^ b1)
    let (mut a0, a1) = a.split_at(Axis(0), size / 2);
    let (b0, b1) = b.split_at(Axis(0), size / 2);
    let (mut dest0, mut dest1) = dest.split_at(Axis(0), size / 2);

    // Use dest0 as the temp storage to avoid memory allocation
    let mut tmp = dest0.view_mut();
    wedge_impl(a1, b0, dest1.view_mut());
    // In-place grade involution of the a0
    for (idx, c) in a0.indexed_iter_mut() {
        if (idx.count_ones() & 1) != 0 {
            *c = c.ref_neg();
        }
    }
    wedge_impl(a0.view_mut(), b1, tmp.view_mut());
    // fill the top and return a0 back to it's original state
    for (idx, c) in tmp.indexed_iter() {
        dest1[idx] = dest1[idx].ref_add(c);
        if (idx.count_ones() & 1) != 0 {
            a0[idx] = a0[idx].ref_neg();
        }
    }
    // fill the bottom
    wedge_impl(a0, b0, dest0);
}

#[test]
fn fast_wedge_test() {
    declare_algebra!(
        Gr6,
        [0, 0, 0, 0, 0, 0],
        ["e0", "e1", "e2", "e3", "e4", "e5"]
    );
    let e = Gr6::basis::<f64>();
    let mut a = &e[0] + &e[1] * &e[2] + &e[1] * &e[3];
    let b = &e[0] + &e[1] * &e[2] + &e[5] * &e[4];

    let mut w = Multivector::<f64, Gr6>::zero();

    let st = time::Instant::now();
    for _ in 0..100 {
        _ = black_box(wedge_impl(
            a.coeffs.array_view_mut(),
            b.coeff_array_view(),
            w.coeffs.array_view_mut(),
        ));
    }
    println!("w = {w} in {:?}", st.elapsed());

    let st = time::Instant::now();
    for _ in 0..100 {
        w = black_box(a.naive_wedge_impl(&b));
    }
    println!("w = {w} in {:?}", st.elapsed());

    // ----

    declare_algebra!(
        Cl6,
        [+, +, +, +, +, +],
        ["e0", "e1", "e2", "e3", "e4", "e5"]
    );

    let a = Multivector::<f64, Cl6>::from_indexed_iter(
        Cl6::index_iter().map(|idx| (idx, rand::random::<f64>())),
    )
    .unwrap();
    let b = Multivector::<f64, Cl6>::from_indexed_iter(
        Cl6::index_iter().map(|idx| (idx, rand::random::<f64>())),
    )
    .unwrap();
    let c = Multivector::<f64, Cl6>::from_indexed_iter(
        Cl6::index_iter().map(|idx| (idx, rand::random::<f64>())),
    )
    .unwrap();
    let mut m = Multivector::<f64, Cl6>::zero();
    let mut mf = Multivector::<f64, Cl6>::zero();

    let st = time::Instant::now();
    for _ in 0..100 {
        m = black_box(a.naive_mul_impl(&(&b + &c)));
    }
    println!("Ref {:?}", st.elapsed());

    let st = time::Instant::now();
    for _ in 0..100 {
        mf = black_box(
            Cl6::ifft::<f64>(
                a.fft()
                    .unwrap()
                    .dot(&(&b.fft().unwrap() + &c.fft().unwrap()))
                    .view(),
            )
            .unwrap(),
        );
    }
    println!("FFT {:?}", st.elapsed());
    assert!(m.approx_eq(&mf, 1e-10));

    // let a = Multivector::<i32, Gr6>::zero();
    // let b = Multivector::<i32, Gr6>::one(); // .......
}
