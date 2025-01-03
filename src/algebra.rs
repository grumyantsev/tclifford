use crate::{
    index_utils,
    types::{IndexType, Ring, Sign},
    Multivector, SparseMultivector,
};
use itertools::Itertools;
use num::{Integer, Zero};

/// An algebra trait imlemented by the [`declare_algebra!`](crate::declare_algebra) macro.
///
/// This trait should not be implemented manually.
// We would want this to be a const trait,
// but const traits are not in the language yet.
// https://github.com/rust-lang/rust/issues/67792
pub trait ClAlgebraBase {
    fn dim() -> usize;
    fn real_dim() -> usize {
        Self::real_mask().count_ones() as usize
    }
    fn imag_dim() -> usize {
        Self::imag_mask().count_ones() as usize
    }
    fn proj_dim() -> usize {
        Self::proj_mask().count_ones() as usize
    }
    fn real_mask() -> IndexType;
    fn imag_mask() -> IndexType;
    fn proj_mask() -> IndexType;
    fn axis_name(n: usize) -> String;
}

/// The Clifford algebra trait. Automatically implemented for any algebra type defined by the [`declare_algebra!`](crate::declare_algebra) macro.
///
/// This trait should not be implemented manually.
pub trait ClAlgebra: ClAlgebraBase {
    fn blade_label(idx: IndexType) -> String;
    fn blade_label_rev(idx: IndexType) -> String;
    fn ac_product_sign(a: IndexType, b: IndexType) -> Sign;
    fn blade_wedge_product_sign(a: IndexType, b: IndexType) -> Sign;
    fn blade_geo_product_sign(a: IndexType, b: IndexType) -> Sign;
    fn index_iter() -> impl Iterator<Item = IndexType>;
    fn grade_index_iter(weight: usize) -> impl Iterator<Item = IndexType>;
}

fn make_array<T, const N: usize>(f: impl Fn(usize) -> T) -> [T; N] {
    let mut ret = std::mem::MaybeUninit::<[T; N]>::uninit();
    unsafe {
        for i in 0..N {
            ret.as_mut_ptr().cast::<T>().wrapping_add(i).write(f(i));
        }
        ret.assume_init()
    }
}

pub trait ClBasis<const DIM: usize>: ClAlgebra {
    fn basis<T>() -> [Multivector<T, Self>; DIM]
    where
        T: Ring + Clone,
        Self: Sized,
    {
        make_array(|n| {
            Multivector::<T, Self>::zero().set_by_mask(1 << n, T::one()) //
        })
    }

    fn basis_sparse<T>() -> [SparseMultivector<T, Self>; DIM]
    where
        T: Ring + Clone,
        Self: Sized,
    {
        make_array(|n| {
            SparseMultivector::<T, Self>::zero().set_by_mask(1 << n, T::one()) //
        })
    }
}

/// A trait that's assigned automatically to algebras of signature `[+,-,+,-,...,+,-]`.
/// For algebras with split signature FFT is possible without complexification.
pub trait SplitSignature: ClAlgebraBase {}

/// A trait that's assigned automatically to algebras of a signature with no null basis vectors.
pub trait NonDegenerate: ClAlgebraBase {}

pub trait ComplexAlgebra: ClAlgebraBase + DivisionAlgebra {}
pub trait QuaternionicAlgebra: ClAlgebraBase + DivisionAlgebra {}
pub trait DivisionAlgebra: ClAlgebraBase {}

impl<AB> ClAlgebra for AB
where
    AB: ClAlgebraBase,
{
    fn blade_label(idx: IndexType) -> String {
        (0..AB::dim())
            .rev()
            .filter(|i| (idx & (1 << *i)) != 0)
            .map(|i| AB::axis_name(i))
            .join("^")
    }

    fn blade_label_rev(idx: IndexType) -> String {
        (0..AB::dim())
            .filter(|i| (idx & (1 << *i)) != 0)
            .map(|i| AB::axis_name(i))
            .join("^")
    }

    /// Sign for the product of sequences of anticommuting values
    fn ac_product_sign(a: IndexType, b: IndexType) -> Sign {
        let mut transpositions = 0;
        for s in (1..(AB::dim() as usize)).rev() {
            let swapped_positions = (a << s) & b;
            transpositions += swapped_positions.count_ones();
        }
        if transpositions & 1 == 0 {
            Sign::Plus
        } else {
            Sign::Minus
        }
    }

    fn blade_wedge_product_sign(a: IndexType, b: IndexType) -> Sign {
        if (a & b) != 0 {
            return Sign::Null;
        }
        AB::ac_product_sign(a, b)
    }

    fn blade_geo_product_sign(a: IndexType, b: IndexType) -> Sign {
        if (a & b & AB::proj_mask()) != 0 {
            return Sign::Null;
        }
        (match (a & b & AB::imag_mask()).count_ones().is_even() {
            true => Sign::Plus,
            false => Sign::Minus,
        }) * AB::ac_product_sign(a, b)
    }

    fn index_iter() -> impl Iterator<Item = IndexType> {
        0..(1 << AB::dim())
    }

    fn grade_index_iter(weight: usize) -> impl Iterator<Item = IndexType> {
        index_utils::grade_iter(AB::dim(), weight)
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! produce_mask_internal_plus {
    ($shift:expr) => {0};
    ($shift:expr, +) => {
        (1 << $shift)
    };
    ($shift:expr, $anything:tt) => {
        (0 << $shift)
    };
    ($shift:expr, +, $($tx:tt),+) => {
        (1 << $shift) | $crate::produce_mask_internal_plus!(($shift + 1), $($tx),+)
    };
    ($shift:expr, $anything:tt, $($tx:tt),+) => {
        (0 << $shift) | $crate::produce_mask_internal_plus!(($shift + 1), $($tx),+)
    };
}
#[doc(hidden)]
#[macro_export]
macro_rules! produce_mask_plus {
    ($($tx:tt),+) => {$crate::produce_mask_internal_plus!(0, $($tx),+)}
}

#[doc(hidden)]
#[macro_export]
macro_rules! produce_mask_internal_minus {
    ($shift:expr) => {0};
    ($shift:expr, -) => {
        (1 << $shift)
    };
    ($shift:expr, $anything:tt) => {
        (0 << $shift)
    };
    ($shift:expr, -, $($tx:tt),+) => {
        (1 << $shift) | $crate::produce_mask_internal_minus!(($shift + 1), $($tx),+)
    };
    ($shift:expr, $anything:tt, $($tx:tt),+) => {
        (0 << $shift) | $crate::produce_mask_internal_minus!(($shift + 1), $($tx),+)
    };
}
#[doc(hidden)]
#[macro_export]
macro_rules! produce_mask_minus {
    ($($tx:tt),+) => {$crate::produce_mask_internal_minus!(0, $($tx),+)}
}

#[doc(hidden)]
#[macro_export]
macro_rules! produce_mask_internal_null {
    ($shift:expr) => {0};
    ($shift:expr, 0) => {
        (1 << $shift)
    };
    ($shift:expr, $anything:tt) => {
        (0 << $shift)
    };
    ($shift:expr, 0, $($tx:tt),+) => {
        (1 << $shift) | $crate::produce_mask_internal_null!(($shift + 1), $($tx),+)
    };
    ($shift:expr, $anything:tt, $($tx:tt),+) => {
        (0 << $shift) | $crate::produce_mask_internal_null!(($shift + 1), $($tx),+)
    };
}
#[doc(hidden)]
#[macro_export]
macro_rules! produce_mask_null {
    ($($tx:tt),+) => {$crate::produce_mask_internal_null!(0, $($tx),+)}
}

#[doc(hidden)]
#[macro_export]
macro_rules! count_items {
    () => {0};
    ($t:tt) => {1};
    ($t:tt, $($tx:tt),*) => {1 + $crate::count_items!($($tx),*)};
}

#[doc(hidden)]
#[macro_export]
macro_rules! order_check_end {
    () => {};
    (0) => {};
    (0, $($tx:tt),*) => {$crate::order_check_end!($($tx),*);};
    (-, $($tx:tt),*) => {compile_error!("The null axes should be the last ones in the signature");};
    (+, $($tx:tt),*) => {compile_error!("The null axes should be the last ones in the signature");};
    (-) => {compile_error!("The null axes should be the last ones in the signature");};
    (+) => {compile_error!("The null axes should be the last ones in the signature");};
}
#[doc(hidden)]
#[macro_export]
macro_rules! order_check {
    (+) => {};
    (-) => {};
    (0) => {};
    (+, $($tx:tt),+) => {$crate::order_check!($($tx),+);};
    (-, $($tx:tt),+) => {$crate::order_check!($($tx),+);};
    (0, $($tx:tt),*) => {$crate::order_check_end!($($tx),*);};
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_split_signature {
    ($name:ident, +,-) => {impl $crate::algebra::SplitSignature for $name {}};
    ($name:ident, +,-, $($s:tt),+) => {$crate::impl_split_signature!($name, $($s),+);};
    ($name:ident, $($s:tt),+) => {};
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_non_degenerate {
    ($name:ident, +) => {
        impl $crate::algebra::NonDegenerate for $name {}
    };
    ($name:ident, -) => {
        impl $crate::algebra::NonDegenerate for $name {}
    };
    ($name:ident, +, $($s:tt),+) => {
        $crate::impl_non_degenerate!($name, $($s),+);
    };
    ($name:ident, -, $($s:tt),+) => {
        $crate::impl_non_degenerate!($name, $($s),+);
    };
    ($name:ident, 0) => {};
    ($name:ident, 0, $($s:tt),*) => {};
}

#[doc(hidden)]
#[macro_export]
macro_rules! axis_name_func {
    ([$($signature:tt),+], [$($axes:literal),+]) => {
        fn axis_name(n: usize) -> String {
            static_assertions::const_assert_eq!($crate::count_items!($($signature),+), $crate::count_items!($($axes),+));
            String::from([$($axes),+][n])
        }
    };
    ([$($signature:tt),+], []) => {
        fn axis_name(n: usize) -> String {
            String::from("e") + &n.to_string()
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_complex_or_quaternionic {
    ($name:ident, [-]) => {
        impl $crate::algebra::ComplexAlgebra for $name {}
        impl $crate::algebra::DivisionAlgebra for $name {}
    };
    ($name:ident, [-,-]) => {
        impl $crate::algebra::QuaternionicAlgebra for $name {}
        impl $crate::algebra::DivisionAlgebra for $name {}
    };
    ($name:ident, [$($tx:tt),+]) => {};
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_algebra_base {
    ($name:ident, [$($signature:tt),+], [$($axes:literal),*]) => {
        impl $crate::algebra::ClAlgebraBase for $name {
            fn dim() -> usize {
                $crate::count_items!($($signature),+)
            }
            fn real_mask() -> $crate::types::IndexType {
                $crate::produce_mask_plus!($($signature),+)
            }
            fn imag_mask() -> $crate::types::IndexType {
                $crate::produce_mask_minus!($($signature),+)
            }
            fn proj_mask() -> $crate::types::IndexType {
                $crate::produce_mask_null!($($signature),+)
            }
            $crate::axis_name_func!([$($signature),+], [$($axes),*]);
        }
        impl $crate::algebra::ClBasis<{$crate::count_items!($($signature),+)}> for $name {}
        $crate::impl_split_signature!($name, $($signature),+);
        $crate::impl_non_degenerate!($name, $($signature),+);
        $crate::impl_complex_or_quaternionic!($name, [$($signature),+]);
    };
}

/**
`declare_algebra!` macro defines type for a Clifford algebra.

Arguments:
 - type name for the algebra
 - algebra signature: array containing `+`, `-`, `0`, signifying generators that square to +1, -1 and 0 respectively.
 - (optional) array of strings: names of the generators for displaying mulivectors as string.

Examples:

```
# use tclifford::declare_algebra;
declare_algebra!(PGA3, [+,+,+,0], ["x","y","z","e"]);
declare_algebra!(Spacetime, [+,-,-,-], ["t","x","y","z"]);
declare_algebra!(Cl8, [+,+,+,+,+,+,+,+]);
```

The algebra type implements TAlgebra trait, and then can be used for building multivectors in this algebra.
Multivectors from different algebras can not interact with each other.
*/
#[macro_export]
macro_rules! declare_algebra {
    // Public type definition
    (pub $name:ident, [$($signature:tt),+], [$($axes:literal),*]) => {
        $crate::order_check!($($signature),+);
        #[derive(Debug)]
        #[doc = concat!("Clifford algebra with signature `", stringify!([$($signature),+]), "`")]
        pub struct $name {}
        $crate::impl_algebra_base!($name, [$($signature),+], [$($axes),*]);
    };
    // Custom visibility type definition
    (pub($v:ident) $name:ident, [$($signature:tt),+], [$($axes:literal),*]) => {
        $crate::order_check!($($signature),+);
        #[derive(Debug)]
        #[doc = concat!("Clifford algebra with signature `", stringify!([$($signature),+]), "`")]
        pub($v) struct $name {}
        $crate::impl_algebra_base!($name, [$($signature),+], [$($axes),*]);
    };
    // Private type definition
    ($name:ident, [$($signature:tt),+], [$($axes:literal),*]) => {
        $crate::declare_algebra!(pub(self) $name, [$($signature),+], [$($axes),*]);
    };
    // Private type with automatic names for the generators
    ($name:ident, [$($signature:tt),+]) => {
        $crate::declare_algebra!($name, [$($signature),+], []);
    };
}

#[test]
fn test_labels() {
    declare_algebra!(ST, [-,+,+,+], ["t", "x", "y", "z"]);

    assert_eq!(ST::blade_label(0b0000), "");
    assert_eq!(ST::blade_label(0b0001), "t");
    assert_eq!(ST::blade_label(0b0010), "x");
    assert_eq!(ST::blade_label(0b0100), "y");
    assert_eq!(ST::blade_label(0b1011), "z^x^t");
}

#[test]
fn test_signs() {
    declare_algebra!(Cl22, [-,+,-,+]);

    assert_eq!(Cl22::blade_geo_product_sign(0b0001, 0b0001), Sign::Minus);
    assert_eq!(Cl22::blade_geo_product_sign(0b0010, 0b0010), Sign::Plus);
    assert_eq!(Cl22::blade_geo_product_sign(0b0100, 0b0100), Sign::Minus);
    assert_eq!(Cl22::blade_geo_product_sign(0b1000, 0b1000), Sign::Plus);

    assert_eq!(Cl22::blade_geo_product_sign(0b1001, 0b0001), Sign::Minus);
    assert_eq!(Cl22::blade_geo_product_sign(0b1010, 0b0010), Sign::Plus);
    assert_eq!(Cl22::blade_geo_product_sign(0b0101, 0b0100), Sign::Plus);
    assert_eq!(Cl22::blade_geo_product_sign(0b1001, 0b1000), Sign::Minus);

    assert_eq!(Cl22::blade_geo_product_sign(0b0001, 0b1001), Sign::Plus);
    assert_eq!(Cl22::blade_geo_product_sign(0b0010, 0b1010), Sign::Minus);
    assert_eq!(Cl22::blade_geo_product_sign(0b0100, 0b0101), Sign::Minus);
    assert_eq!(Cl22::blade_geo_product_sign(0b1000, 0b1001), Sign::Plus);
}
