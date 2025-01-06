# tclifford

Clifford algebras implementation for Rust.


## Features

- Support for arbitrary signatures. \
For example, `declare_algebra!(A, [+,+,-,-,0])` defines a type `A` corresponding to Cl(2,2,1).

- Strong typing. Operations on two multivectors are only allowed when they belong to the same algebra. Everything that can be checked in compile time is checked in compile time.

- Abstraction from the underlying field. Any type that implements arithmetic operations can be used. So, Clifford algebras over fields other that R and C can be represented, as well as constructions like direct sum of different Clifford algebras.

- Optimizations for high dimensional algebras. \
The type `FFTRepr` is a special representation of multivectors based on the [Fast matrix representation](https://arxiv.org/abs/2410.06103) algorithm. \
It allows fast multiplication of high-dimensional dense multivectors, as well as easily finding the inverse multivectors.

- Support for both "dense" and "sparse" multivectors. The operations limited to low grades can be faster with usage of the sparse ones. But be vary of floating point errors that can introduce unintended near-zero coefficients.


## Development status

In active development. Any API can be changed at any time.

TODOs:
- Rework and systematize tests.
- Refactor interfaces based on the user experience.
- Implement more operations that can be useful.
- Implement FFTRepr for types other than Complex64. Both for real representations of split algebras, and for other fields where sqrt(-1) exists.

## Examples

A simple example of usage for PGA. The `PGAMV` trait is only implemented for algebras of appropriate signature. 
```rust
// Create types for the algebra and multivectors in it
declare_algebra!(PGA2, [+,+,0], ["x", "y", "n"]);
type M = Multivector<f64, PGA2>;

// Create PGA points
let ax = M::pga_point([10., 0.]).unwrap();
let ay = M::pga_point([0., 8.]).unwrap();

// Line connecting the two points
let line1 = ax.meet(&ay);

// Line defined by the equation `-2 x + 5 y - 10 = 0`
let line2 = M::pga_hyperplane([-2., 5., -10.]).unwrap();

// Find the intersection point
let intersection = line1.wedge(&line2);
println!("{:?}", intersection.pga_extract_point()); // [5.0, 4.0]
assert_eq!(intersection.pga_extract_point(), [5., 4.]);

```


Advanced example: produce the 3 irreducible representations of a random element of Spin(8). (Utilizing some [ndarray](https://docs.rs/ndarray/latest/ndarray/))
```rust
// Create types for the algebra and multivectors in it
declare_algebra!(Cl8, [+,+,+,+,+,+,+,+]);
type MV = Multivector<f64, Cl8>;

// Generate a random unitary blade.
let a = MV::from_vector((0..8).map(|_| rand::random::<f64>() - 0.5)).unwrap();
let b = MV::from_vector((0..8).map(|_| rand::random::<f64>() - 0.5)).unwrap();
let mut blade = a.wedge(&b);
blade = &blade / blade.mag2().sqrt(); // normalize it

// Produce a rotor generated by rotation in this plane by a random angle.
let angle = PI * rand::random::<f64>();
// Or, rather, its full-spin representation.
let rot: FFTRepr<Cl8> = (blade * angle).fft().exp();

// Compute the SO(8) representation from the rotor action on the basis vectors.
let e: [MV; 8] = MV::basis();
let mut so_repr = ndarray::Array2::default((8, 8));
for (i, ei) in e.iter().enumerate() {
    // Apply the rotor to each basis vector
    let rotated_ei: MV = (rot.rev() * ei.fft() * &rot).ifft();
    // And collect the results into a matrix
    let ei_column = rotated_ei.extract_vector();
    so_repr
        .index_axis_mut(ndarray::Axis(1), i)
        .assign(&ei_column);
}
// Check that the resulting matrix indeed has a determinant of 1 (allowing some floating point error)
assert!((so_repr.det().unwrap() - 1.0).abs() < 1e-12);

// Irreducible spin representations are built-in!
let (even_spin_repr, odd_spin_repr) = rot.into_half_spin_repr();

println!("The 3 irreducible representations of this Spin(8) element are:");
println!("{:.2}", so_repr);
println!("{:.2}", even_spin_repr);
println!("{:.2}", odd_spin_repr);
```
