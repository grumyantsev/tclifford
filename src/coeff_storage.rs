use ndarray::{Array1, ArrayView1, ArrayViewMut1};

use crate::index_utils;
use std::collections::HashMap;

use crate::types::{IndexType, Ring};

pub trait CoeffStorage<T>: PartialEq + Clone {
    fn new(dim: usize) -> Self;
    fn get_by_mask(&self, idx: IndexType) -> T;
    fn set_by_mask(&mut self, idx: IndexType, value: T);
    fn coeff_enumerate<'a>(&'a self) -> impl Iterator<Item = (IndexType, &'a T)>
    where
        T: 'a;
    fn grade_enumerate<'a>(&'a self, grade: usize) -> impl Iterator<Item = (IndexType, &'a T)>
    where
        T: 'a;
    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
}

#[derive(Debug, Clone)]
pub struct ArrayStorage<T> {
    dim: usize,
    array: ndarray::Array1<T>,
}

impl<T> CoeffStorage<T> for ArrayStorage<T>
where
    T: Ring + Clone,
{
    fn new(dim: usize) -> Self {
        ArrayStorage {
            dim,
            array: ndarray::Array1::zeros(1 << dim),
        }
    }

    fn get_by_mask(&self, idx: IndexType) -> T {
        if idx > self.array.len() {
            return T::zero();
        }
        self.array[idx].clone()
    }

    fn set_by_mask(&mut self, idx: IndexType, value: T) {
        if idx < self.array.len() {
            self.array[idx] = value;
        }
    }

    fn coeff_enumerate<'a>(&'a self) -> impl Iterator<Item = (IndexType, &'a T)>
    where
        T: 'a,
    {
        self.array.indexed_iter()
    }

    fn grade_enumerate<'a>(&'a self, grade: usize) -> impl Iterator<Item = (IndexType, &'a T)>
    where
        T: 'a,
    {
        index_utils::grade_iter(self.dim, grade).map(|idx| (idx, &self.array[idx]))
    }

    fn add(&self, rhs: &Self) -> Self {
        ArrayStorage {
            dim: self.dim,
            array: &self.array + &rhs.array,
        }
    }

    fn sub(&self, rhs: &Self) -> Self {
        ArrayStorage {
            dim: self.dim,
            array: &self.array - &rhs.array,
        }
    }
}

impl<'a, T> ArrayStorage<T> {
    pub fn array_view(&'a self) -> ArrayView1<'a, T> {
        self.array.view()
    }

    pub fn array_view_mut(&'a mut self) -> ArrayViewMut1<'a, T> {
        self.array.view_mut()
    }

    pub fn from_array(a: Array1<T>) -> Result<Self, ()> {
        let dim = index_utils::log2(a.len() as u64) as usize;
        if (1 << dim) != a.len() {
            Err(())
        } else {
            Ok(Self { dim, array: a })
        }
    }
}

impl<T> PartialEq for ArrayStorage<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.array == other.array
    }
}

#[derive(Debug, Clone)]
pub struct SparseStorage<T> {
    dim: usize,
    size: IndexType,
    coeffs: HashMap<IndexType, T>,
}

impl<T> CoeffStorage<T> for SparseStorage<T>
where
    T: Ring + Clone,
{
    fn new(dim: usize) -> Self {
        SparseStorage {
            dim,
            size: (1 << dim) as IndexType,
            coeffs: HashMap::new(),
        }
    }

    fn get_by_mask(&self, idx: IndexType) -> T {
        self.coeffs.get(&idx).cloned().unwrap_or(T::zero())
    }

    fn set_by_mask(&mut self, idx: IndexType, value: T) {
        if idx < self.size {
            if value.is_zero() {
                self.coeffs.remove(&idx);
            } else {
                self.coeffs.insert(idx, value);
            }
        }
    }

    fn coeff_enumerate<'a>(&'a self) -> impl Iterator<Item = (IndexType, &'a T)>
    where
        T: 'a,
    {
        self.coeffs.iter().map(|(idx, val)| (*idx, val))
    }

    fn grade_enumerate<'a>(&'a self, grade: usize) -> impl Iterator<Item = (IndexType, &'a T)>
    where
        T: 'a,
    {
        self.coeff_enumerate()
            .filter(move |(idx, _)| idx.count_ones() == (grade as u32))
    }

    fn add(&self, rhs: &Self) -> Self {
        let mut ret = SparseStorage::new(self.dim);
        for (idx, c) in self.coeff_enumerate() {
            ret.set_by_mask(idx, c.clone());
        }
        for (idx, c) in rhs.coeff_enumerate() {
            ret.set_by_mask(idx, ret.get_by_mask(idx).ref_add(c));
        }
        ret
    }

    fn sub(&self, rhs: &Self) -> Self {
        let mut ret = SparseStorage::new(self.dim);
        for (idx, c) in self.coeff_enumerate() {
            ret.set_by_mask(idx, c.clone());
        }
        for (idx, c) in rhs.coeff_enumerate() {
            ret.set_by_mask(idx, ret.get_by_mask(idx).ref_sub(c));
        }
        ret
    }
}

impl<T> PartialEq for SparseStorage<T>
where
    T: Clone + Ring,
{
    fn eq(&self, other: &Self) -> bool {
        self.coeff_enumerate()
            .all(|(idx, c)| other.get_by_mask(idx).eq(c))
            && other
                .coeff_enumerate()
                .all(|(idx, c)| self.get_by_mask(idx).eq(c))
    }
}
