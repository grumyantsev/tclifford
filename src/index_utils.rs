use crate::types::IndexType;

#[derive(Debug, Clone)]
pub struct GradeIter {
    weight: usize, // weight of numbers to produce
    pos: isize,    // highest bit position at this iteration

    // this iterator is recursive, and this is basically a callstack
    subiter: Option<Box<GradeIter>>,
}

impl GradeIter {
    // Avoid memory reallocations for subiters re-creation, just reuse the old ones
    fn refresh(&mut self, len: isize) {
        self.pos = len - 1;
        //weight stays unchanged
        if !self.subiter.is_none() {
            self.subiter.as_mut().unwrap().refresh(self.pos)
        }
    }
}

impl std::iter::Iterator for GradeIter {
    type Item = IndexType;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos + 1 < self.weight as isize {
            return None;
        }

        if self.weight == 0 {
            // Return 0 and ensure that the next iteration will return None
            self.pos = -2;
            return Some(0);
        }

        if self.subiter.is_none() {
            // If we just started the iteration subiter is None, create a new one
            self.subiter = Some(Box::new(grade_iter(self.pos as usize, self.weight - 1)));
        }

        let subiter = self.subiter.as_mut().unwrap();
        match subiter.next() {
            Some(subindex) => Some((1 << self.pos) | subindex),
            None => {
                // subiter finished, shift the highest bit position
                self.pos -= 1;
                subiter.refresh(self.pos);
                self.next()
            }
        }
    }
}

/**
 * Create an iterator that produces all integers of bit length `<= len` with exactly `weight` bits set to 1.
 */
pub fn grade_iter(len: usize, weight: usize) -> GradeIter {
    GradeIter {
        weight: weight,
        pos: (len as isize) - 1,
        subiter: None,
    }
}

#[test]
fn grade_iter_test() {
    for n in 0..16 {
        // all 2^n indices of the check_arr must be set after running all the grade iters
        let mut check_arr = vec![];
        for _ in 0..(1 << n) {
            check_arr.push(false)
        }

        for k in 0..(n + 1) {
            let mut count = 0;
            for idx in grade_iter(n, k) {
                assert!((idx.count_ones() as usize) == k);
                count += 1;

                assert!(check_arr[idx] == false);
                check_arr[idx] = true;
                //println!("{:b}", idx)
            }
            assert!(count == num::integer::binomial(n, k));
        }
        assert!(grade_iter(n, n + 1).count() == 0);
        assert!(check_arr.iter().all(|x| *x));
    }
}
