pub(crate) const AC_PRODUCT_SIGNS: [&[u64]; 7] = [
    &[0x0],
    &[0x0, 0x0],
    &[0x0, 0xc, 0x0, 0xc],
    &[0x0, 0x3c, 0xf0, 0xcc, 0x0, 0x3c, 0xf0, 0xcc],
    &[
        0x0, 0xc33c, 0xff0, 0xcccc, 0xff00, 0x3c3c, 0xf0f0, 0x33cc, 0x0, 0xc33c, 0xff0, 0xcccc,
        0xff00, 0x3c3c, 0xf0f0, 0x33cc,
    ],
    &[
        0x0, 0x3cc3c33c, 0xf00f0ff0, 0xcccccccc, 0xffff00, 0x3c3c3c3c, 0xf0f0f0f0, 0xcc3333cc,
        0xffff0000, 0xc33cc33c, 0xff00ff0, 0x3333cccc, 0xff00ff00, 0xc3c33c3c, 0xf0ff0f0,
        0x33cc33cc, 0x0, 0x3cc3c33c, 0xf00f0ff0, 0xcccccccc, 0xffff00, 0x3c3c3c3c, 0xf0f0f0f0,
        0xcc3333cc, 0xffff0000, 0xc33cc33c, 0xff00ff0, 0x3333cccc, 0xff00ff00, 0xc3c33c3c,
        0xf0ff0f0, 0x33cc33cc,
    ],
    &[
        0x0,
        0xc33c3cc33cc3c33c,
        0xff0f00ff00f0ff0,
        0xcccccccccccccccc,
        0xff0000ff00ffff00,
        0x3c3c3c3c3c3c3c3c,
        0xf0f0f0f0f0f0f0f0,
        0x33cccc33cc3333cc,
        0xffffffff0000,
        0xc33cc33cc33cc33c,
        0xff00ff00ff00ff0,
        0xcccc33333333cccc,
        0xff00ff00ff00ff00,
        0x3c3cc3c3c3c33c3c,
        0xf0f00f0f0f0ff0f0,
        0x33cc33cc33cc33cc,
        0xffffffff00000000,
        0x3cc3c33c3cc3c33c,
        0xf00f0ff0f00f0ff0,
        0x33333333cccccccc,
        0xffff0000ffff00,
        0xc3c3c3c33c3c3c3c,
        0xf0f0f0ff0f0f0f0,
        0xcc3333cccc3333cc,
        0xffff0000ffff0000,
        0x3cc33cc3c33cc33c,
        0xf00ff00f0ff00ff0,
        0x3333cccc3333cccc,
        0xff00ffff00ff00,
        0xc3c33c3cc3c33c3c,
        0xf0ff0f00f0ff0f0,
        0xcc33cc3333cc33cc,
        0x0,
        0xc33c3cc33cc3c33c,
        0xff0f00ff00f0ff0,
        0xcccccccccccccccc,
        0xff0000ff00ffff00,
        0x3c3c3c3c3c3c3c3c,
        0xf0f0f0f0f0f0f0f0,
        0x33cccc33cc3333cc,
        0xffffffff0000,
        0xc33cc33cc33cc33c,
        0xff00ff00ff00ff0,
        0xcccc33333333cccc,
        0xff00ff00ff00ff00,
        0x3c3cc3c3c3c33c3c,
        0xf0f00f0f0f0ff0f0,
        0x33cc33cc33cc33cc,
        0xffffffff00000000,
        0x3cc3c33c3cc3c33c,
        0xf00f0ff0f00f0ff0,
        0x33333333cccccccc,
        0xffff0000ffff00,
        0xc3c3c3c33c3c3c3c,
        0xf0f0f0ff0f0f0f0,
        0xcc3333cccc3333cc,
        0xffff0000ffff0000,
        0x3cc33cc3c33cc33c,
        0xf00ff00f0ff00ff0,
        0x3333cccc3333cccc,
        0xff00ffff00ff00,
        0xc3c33c3cc3c33c3c,
        0xf0ff0f00f0ff0f0,
        0xcc33cc3333cc33cc,
    ],
];

#[cfg(test)]
mod test {
    use super::AC_PRODUCT_SIGNS;
    use crate::algebra::ac_product_sign;
    use crate::types::Sign;

    #[test]
    fn precompute_test() {
        for dim in 0..=6 {
            let mut signs: Vec<u64> = vec![];
            for i in 0..(1 << dim) {
                signs.push(0);
                for j in 0..(1 << dim) {
                    signs[i] |= match ac_product_sign(dim, i, j) {
                        Sign::Null => unreachable!(),
                        Sign::Plus => 0,
                        Sign::Minus => 1,
                    } << j;
                }
            }
            println!("&[");
            for (i, x) in signs.iter().enumerate() {
                println!("{:#x},", x);
                assert!(AC_PRODUCT_SIGNS[dim][i] == *x);
            }
            println!("],");
        }
    }
}
