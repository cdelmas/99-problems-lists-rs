fn is_prime(n: u64) -> bool {
    if n < 2 || (n != 2 && n % 2 == 0) {
        false
    } else {
        let limit: u64 = (n as f64).sqrt().floor() as u64;
        for i in (3..limit).step_by(2) {
            if n % i == 0 {
                return false;
            }
        }
        true
    }
}

fn gcd(n: u32, m: u32) -> u32 {
    if n == 0 || m == 0 {
        panic!()
    } else {
        let r = n % m;
        if r == 0 {
            m
        } else {
            gcd(m, r)
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use proptest::prelude::*;

    #[test]
    fn two_is_prime() {
        assert!(is_prime(2))
    }

    proptest! {
        #[test]
        fn even_numbers_are_not_prime_except_2(n in (4u64..1000000).prop_filter("is_even", |x| x%2 == 0)  ) {
            prop_assert!( !is_prime(n))
        }
    }

    // test with well-known primes
    proptest! {

        #[test]
        fn well_known_primes_are_primes(n in (2u64..1000).prop_filter("well known prime", |x| {
            let primes = vec![2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997];
            primes.contains(x)
        }  ) ) {
            prop_assert!(is_prime(n))
        }
    }

    // GCD Euclid algorithm
    use num::integer::gcd as oracle;

    #[test]
    #[should_panic]
    fn gcd_does_not_compute_for_n_eq_0() {
        gcd(0, 42);
    }

    #[test]
    #[should_panic]
    fn gcd_does_not_compute_for_m_eq_0() {
        gcd(666, 0);
    }

    proptest! {
        #[test]
        fn gcd_is_same_as_oracle(n in 1u32..100000, m in 1u32..100000) {
            prop_assert_eq!(oracle(n, m), gcd(n, m))
        }
    }

    
}
