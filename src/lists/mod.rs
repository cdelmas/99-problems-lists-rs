fn last<A>(l: &Vec<A>) -> Option<&A> {
    l.last()
}

fn but_last<A>(l: &Vec<A>) -> Option<&A> {
    if l.len() >= 2 {
        l.get(l.len() - 2)
    } else {
        None
    }
}

fn k_th<A>(l: &Vec<A>, index: usize) -> Option<&A> {
    if index == 0 {
        None
    } else {
        l.get(index - 1)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use quickcheck::TestResult;
    use quickcheck_macros::*;

    #[test]
    fn it_does_pass() {
        assert_eq!(1, 1);
    }

    #[quickcheck]
    fn last_is_reverse_head(list: Vec<isize>) -> bool {
        let mut reversed = list.clone();
        reversed.reverse();
        reversed.first() == last(&list)
    }

    #[quickcheck]
    fn but_last_is_reverse_second(list: Vec<isize>) -> bool {
        let mut reversed = list.clone();
        reversed.reverse();
        reversed.get(1) == but_last(&list)
    }

    #[quickcheck]
    fn kth_with_index_bigger_than_size_is_none(list: Vec<isize>, idx: usize) -> TestResult {
        if idx < list.len() && idx != 0 {
            TestResult::discard()
        } else {
            TestResult::from_bool(k_th(&list, idx).is_none())
        }
    }

    #[quickcheck]
    fn kth_with_index_in_range_1_len_is_some(list: Vec<isize>, idx: usize) -> TestResult {
        if idx == 0 || idx >= list.len() {
            TestResult::discard()
        } else {
            TestResult::from_bool(k_th(&list, idx).is_some())
        }
    }
}
