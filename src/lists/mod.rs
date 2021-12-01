fn last<A>(l: &[A]) -> Option<&A> {
    l.last()
}

fn but_last<A>(l: &[A]) -> Option<&A> {
    if l.len() >= 2 {
        l.get(l.len() - 2)
    } else {
        None
    }
}

fn k_th<A>(l: &[A], index: usize) -> Option<&A> {
    if index == 0 {
        None
    } else {
        l.get(index - 1)
    }
}

fn length<A>(l: &[A]) -> usize {
    l.len()
}

fn reverse<A: Clone>(l: &[A]) -> Vec<A> {
    let mut copy = (*l).to_vec();
    copy.reverse();
    copy
}

fn is_palindrome<A: Eq>(l: &[A]) -> bool {
    let (forward, backward) = l.split_at(l.len() / 2);
    let forward = forward.iter();
    let backward = backward.iter().rev();
    forward.zip(backward).all(|(a, b)| b == a)
}

#[derive(Debug, Clone)]
enum NestedList<A> {
    Elem(A),
    List(Vec<Box<NestedList<A>>>),
}

fn flatten<A>(list: &NestedList<A>) -> Vec<&A> {
    let mut res: Vec<&A> = vec![];
    match list {
        NestedList::Elem(e) => {
            res.push(e);
        }
        NestedList::List(v) => {
            for nl in v {
                let mut sub = flatten(nl);
                res.append(&mut sub);
            }
        }
    }
    res
}

fn remove_consecutive_duplicates<A: Eq>(list: &[A]) -> Vec<&A> {
    list.iter().fold(vec![], |mut acc, e| match acc.last() {
        Some(last) if last == &e => acc,
        _ => {
            acc.push(e);
            acc
        }
    })
}

fn pack_consecutive_duplicates<A: Eq>(list: &[A]) -> Vec<Vec<&A>> {
    list.iter()
        .fold(vec![] as Vec<Vec<&A>>, |mut acc, e| match acc.last_mut() {
            Some(prev) if prev.contains(&e) => {
                prev.push(e);
                acc
            }
            _ => {
                acc.push(vec![e]);
                acc
            }
        })
}

fn occurrences<A: Eq>(list: &[A]) -> Vec<(usize, &A)> {
    pack_consecutive_duplicates(list)
        .iter()
        .map(|v| (v.len(), v[0]))
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
enum Occurrence<'a, A> {
    Single(&'a A),
    Multiple(usize, &'a A),
}

impl<'a, A> Occurrence<'a, A>
where
    A: Eq,
{
    fn new(size: usize, elem: &'a A) -> Self {
        if size == 1 {
            Occurrence::Single(elem)
        } else {
            Occurrence::Multiple(size, elem)
        }
    }

    fn can_increment(self: &Occurrence<'a, A>, elem: &'a A) -> bool {
        match &self {
            Occurrence::Single(a) => *a == elem,
            Occurrence::Multiple(_, a) => *a == elem,
        }
    }

    fn increment(occurrence: &'a Occurrence<A>, elem: &'a A) -> Option<Occurrence<'a, A>> {
        match &occurrence {
            Occurrence::Single(a) if *a == elem => Some(Occurrence::Multiple(2, *a)),
            Occurrence::Multiple(n, a) if *a == elem => Some(Occurrence::Multiple(n + 1, *a)),
            _ => None,
        }
    }
}

fn encode<A: Eq>(list: &[A]) -> Vec<Occurrence<A>> {
    pack_consecutive_duplicates(list)
        .iter()
        .map(|v| Occurrence::new(v.len(), v[0]))
        .collect()
}

fn decode<'a, A>(list: &'a [Occurrence<A>]) -> Vec<&'a A> {
    list.iter()
        .fold(vec![] as Vec<&'a A>, |mut acc, o| match *o {
            Occurrence::Single(x) => {
                acc.push(x);
                acc
            }
            Occurrence::Multiple(n, x) => {
                let mut expanded = vec![x; n];
                acc.append(&mut expanded);
                acc
            }
        })
}

fn encode_no_intermediary<A: Eq>(list: &[A]) -> Vec<Occurrence<A>> {
    list.iter()
        .fold(vec![] as Vec<Occurrence<A>>, |mut acc, e| {
            let value = match acc.pop() {
                Some(Occurrence::Single(a)) if a == e => Occurrence::Multiple(2, e),
                Some(Occurrence::Multiple(n, a)) if a == e => Occurrence::Multiple(n + 1, e),
                Some(old) => {
                    acc.push(old);
                    Occurrence::Single(e)
                }
                None => Occurrence::Single(e),
            };
            acc.push(value);
            acc
        })
}

fn duplicate<A: Clone>(list: &[A]) -> Vec<A> {
    list.iter().fold(vec![], |mut acc, e| {
        acc.push(e.clone());
        acc.push(e.clone());
        acc
    })
}

fn replicate<A: Clone>(list: &[A], repl_num: u8) -> Vec<A> {
    list.iter().fold(vec![], |mut acc, e| {
        for _ in 0..repl_num {
            acc.push(e.clone())
        }
        acc
    })
}

#[cfg(test)]
mod test {
    use super::*;

    use proptest::prelude::*;
    use NestedList::*;

    proptest! {
        #[test]
        fn last_is_reverse_head(list: Vec<isize>) {
            let mut reversed = list.clone();
            reversed.reverse();
            prop_assert_eq!(reversed.first(), last(&list));
        }
    }

    proptest! {
        #[test]
        fn but_last_is_reverse_second(list: Vec<isize>) {
            let mut reversed = list.clone();
            reversed.reverse();
            prop_assert_eq!(reversed.get(1), but_last(&list));
        }
    }

    #[derive(Clone)]
    enum IndexRange {
        OutOfBounds,
        InBounds(usize), // defines an offset
    }

    fn vec_and_index(range: IndexRange) -> impl Strategy<Value = (Vec<isize>, usize)> {
        prop::collection::vec(any::<isize>(), 1..100).prop_flat_map(move |vec| {
            let len = vec.len();
            let index_range = match range.clone() {
                IndexRange::OutOfBounds => len..usize::MAX,
                IndexRange::InBounds(n) => n..len + n,
            };
            (Just(vec), index_range)
        })
    }

    proptest! {
        #[test]
        fn kth_with_index_bigger_than_size_is_none(list_and_idx in vec_and_index(IndexRange::OutOfBounds)) {
            let (list, idx) = list_and_idx;
            prop_assert!(k_th(&list, idx).is_none());
        }
    }

    proptest! {
        #[test]
        fn kth_with_index_in_range_1_len_is_some(list_and_idx in vec_and_index(IndexRange::InBounds(1))) {
            let (list, idx) = list_and_idx;
            prop_assert!(k_th(&list, idx).is_some());
        }
    }

    proptest! {
        #[test]
        fn reverse_the_reversed_list_gives_the_original(list: Vec<isize>) {
            prop_assert_eq!(reverse(&reverse(&list)), list);
        }
    }

    #[test]
    fn is_a_palindrom() {
        assert_eq!(true, is_palindrome(&vec![1, 2, 4, 2, 1]));
    }

    proptest! {
        #[test]
        fn concatenate_the_reversed_is_a_palindrom(list: Vec<isize>) {
            let mut copy = list.clone();
            copy.reverse();
            let mut v = list.clone();
            v.append(&mut copy);
            prop_assert!(is_palindrome(&v));
        }
    }

    fn nested_list<A: Arbitrary + Clone + 'static>() -> impl Strategy<Value = NestedList<A>> {
        let leaf = any::<A>().prop_map(|a| NestedList::Elem(a));
        leaf.prop_recursive(
            8,   // 8 levels deep
            256, // Shoot for maximum size of 256 nodes
            10,  // We put up to 10 items per collection
            |inner| {
                prop::collection::vec(inner.clone(), 0..10).prop_map(|v| {
                    NestedList::List(v.into_iter().map(|e| Box::new(e.clone())).collect())
                })
            },
        )
    }

    #[test]
    fn flatten_an_elem_is_a_vector_with_the_elem() {
        let e = Elem(3);

        let result = flatten(&e);

        assert_eq!(vec![&3], result);
    }

    fn count_leaves<A>(list: &NestedList<A>) -> usize {
        match list {
            NestedList::Elem(_) => 1,
            NestedList::List(v) => v.iter().map(|bl| count_leaves(bl)).sum(),
        }
    }

    proptest! {
        #[test]
        fn vector_size_is_nodes_size(list in nested_list::<usize>()) {
            let leaf_number = count_leaves(&list);
            prop_assert_eq!(leaf_number, flatten(&list).len());
        }
    }

    #[test]
    fn flattened_contains_all_elems() {
        let l = List(vec![
            Box::new(List(vec![Box::new(Elem(4))])),
            Box::new(Elem(2)),
            Box::new(List(vec![Box::new(Elem(6))])),
            Box::new(Elem(17)),
        ]);
        let res = flatten(&l);

        assert_eq!(vec![&4, &2, &6, &17], res);
    }

    proptest! {
        #[test]
        fn removing_duplicates_leaves_a_smaller_or_equal_length_list(list: Vec<isize>) {
            let no_dups = remove_consecutive_duplicates(&list);
            prop_assert!(
                (list.len() == 0 && no_dups.len() == 0)
                || (no_dups.len() <= list.len() && no_dups.len() > 0)
            );
        }
    }

    // another property for removing duplicate: elements in the compressed vector appear in the same order than in the original

    proptest! {
        #[test]
        fn packing_leads_to_the_same_length_as_removing_dups(list: Vec<isize>) {
            let no_dups = remove_consecutive_duplicates(&list);
            let packed = pack_consecutive_duplicates(&list);

            prop_assert_eq!(no_dups.len(), packed.len());
        }
    }

    #[test]
    fn occurrences_tests() {
        let v = vec![1, 1, 2, 3, 3, 3, 2, 3, 3, 2, 2];
        let res = occurrences(&v);

        assert_eq!(res.len(), 6);

        assert_eq!(
            vec![(2, &1), (1, &2), (3, &3), (1, &2), (2, &3), (2, &2)],
            res
        );
    }

    #[test]
    fn encode_tests() {
        use Occurrence::*;
        let v = vec![1, 1, 2, 3, 3, 3, 2, 3, 3, 2, 2];
        let res = encode(&v);

        assert_eq!(6, res.len());

        assert_eq!(
            vec![
                Multiple(2, &1),
                Single(&2),
                Multiple(3, &3),
                Single(&2),
                Multiple(2, &3),
                Multiple(2, &2)
            ],
            res
        );
    }

    proptest! {
        #[test]
        fn encoding_then_decoding_gives_the_original_array(list: Vec<i32>) {
            let initial: Vec<&i32> = list.iter().collect();
            let encoded = encode(&list);
            let decoded = decode(&encoded);
            prop_assert_eq!(initial, decoded);
        }
    }

    #[test]
    fn decode_tests() {
        use Occurrence::*;
        let v = vec![
            Multiple(2, &1),
            Single(&2),
            Multiple(3, &3),
            Single(&2),
            Multiple(2, &3),
            Multiple(2, &2),
        ];

        let res = decode(&v);

        assert_eq!(11, res.len());
        assert_eq!(vec![&1, &1, &2, &3, &3, &3, &2, &3, &3, &2, &2], res);
    }

    #[test]
    fn encode_no_intermediary_tests() {
        use Occurrence::*;
        let v = vec![1, 1, 2, 3, 3, 3, 2, 3, 3, 2, 2];
        let res = encode_no_intermediary(&v);

        assert_eq!(6, res.len());

        assert_eq!(
            vec![
                Multiple(2, &1),
                Single(&2),
                Multiple(3, &3),
                Single(&2),
                Multiple(2, &3),
                Multiple(2, &2)
            ],
            res
        );
    }

    proptest! {
        #[test]
        fn encode_no_intermediary_should_have_same_results_as_encode(list: Vec<i64>) {
            prop_assert_eq!(encode(&list), encode_no_intermediary(&list));
        }
    }

    proptest! {
        #[test]
        fn duplicates_has_two_times_the_size_of_the_original(list: Vec<isize>) {
            prop_assert_eq!(list.len()*2, duplicate(&list).len());
        }
    }

    proptest! {
        #[test]
        fn each_element_in_the_original_is_in_the_duplicates(list: Vec<isize>) {
            let res = duplicate(&list);
            prop_assert!(list.iter().all(|o| res.contains(&o)));
        }
    }

    proptest! {
        #[test]
        fn replicates_has_two_times_the_size_of_the_original(list: Vec<isize>, repl_num in 2..10) {
            prop_assert_eq!(list.len()*repl_num as usize, replicate(&list, repl_num as u8).len());
        }
    }

    proptest! {
        #[test]
        fn each_element_in_the_original_is_in_the_replicates(list: Vec<isize>, repl_num in 2..10) {
            let res = replicate(&list, repl_num as u8);
            prop_assert!(list.iter().all(|o| res.contains(&o)));
        }
    }
}
