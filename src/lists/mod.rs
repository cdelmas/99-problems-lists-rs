use itertools::Itertools;
use rand::seq::SliceRandom;
use rand::Rng;
use std::cmp::min;

fn last<A>(l: &[A]) -> Option<&A> {
    //l.last()
    match l {
        [] => None,
        [.., a] => Some(a),
    }
}

fn but_last<A>(l: &[A]) -> Option<&A> {
    match l {
        [] | [_] => None,
        [.., x, _] => Some(x),
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
    let mut copy = (*l).to_vec(); // clones all the A in the vector; could use l.iter().rev().collect() but would get a Vec<&A> instead
    copy.reverse();
    copy
}

fn is_palindrome<A: Eq>(l: &[A]) -> bool {
    l.iter().zip(l.iter().rev()).all(|(a, b)| b == a)
}

#[derive(Debug, Clone)]
enum NestedList<A> {
    Elem(A),
    List(Vec<Box<NestedList<A>>>),
}

fn flatten<A>(list: &NestedList<A>) -> Vec<&A> {
    match list {
        NestedList::Elem(e) => vec![e],
        NestedList::List(v) => v.iter().flat_map(|e| flatten(e)).collect(),
    }
}

fn remove_consecutive_duplicates<A: Eq>(list: &[A]) -> Vec<&A> {
    let mut vec: Vec<&A> = list.iter().collect();
    vec.dedup();
    vec
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
        .flat_map(|o| match o {
            Occurrence::Single(x) => vec![*x],
            Occurrence::Multiple(ref n, x) => vec![*x; *n],
        })
        .collect()
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
    replicate(list, 2)
}

fn replicate<A: Clone>(list: &[A], repl_num: u8) -> Vec<A> {
    list.iter()
        .flat_map(|e| vec![e.clone(); repl_num as usize])
        .collect()
}

fn drop_every_nth<A>(list: &[A], each: usize) -> Vec<&A> {
    list.iter()
        .enumerate()
        .filter_map(|(i, e)| {
            if each == 0 || i % each < each - 1 {
                Some(e)
            } else {
                None
            }
        })
        .collect()
}

fn split_n<A>(list: &[A], at: usize) -> (&[A], &[A]) {
    if at > list.len() {
        (list, &[])
    } else {
        list.split_at(at)
    }
}

fn slice<A>(list: &[A], from: usize, to: usize) -> Option<&[A]> {
    if to < from || from >= list.len() || to >= list.len() {
        None
    } else {
        Some(&list[from..(to + 1)])
    }
}

fn rotate<A: Clone>(list: &[A], n: isize) -> Vec<A> {
    //(len + (index % len)) % length
    let len = list.len() as isize;
    let split_index = (n.checked_rem_euclid(len).unwrap_or_default() + len)
        .checked_rem_euclid(len)
        .unwrap_or_default();

    let (first, second) = split_n(list, split_index as usize);
    [second, first].concat()
}

fn remove_at<A: Clone>(list: &[A], n: usize) -> Option<(A, Vec<A>)> {
    if n >= list.len() {
        None
    } else {
        let mut v = list.to_vec();
        let e = v.remove(n);
        Some((e, v))
    }
}

fn insert_at<A: Clone>(list: &[A], a: A, at: usize) -> Vec<A> {
    let mut v = list.to_vec();
    v.insert(min(at, list.len()), a);
    v
}

fn range(from: u32, to: u32) -> Vec<u32> {
    (from..to + 1).collect()
}

fn random_select<'a, A, R>(list: &'a [A], n: usize, rng: &mut R) -> Vec<&'a A>
where
    R: Rng,
{
    list.choose_multiple(rng, n).collect()
}

fn lottery_draw<R>(upper_bound: u32, n: u32, rng: &mut R) -> Vec<u32>
where
    R: Rng,
{
    let numbers: Vec<u32> = (1..upper_bound + 1).collect();
    random_select(&numbers, n as usize, rng)
        .into_iter()
        .cloned()
        .collect()
}

fn random_permutation<'a, A, R>(list: &'a [A], rng: &mut R) -> Vec<&'a A>
where
    R: Rng,
{
    let mut v: Vec<&A> = list.iter().collect();
    v.shuffle(rng);
    v
}

fn combinations<A>(list: &[A], n: usize) -> Vec<Vec<&A>> {
    list.iter().combinations(n).collect()
}

fn group<A>(sizes: &[u8], list: &[A]) -> Option<Vec<Vec<Vec<A>>>>
where
    A: Clone + PartialEq,
{
    type Group<A> = Vec<A>;
    type Groups<A> = Vec<Group<A>>;
    if sizes.iter().sum::<u8>() as usize != list.len() {
        // NOTE: sum could overflow, and it is not handled
        None
    } else {
        // accumulator: Vec(partial solutions, remaining elements);
        // when remaining is empty (= when we finish the fold), we have the solutions
        let init: Vec<(Groups<A>, Vec<A>)> = vec![(vec![], list.to_vec())];
        Some(
            sizes
                .iter()
                .fold(init, |acc, size| {
                    acc.iter()
                        .flat_map(|(partial_solution, remaining_elements)| {
                            remaining_elements
                                .iter()
                                .combinations(*size as usize) // compute all combinations of s elements in remaining_elements
                                .map(move |combo| {
                                    remaining_elements
                                        .iter()
                                        .cloned()
                                        .partition(|e| combo.contains(&e))
                                }) // for each combination, get the remaining elements
                                .map(move |(combo, new_remaining_elements)| {
                                    // for each pair combo/remaining elements,
                                    // add the combo to the partial solution,
                                    // and keep the remaining elements for the next turn
                                    let mut new_partial_solution = vec![];
                                    new_partial_solution.extend(partial_solution.clone());
                                    new_partial_solution.push(combo);
                                    (new_partial_solution, new_remaining_elements)
                                })
                        })
                        .collect::<Vec<(Groups<A>, Vec<A>)>>()
                })
                .iter()
                .map(|x| x.0.clone())
                .collect::<Vec<Groups<A>>>(),
        ) // solutions are the _.0 part (_.1 is [])
    }
}

fn sort_by_length<A>(list: &[Vec<A>]) -> Vec<Vec<A>>
where
    A: Clone,
{
    let mut v = list.to_vec();
    v.sort_by(|x, y| x.len().cmp(&y.len()));
    v
}

#[cfg(test)]
mod test {
    use super::*;

    use factorial::Factorial;
    use proptest::prelude::*;
    use std::collections::HashSet;
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
        #[test]
        fn each_element_in_the_original_is_in_the_duplicates(list: Vec<isize>) {
            let res = duplicate(&list);
            prop_assert!(list.iter().all(|o| res.contains(&o)));
        }
    }

    proptest! {
        #[test]
        fn replicates_has_n_times_the_size_of_the_original(list: Vec<isize>, repl_num in 2..10u8) {
            prop_assert_eq!(list.len()*repl_num as usize, replicate(&list, repl_num).len());
        }
        #[test]
        fn each_element_in_the_original_is_in_the_replicates(list: Vec<isize>, repl_num in 2..10u8) {
            let res = replicate(&list, repl_num);
            prop_assert!(list.iter().all(|o| res.contains(&o)));
        }
    }

    proptest! {
        #[test]
        fn drop_every_nth_elem_of_the_list_removes_up_to_len_minus_len_div_n_elements(list: Vec<isize>, to_remove in 0..10usize) {
            let res_length = list.len() - list.len().checked_div_euclid(to_remove).unwrap_or_default();

            prop_assert_eq!(res_length, drop_every_nth(&list, to_remove).len());
        }
        // other properties: keeps the order, all elements in res is also in the original
    }

    #[test]
    fn drop_every_nth_test() {
        let list = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        let res = drop_every_nth(&list, 2);

        assert_eq!(vec![&0, &2, &4, &6, &8, &10], res);
    }

    proptest! {
        #[test]
        fn split_then_join_leads_to_the_original(list: Vec<isize>, at in 0..10usize) {
            let (part1, part2) = split_n(&list, at);

            prop_assert_eq!(list.as_slice(), [part1, part2].concat());
        }

        #[test]
        fn first_part_has_size_at_or_is_full_original(list: Vec<isize>, at in 0..10usize) {
            let (part1, _) = split_n(&list, at);

            prop_assert_eq!(std::cmp::min(at, list.len()), part1.len());
        }
    }

    #[test]
    fn slice_test() {
        let v = &[1, 2, 3, 4, 89, 143];
        let res = slice(v, 0, 5);
        assert_eq!(v, res.unwrap());

        let v = &[1, 2, 3, 4, 89, 143];
        let res = slice(v, 1, 4);
        assert_eq!(&[2, 3, 4, 89], res.unwrap());

        let v = &[1, 2, 3, 4, 89, 143];
        let res = slice(v, 4, 5);
        assert_eq!(&[89, 143], res.unwrap());
    }

    #[test]
    fn slice_out_of_bounds() {
        let v = &[1, 2, 3, 4, 89, 143];
        let res = slice(v, 0, 45);
        assert!(res.is_none());
    }

    #[test]
    fn slice_from_bigger_than_to() {
        let v = &[1, 2, 3, 4, 89, 143];
        let res = slice(v, 3, 2);
        assert!(res.is_none());
    }

    proptest! {
        #[test]
        fn rotate_keeps_the_length_invariant(list: Vec<char>, n in 0..10isize) {
            prop_assert_eq!(list.len(), rotate(&list, n).len());
        }
        #[test]
        fn rotate_keeps_the_same_elements(list: Vec<char>, n in 0..10isize) {
            prop_assert!(rotate(&list, n).iter().all(|e| list.contains(e)));
        }
        #[test]
        fn rotate_then_rotate_with_the_opposite_leaves_the_original(list: Vec<char>, n in 0..10isize) {
            let rotated = rotate(&list, n).to_vec();
            let re_rotated = rotate(&rotated, -n);
            prop_assert_eq!(list, re_rotated);
        }
    }

    proptest! {
        #[test]
        fn remove_at_is_some_list_with_one_less_elem_if_index_in_bounds_or_is_none(list: Vec<char>, n in 0..10usize) {
            let res = remove_at(&list, n);

            let condition = match res {
                        Some((e, v)) => {v.len() == list.len()-1 && list.contains(&e) && v.iter().all(|x| list.contains(x))},
                        None => n >= list.len()
                    };

            prop_assert!(condition);
        }

        // another prop: remove_at n followed by insert_at n is the original
    }

    proptest! {
        #[test]
        fn insert_at_increase_length_by_1_and_all_elems_are_in_list(list: Vec<char>, c: char, at in 0..10usize) {
            let res = insert_at(&list, c, at);
            prop_assert!(
                res.len() == list.len()+1 && res.contains(&c) && list.iter().all(|e| res.contains(e))
            );
        }
    }

    proptest! {
        #[test]
        fn range_has_to_minus_from_elements_all_between_from_and_to(from in 0..10000u32, to in 0..10000u32) {
            let res = range(from, to);
            let size = if from <= to { to - from + 1 } else { 0 } as usize;

            prop_assert_eq!(size, res.len());
            prop_assert!(res.iter().all(|e| *e >= from && *e <= to));
        }
    }

    proptest! {
        #[test]
        fn test_of_chosen_is_min_original_length_n(list: Vec<char>, n in 0..10000usize) {
            let res = random_select(&list, n, &mut rand::thread_rng());

            prop_assert_eq!(min(list.len(), n), res.len());
        }

        #[test]
        fn all_chosen_are_in_the_original_list(list: Vec<char>, n in 0..10000usize) {
            let res = random_select(&list, n, &mut rand::thread_rng());

            prop_assert!(res.iter().all(|e| list.contains(e)));
        }
    }

    proptest! {
        #[test]
        fn lottery_draw_result_has_size_equal_to_min_upper_bound_and_n(upper_bound in 0..10000u32, n in 0..10000u32) {
            prop_assume!(upper_bound != 0);

            let res = lottery_draw(upper_bound, n, &mut rand::thread_rng());

            prop_assert_eq!(min(n, upper_bound), res.len() as u32);
        }

        #[test]
        fn lottery_draw_every_number_is_between_1_and_upper_bound(upper_bound in 0..10000u32, n in 0..10000u32) {
            let res = lottery_draw(upper_bound, n, &mut rand::thread_rng());

            prop_assert!(res.iter().all(|e| *e >= 1 && *e <= upper_bound));
        }

    }

    proptest! {

        #[test]
        fn permutation_has_same_size_as_original(list: Vec<char>) {
            let res = random_permutation(&list, &mut rand::thread_rng());

            prop_assert_eq!(list.len(), res.len());
        }

        #[test]
        fn permutation_has_same_elements_as_original(list: Vec<char>) {
            let res = random_permutation(&list, &mut rand::thread_rng());

            prop_assert!(res.iter().all(|e| list.contains(e)) && list.iter().all(|e| res.contains(&e)));
        }
    }

    fn small_vector(lower_bound: usize, upper_bound: usize) -> impl Strategy<Value = Vec<char>> {
        proptest::collection::vec(proptest::char::any(), lower_bound..upper_bound)
    }

    proptest! {
        #[test]
        fn combinations_has_c_length_n_size(list in small_vector(3, 10), n in 1..5u16) {
            let expected_total = num_integer::binomial::<u16>(list.len() as u16, n) as usize;
            let res = combinations(&list, n as usize);

            prop_assert_eq!(expected_total, res.len());
        }

        #[test]
        fn combinations_have_elements_in_original(list in small_vector(3, 10), n in 1..5u16) {
            let res = combinations(&list, n as usize);

            prop_assert!(res.iter().flatten().all(|e| list.contains(e)));
        }
    }

    fn decompose_n(n: u8) -> Vec<Vec<u8>> {
        let tools = 1..n;
        let components = min(n / 3 + 1, 2);
        tools
            .combinations(components.into())
            .filter(|combo| combo.iter().sum::<u8>() == n)
            .collect::<Vec<Vec<u8>>>()
    }

    fn decompose<R>(n: u8, rng: &mut R) -> Vec<u8>
    where
        R: Rng,
    {
        let choices = decompose_n(n);
        let idx = rng.gen_range(0..choices.len());
        choices[idx].clone()
    }

    fn small_vector_no_dup_and_size(
        lower_bound: usize,
        upper_bound: usize,
    ) -> impl Strategy<Value = (Vec<char>, Vec<u8>)> {
        proptest::collection::hash_set(proptest::char::any(), lower_bound..upper_bound).prop_map(
            |v| {
                let sizes = decompose(v.len() as u8, &mut rand::thread_rng());
                (v.into_iter().collect::<Vec<char>>(), sizes)
            },
        )
    }

    proptest! {

        #[test]
        fn groups_has_multinomial_nb_of_solutions(list_and_sizes in small_vector_no_dup_and_size(5, 10)) {
            let (list, sizes) = list_and_sizes;
            let res = group(&sizes, &list);

            let expected_num_solutions = sizes.iter().map(|e| *e as usize).sum::<usize>().factorial() / sizes.iter().map(|e| (*e as usize).factorial()).product::<usize>();
            prop_assert_eq!(expected_num_solutions, res.unwrap().len());
        }

        #[test]
        fn joining_each_solution_gives_back_the_original(list_and_sizes in small_vector_no_dup_and_size(5, 10)) {
            let (list, sizes) = list_and_sizes;
            let res = group(&sizes, &list);

            let list_set = list.into_iter().collect::<HashSet<char>>();
            let property = res.unwrap().into_iter().all(|solution| {
                list_set == solution.into_iter().flatten().collect::<HashSet<char>>()
            });
            prop_assert!(property);
        }
    }

    proptest! {

        #[test]
        fn sort_by_length_has_same_elements_than_original(list: Vec<Vec<char>>) {
            let res = sort_by_length(&list);

            prop_assert!(res.iter().all(|e| list.contains(e)) && list.len() == res.len());
        }

        // TODO: add property: each elem has length >= previous

    }

    // TODO: extract to sort_by_length_freq
    // add tests; what kind of property can we have?
    // - same elements than original
    // - ???
    #[test]
    fn group_by() {
        use itertools::Itertools;
        use std::collections::BTreeMap;
        use std::iter::FromIterator;

        // input: list of lists

        // TODO: tri: nb occurrences sur la longueur de la liste: la longueur la moins courante en premier
        // => creer

        let v: Vec<&str> = vec!["alpha", "beta", "gamma", "delta", "omicron", "omega"];
        let lookup = v.into_iter().into_group_map_by(|e| e.len());
        let mut words = lookup.values().cloned().collect::<Vec<_>>();
        words.sort_by(|v0, v1| v0.len().cmp(&v1.len()));
        let r = words
            .iter_mut()
            .map(|v| {
                v.sort();
                v
            })
            .flatten()
            .collect::<Vec<_>>();

        println!("{:?}", r);
    }
}
