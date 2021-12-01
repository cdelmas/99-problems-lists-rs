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

fn length<A>(l: &Vec<A>) -> usize {
    l.len()
}

fn reverse<A: Clone>(l: &Vec<A>) -> Vec<A> {
    let mut copy = l.clone();
    copy.reverse();
    copy
}

fn is_palindrome<A: Eq>(l: &Vec<A>) -> bool {
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

fn remove_consecutive_duplicates<A: Eq>(list: &Vec<A>) -> Vec<&A> {
    let (_, res) = list
        .iter()
        .fold((None, vec![]), |(prev, mut acc), e| match prev {
            Some(last) if last == e => (prev, acc),
            _ => {
                acc.push(e);
                (Some(e), acc)
            }
        });
    res
}

fn pack_consecutive_duplicates<A: Eq>(list: &Vec<A>) -> Vec<Vec<&A>> {
    let (_, res) = list.iter().fold(
        (vec![], vec![]) as (Vec<&A>, Vec<Vec<&A>>),
        |(mut prev, mut acc), e| {
            if prev.contains(&e) {
                prev.push(e);
                (prev, acc)
            } else {
                acc.push(prev);
                (vec![e], acc)
            }
        },
    );
    res
}

#[cfg(test)]
mod test {
    use super::*;

    use proptest::prelude::*;
    use NestedList::*;

    #[test]
    fn it_does_pass() {
        assert_eq!(1, 1);
    }

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
        assert_eq!(is_palindrome(&vec![1, 2, 4, 2, 1]), true);
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

        assert_eq!(result, vec![&3]);
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
            prop_assert_eq!(flatten(&list).len(), leaf_number);
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

        assert_eq!(res, vec![&4, &2, &6, &17]);
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
}
