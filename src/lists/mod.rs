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

#[derive(Clone)]
enum NestedList<A> {
    Elem(A),
    List(Vec<Box<NestedList<A>>>),
}

fn flatten<A>(list: &NestedList<A>) -> Vec<&A> {
    match list {
        NestedList::Elem(e) => vec![e],
        NestedList::List(_) => vec![],
    }
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
                IndexRange::InBounds(n) => n..len+n,
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

    // TODO: implement a strategy for NestedList

    #[test]
    fn flatten_an_elem_is_a_vector_with_the_elem() {
        let e = Elem(3);

        let result = flatten(&e);

        assert_eq!(result, vec![&3]);
    }
}
