// TODO: proper docs

trait Reductor {
    type Item;
    type Output;

    // add a new reductor to the chain
    fn with<O, IR, OR>(r: IR) -> OR
    where
        IR: Reductor<Item = Self::Item, Output = O>,
        OR: Reductor<Item = Self::Item, Output = (Self::Output, O)>;

    // (optional?) use the `iter.size_hint()` to initialize any local state,
    // for optimization purposes
    fn apply_size_hint(&mut self, hint: (usize, Option<usize>));

    // pass the `iter.next()` value through the chain of reductors
    fn signal_next(&mut self, item: Self::Item);

    // signal that we're done iterating, and return the output
    fn finish(self) -> Self::Output;
}

trait Reducible {
    type Output;

    fn reduce(self) -> Self::Output;
}

impl<T, R, I> Reducible for (R, I)
where
    R: Reductor<Item = T>,
    I: Iterator<Item = T>,
{
    type Output = <R as Reductor>::Output;

    fn reduce(self) -> Self::Output {
        let (mut r, mut i) = self;
        
        r.apply_size_hint(i.size_hint());
        while let Some(x) = i.next() {
            r.signal_next(x);
        }

        r.finish()
    }
}
