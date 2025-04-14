// TODO: proper docs

pub trait Reductor: Sized {
    type Item;
    type Output;

    // add a new reductor to the chain
    fn with_val<IR>(self, r: IR) -> impl Reductor<Item = Self::Item, Output = (Self::Output, IR::Output)>
    where
        IR: Reductor<Item = Self::Item>
    {
        CompositeReductor {
            r1: self,
            r2: r,
        }
    }

    // maybe better than with_val?
    fn with<IR>(self) -> impl Reductor<Item = Self::Item, Output = (Self::Output, IR::Output)>
    where
        IR: Reductor<Item = Self::Item> + Default
    {
        CompositeReductor {
            r1: self,
            r2: IR::default(),
        }
    }

    // (optional) use the `iter.size_hint()` to initialize any local state,
    // for optimization purposes
    fn apply_size_hint(&mut self, _hint: (usize, Option<usize>)) {}

    // pass the `iter.next()` value through the chain of reductors
    fn signal_next(&mut self, item: &Self::Item);

    // signal that we're done iterating, and return the output
    fn finish(self) -> Self::Output;
}

pub struct CompositeReductor<R1, R2> {
    r1: R1,
    r2: R2,
}

impl<RI, R1, R2> Reductor for CompositeReductor<R1, R2>
where
    R1: Reductor<Item = RI>,
    R2: Reductor<Item = RI>,
{
    type Item = R1::Item;
    type Output = (R1::Output, R2::Output);

    fn apply_size_hint(&mut self, hint: (usize, Option<usize>)) {
        self.r1.apply_size_hint(hint);
        self.r2.apply_size_hint(hint);
    }

    fn signal_next(&mut self, item: &Self::Item) {
        self.r1.signal_next(item);
        self.r2.signal_next(item);
    }

    fn finish(self) -> Self::Output {
        (self.r1.finish(), self.r2.finish())
    }
}

#[derive(Default)]
pub struct MinReductor {
    min: f64,
}

impl Reductor for MinReductor {
    type Item = f64;
    type Output = f64;

    fn signal_next(&mut self, item: &Self::Item) {
        if *item < self.min {
            self.min = *item;
        }
    }

    fn finish(self) -> Self::Output {
        self.min
    }
}

#[derive(Default)]
pub struct MaxReductor {
    max: f64,
}

impl Reductor for MaxReductor {
    type Item = f64;
    type Output = f64;

    fn signal_next(&mut self, item: &Self::Item) {
        if *item > self.max {
            self.max = *item;
        }
    }

    fn finish(self) -> Self::Output {
        self.max
    }
}

#[derive(Default)]
pub struct MeanReductor {
    count: f64,
    running_mean: f64,
}

impl Reductor for MeanReductor {
    type Item = f64;
    type Output = f64;

    fn signal_next(&mut self, item: &Self::Item) {
        self.count += 1.0;
        self.running_mean += (*item - self.running_mean) / self.count;
    }

    fn finish(self) -> Self::Output {
        if self.count > 0.0 {
            self.running_mean
        } else {
            f64::NAN
        }
    }
}

#[derive(Default)]
pub struct GeometricMeanReductor {
    count: f64,
    running_ln: f64,
}

impl Reductor for GeometricMeanReductor {
    type Item = f64;
    type Output = f64;

    fn signal_next(&mut self, item: &Self::Item) {
        self.count += 1.0;
        self.running_ln += (*item).ln();
    }

    fn finish(self) -> Self::Output {
        if self.count > 0.0 {
            (self.running_ln / self.count).exp()
        } else {
            f64::NAN
        }
    }
}

#[derive(Default)]
pub struct VarianceReductor {
    count: f64,
    sum: f64,
    variance: f64,
}

impl Reductor for VarianceReductor {
    type Item = f64;
    type Output = f64;

    fn signal_next(&mut self, item: &Self::Item) {
        if self.count == 0.0 {
            self.count = 1.0;
            self.sum = *item;
            return;
        }

        self.count += 1.0;
        self.sum += *item;

        let diff = self.count * item - self.sum;
        self.variance += diff * diff / (self.count * (self.count - 1.0));
    }

    fn finish(self) -> Self::Output {
        if self.count > 1.0 {
            self.variance / (self.count - 1.0)
        } else {
            f64::NAN
        }
    }
}

pub trait Reducible {
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
            r.signal_next(&x);
        }

        r.finish()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn basic_usage() {
        let data = vec![5.0, 2.5, 3.0, 10.0, 5.0];

        let reductor = MinReductor::default()
            .with::<MeanReductor>()
            .with::<GeometricMeanReductor>()
            .with::<VarianceReductor>();

        let result = (reductor, data.iter().cloned()).reduce();

        let (((min, mean), geomean), variance) = result;

        println!("(((min, mean), geomean), variance)");
        println!("((({min}, {mean}), {geomean}), {variance})");
    }
}
