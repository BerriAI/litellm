/// A function from one stack element to the next, in the tower sense. The input has no
/// bound on purpose: the innermost element is usually a factory of attempt machines, not a
/// machine, and a layer may wrap either.
pub trait Layer<T> {
    type Output;

    fn layer(&self, inner: T) -> Self::Output;
}

/// Builds a stack inside out. The last layer added is the outermost, so a call site reads
/// top down as the order ops flow up:
///
/// ```ignore
/// Stack::new(attempt_factory)
///     .layer(RouterLayer::new(plan, picker, clock, seed))
///     .build()
/// ```
pub struct Stack<T>(T);

impl<T> Stack<T> {
    pub fn new(inner: T) -> Self {
        Self(inner)
    }

    pub fn layer<L: Layer<T>>(self, layer: L) -> Stack<L::Output> {
        Stack(layer.layer(self.0))
    }

    pub fn build(self) -> T
    where
        T: crate::machine::Machine,
    {
        self.0
    }
}

/// Wraps nothing; useful for optional layers.
#[derive(Clone, Copy, Debug, Default)]
pub struct Identity;

impl<T> Layer<T> for Identity {
    type Output = T;

    fn layer(&self, inner: T) -> T {
        inner
    }
}

/// `Some(layer)` applies it, `None` is [`Identity`].
impl<T, L: Layer<T, Output = T>> Layer<T> for Option<L> {
    type Output = T;

    fn layer(&self, inner: T) -> T {
        match self {
            Some(layer) => layer.layer(inner),
            None => inner,
        }
    }
}
