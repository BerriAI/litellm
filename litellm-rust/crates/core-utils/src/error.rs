#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("expected {expected}")]
pub struct Invalid {
    expected: &'static str,
}

impl Invalid {
    pub fn expected(expected: &'static str) -> Self {
        Self { expected }
    }
}
