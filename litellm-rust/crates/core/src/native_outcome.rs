#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decline {
    reason: &'static str,
}

impl Decline {
    pub const fn new(reason: &'static str) -> Self {
        Self { reason }
    }

    pub const fn reason(self) -> &'static str {
        self.reason
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeOutcome<T> {
    Completed(T),
    Declined(Decline),
}

impl<T> NativeOutcome<T> {
    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> NativeOutcome<U> {
        match self {
            Self::Completed(value) => NativeOutcome::Completed(map(value)),
            Self::Declined(decline) => NativeOutcome::Declined(decline),
        }
    }
}
