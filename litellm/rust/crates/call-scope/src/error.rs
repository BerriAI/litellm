/// Why the code asking has no call of the type it asked for.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The code runs outside any call: a spawn that skipped [`crate::spawn_in_call`], or work that
    /// was never part of a call.
    #[error("no call in scope")]
    Outside,
    /// The call in scope holds something other than what was asked for.
    #[error("the call in scope does not hold a {expected}")]
    OtherKind { expected: &'static str },
}
