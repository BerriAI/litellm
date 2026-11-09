/// A `resume` the coroutine refused, leaving it as it was.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ResumeError {
    #[error("coroutine resumed after it finished")]
    Finished,
    #[error("coroutine resumed before the reply to its last yield was sent or dropped")]
    Unanswered,
}

/// No answer will come to a yield: its [`Reply`](crate::Reply) was dropped unsent, or the
/// coroutine it was sent to is gone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the yield was abandoned before it was answered")]
pub struct Abandoned;
