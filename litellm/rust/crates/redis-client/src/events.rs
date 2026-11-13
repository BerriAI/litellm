use std::{any::Any, future::Future, sync::Arc, time::Duration};

use litellm_circuit_breaker::{State, Transition};

use crate::Error;

/// What the host attached to the calls made inside `tagged`, opaque to the client.
pub type CallTag = Arc<dyn Any + Send + Sync>;

tokio::task_local! {
    static TAG: CallTag;
}

pub async fn tagged<F: Future>(tag: CallTag, operation: F) -> F::Output {
    TAG.scope(tag, operation).await
}

/// The tag of the `tagged` scope this task runs in. `RedisEvents::call` runs on the calling task,
/// so a sink reads its call's tag here.
pub fn current_tag() -> Option<CallTag> {
    TAG.try_with(Arc::clone).ok()
}

pub trait RedisEvents: Send + Sync {
    fn call(&self, event: &CallEvent);

    fn breaker(&self, transition: &Transition<State>);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Command(&'static str),
    Script,
    Pipeline { ops: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Succeeded,
    Failed(FailureKind),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureKind {
    Timeout,
    Transport,
    Server,
    Decode,
    BreakerOpen,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallEvent {
    pub operation: Operation,
    pub duration: Duration,
    pub outcome: Outcome,
}

impl FailureKind {
    pub fn of(error: &Error) -> Self {
        match error {
            Error::Transport(error) if error.is_timeout() => Self::Timeout,
            Error::Transport(_) => Self::Transport,
            Error::Server(_) => Self::Server,
            Error::Decode(_) => Self::Decode,
            Error::BreakerOpen => Self::BreakerOpen,
        }
    }
}

impl Outcome {
    pub(crate) fn of<T>(result: &Result<T, Error>) -> Self {
        match result {
            Ok(_) => Self::Succeeded,
            Err(error) => Self::Failed(FailureKind::of(error)),
        }
    }
}
