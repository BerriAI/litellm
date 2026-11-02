use std::time::Duration;

use crate::Error;

pub trait RedisEvents: Send + Sync {
    fn call(&self, event: &CallEvent);
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
