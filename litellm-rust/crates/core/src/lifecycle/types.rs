use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ActionKind {
    RequestBuild,
    RequestPolicy,
    ProviderCall,
    Deployment,
    TerminalSuccess,
    TerminalFailure,
    Restore,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActionResult<T, E> {
    Continue(T),
    Replace(T),
    Reject(E),
}

impl<T, E> ActionResult<T, E> {
    pub fn into_result(self) -> Result<T, E> {
        match self {
            Self::Continue(value) | Self::Replace(value) => Ok(value),
            Self::Reject(error) => Err(error),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum FailurePolicy {
    Propagate,
    RecordAndContinue,
    PreserveOriginalFailure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Delivery {
    InlineDirect,
    InlineAwaited,
    BlockingWorker,
    BackgroundTask,
    Deferred,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Outcome {
    Success,
    Failure,
    Abort,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ErrorDisposition {
    Preserve,
    Replace,
}
