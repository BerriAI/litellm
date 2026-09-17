#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CallbackId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallbackMethod {
    LogPreApiCall,
    LogPostApiCall,
    PreCallDeploymentHook,
    PostCallSuccessDeploymentHook,
    PostCallFailureDeploymentHook,
    LoggingHook,
    AsyncLoggingHook,
    LogSuccessEvent,
    AsyncLogSuccessEvent,
    LogFailureEvent,
    AsyncLogFailureEvent,
}

/// Language-neutral execution semantics for a callback invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delivery {
    Inline,
    Await,
    Worker,
    Background,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallbackInvocation {
    pub target: CallbackId,
    pub method: CallbackMethod,
    pub delivery: Delivery,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoggedMarker {
    SyncSuccess,
    AsyncSuccess,
    SyncFailure,
    AsyncFailure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvocationOutcome {
    Completed,
    Failed,
}
