//! The legacy `@client` wrapper as the native call sees it: litellm's `Logging` object, the
//! sync and async callback registries it fans out to, the deployment hooks and the deferred
//! proxy release. All of it sits behind one
//! [`PythonCallHooks`](litellm_host_python::PythonCallHooks), so the driver, the routes and
//! core never learn which Python object is on the other end.
//!
//! Legacy callbacks receive the caller's own objects and may mutate them. [`PublicCall`]
//! is where those objects live.

mod adapter;
mod body;
mod call;
mod callbacks;
mod deferred;
mod logger;
mod mapping;
mod python;
pub use adapter::LegacyLogging;
pub use body::HandlerBody;
pub use call::PublicCall;
pub(crate) use callbacks::{LegacyCallbacks, is_internal_call};
use litellm_host_python::HookChain;
pub(crate) use logger::{DeploymentHooks, PythonLogger, finalize, setup};
pub use mapping::{CallBoundary, CallbackMapping, Dispatch, Operations, callback_mappings};
use pyo3::Python;
use strum::IntoStaticStr;

#[derive(Clone, Copy, Debug, IntoStaticStr, PartialEq, Eq)]
pub enum LoggingOperation {
    #[strum(serialize = "completion")]
    Completion,
    #[strum(serialize = "responses")]
    Responses,
    #[strum(serialize = "messages")]
    Messages,
    #[strum(serialize = "ocr")]
    Ocr,
}

/// The legacy contract for one native call as layers: `Logging` for every route, and inside
/// it the body hooks the operation's async Python handler fires, so the body transforms the
/// response before the wrapper's own success hook sees it.
pub fn legacy_hooks(
    py: Python<'_>,
    operation: LoggingOperation,
    call: PublicCall,
    asynchronous: bool,
) -> HookChain {
    let body = HandlerBody::new(py, operation, &call, asynchronous);
    let chain = HookChain::new().with(LegacyLogging::new(py, operation, call, asynchronous));
    match body {
        Some(body) => chain.with(body),
        None => chain,
    }
}

#[cfg(test)]
mod test_support;
