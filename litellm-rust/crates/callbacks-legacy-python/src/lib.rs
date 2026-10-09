//! The legacy `@client` wrapper as the native call sees it: litellm's `Logging` object, the
//! sync and async callback registries it fans out to, the deployment hooks, the deferred
//! proxy release and the Python handler body the route would have run. All of it sits
//! behind [`hooks`], so the driver, the routes and core never learn which Python object is
//! on the other end.
//!
//! Legacy callbacks receive the caller's own objects and may mutate them. [`PublicCall`]
//! is where those objects live.

mod adapter;
mod call;
mod callbacks;
mod deferred;
mod logger;
mod mapping;
mod messages_body;
mod operation;
mod python;
use litellm_host::call::Operation;
use litellm_host_python::{HookLayer, Hooks};
use pyo3::Python;

pub(crate) use adapter::LegacyLogging;
pub use call::PublicCall;
pub(crate) use callbacks::{LegacyCallbacks, is_internal_call};
pub(crate) use logger::{DeploymentHooks, PythonLogger, finalize, setup};
pub use mapping::{CallbackMapping, Dispatch, Operations, callback_mappings};
use messages_body::body_hooks;

/// The legacy hooks a native call runs, outermost first: the `@client` wrapper's `Logging`,
/// then the Python handler body the route would have run after it.
pub struct LegacyLayer {
    operation: Operation,
    asynchronous: bool,
}

impl LegacyLayer {
    pub fn new(operation: Operation, asynchronous: bool) -> Self {
        Self {
            operation,
            asynchronous,
        }
    }
}

impl HookLayer<PublicCall> for LegacyLayer {
    fn layer(&self, py: Python<'_>, call: PublicCall) -> impl IntoIterator<Item = Hooks> {
        let body = body_hooks(py, self.operation, &call, self.asynchronous).map(Hooks::new);
        std::iter::once(Hooks::new(LegacyLogging::new(
            py,
            self.operation,
            call,
            self.asynchronous,
        )))
        .chain(body)
    }
}

#[cfg(test)]
mod test_support;
