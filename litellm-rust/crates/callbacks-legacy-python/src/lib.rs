//! The legacy `@client` wrapper as the native call sees it: litellm's `Logging` object, the
//! sync and async callback registries it fans out to, the deployment hooks and the deferred
//! proxy release. All of it sits behind one
//! [`PythonCallHooks`](litellm_host_python::PythonCallHooks), so the driver, the routes and
//! core never learn which Python object is on the other end.
//!
//! Legacy callbacks receive the caller's own objects and may mutate them. [`PublicCall`]
//! is where those objects live.

mod adapter;
mod call;
mod callbacks;
mod deferred;
mod logger;
mod mapping;
mod python;
pub use adapter::LegacyLogging;
pub use call::PublicCall;
pub(crate) use callbacks::{LegacyCallbacks, is_internal_call};
pub(crate) use logger::{DeploymentHooks, PythonLogger, finalize, setup};
pub use mapping::{CallBoundary, CallbackMapping, Dispatch, callback_mappings};

#[cfg(test)]
mod test_support;
