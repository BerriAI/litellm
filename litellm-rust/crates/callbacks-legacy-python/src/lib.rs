//! The legacy `@client` wrapper as the native call sees it: litellm's `Logging` object, the
//! sync and async callback registries it fans out to, the deployment hooks and the deferred
//! proxy release. All of it sits behind one
//! [`PythonCallHooks`](litellm_host_python::PythonCallHooks), so the driver, the routes and
//! core never learn which Python object is on the other end. The SDK's own request policy
//! (credential inheritance, the budget and retry limits) is the driver's preflight, not this
//! crate's.
//!
//! Legacy callbacks receive the caller's own objects and may mutate them. [`PublicCall`]
//! is where those objects live, and [`run_legacy_call`] is how a route hands them over
//! without keeping a copy.

mod adapter;
mod call;
mod callbacks;
mod deferred;
mod logger;
mod mapping;
mod python;
pub(crate) use adapter::LegacyLogging;
pub use adapter::{LegacySurface, PassThroughStream};
pub use call::{PublicCall, run_legacy_call};
pub(crate) use callbacks::{LegacyCallbacks, is_internal_call};
pub(crate) use logger::{DeploymentHooks, PythonLogger, finalize, setup};
pub use mapping::{CallBoundary, CallbackMapping, Dispatch, callback_mappings};

#[cfg(test)]
mod test_support;
