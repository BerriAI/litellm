//! The legacy `@client` wrapper as the native call sees it: litellm's `Logging` object, the
//! sync and async callback registries it fans out to, the deployment hooks, the deferred
//! proxy release, and the kwargs rewrites the wrapper makes on the way in (credential-name
//! inheritance, budget and retry-count limits). All of it sits behind one
//! [`PythonLifecycle`](litellm_host_python::PythonLifecycle), so the driver, the routes and
//! core never learn which Python object is on the other end.
//!
//! Legacy callbacks receive the caller's own objects and may mutate them. [`PublicCall`]
//! is where those objects live, and [`run_legacy_call`] is how a route hands them over
//! without keeping a copy.

mod adapter;
mod call;
mod callbacks;
mod deferred;
mod legacy_python;
mod logger;
mod preparation;
#[cfg(test)]
#[path = "../tests/support.rs"]
mod test_support;

pub(crate) use adapter::LegacyLogging;
pub use adapter::LegacySurface;
pub use call::{PublicCall, run_legacy_call};
pub(crate) use callbacks::{LegacyCallbacks, is_internal_call};
pub(crate) use logger::{DeploymentHooks, PythonLogger, finalize, setup};
pub(crate) use preparation::prepare;
