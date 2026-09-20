//! The legacy `@client` wrapper as the native call sees it: litellm's `Logging` object, the
//! sync and async callback registries it fans out to, the deployment hooks, the deferred
//! proxy release, and the kwargs rewrites the wrapper makes on the way in (credential-name
//! inheritance, budget and retry-count limits). All of it sits behind one
//! [`PythonLifecycle`](litellm_host_python::PythonLifecycle), so the driver, the routes and
//! core never learn which Python object is on the other end.
//!
//! Legacy callbacks receive the caller's own objects and may mutate them. [`PublicCall`]
//! is where those objects live; the bridge hands it to [`LegacyPythonLifecycle::new`] and
//! keeps no copy.

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

pub use adapter::{LegacyPythonLifecycle, LegacyPythonSurface, PassThroughStream};
pub use call::PublicCall;
pub(crate) use callbacks::is_internal_call;
pub(crate) use logger::{
    PythonLogger, after_deployment_failure, after_deployment_success, before_deployment_call,
    finalize, setup,
};
pub(crate) use preparation::prepare;
