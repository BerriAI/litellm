//! The legacy callback contract: litellm's `Logging` object, the sync and async callback
//! registries it fans out to, the deployment hooks and the deferred proxy release. All of
//! it sits behind one [`CallbackAdapter`](litellm_host_python::CallbackAdapter), so the
//! driver, the routes and core never learn which Python object is on the other end.
//!
//! Legacy callbacks receive the caller's own objects and may mutate them. [`PublicCall`]
//! is where those objects live, and [`run_legacy_call`] is how a route hands them over
//! without keeping a copy.

mod adapter;
mod call;
mod callbacks;
mod deferred;
mod logger;
mod preparation;

pub use adapter::{LegacyLogging, LegacySurface};
pub use call::{PublicCall, lookup, run_legacy_call};
pub use callbacks::{LegacyCallbacks, is_internal_call};
pub use logger::{DeploymentHooks, PythonLogger, SetupResult, finalize, setup};
pub use preparation::prepare;
