//! Whether the proxy accepts a request body. One decision serves both hosts: the Python proxy
//! calls it through the bridge, and the gateway before it serves a request itself.
//!
//! A host reads the body into [`Params`], says what the route reads it as with [`Surface`], and
//! answers what the check asks about the proxy's own state through [`Host`]. [`check`] refuses
//! with a [`Rejection`] the caller sees.

mod check;
mod grant;
mod host;
mod names;
mod policy;
mod surface;
mod value;

pub use check::{Failure, Request, check};
pub use host::{Deployment, Deployments, Host, OptIn};
pub use names::{HostTables, Names, Reading, VERTEX_TARGET_PARAMS};
pub use policy::{Deferred, Policy};
pub use surface::Surface;
pub use value::{CallerValue, Params};

pub use crate::error::body::{Allowlist, Rejection, ServerOwnedKind};
