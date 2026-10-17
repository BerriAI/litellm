//! Hooks and observers for the gateway.
//!
//! A hook runs on the request path. It declares the facts it reads and the parts of the request
//! it may change, and runs in config order with the others. An observer sees each request after
//! its response ends, off the request path, and declares the facts it reads. The gateway builds
//! only the facts something reads, so with nothing configured a request pays nothing.

mod facts;
mod hook;
mod observer;
mod pipeline;
mod pricing;
mod usage;

pub use facts::{
    Annotations, DeploymentFacts, Facts, KeyFacts, Outcome, Record, RequestFacts, Route,
};
pub use hook::{HeaderPattern, Hook, Refusal, RequestEdit, Settle, Targets};
pub use observer::{DEFAULT_QUEUE, Observer, Observers};
pub use pipeline::{Admission, Pipeline};
pub use pricing::{Prices, deployment_fields};
pub use usage::{ServiceTier, StreamFormat, StreamUsage, Usage};
