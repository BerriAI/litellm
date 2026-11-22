mod deadline;
mod endpoint;
mod error;
mod failure;
mod health;
mod observe;
mod pools;
mod statements;
#[cfg(test)]
mod test_support;

pub use deadline::within;
pub use endpoint::{ClientIdentity, Endpoint, PoolLimits, SslMode};
pub use error::{Error, ErrorKind};
pub use failure::DbFailure;
pub use health::health_check;
pub use observe::{Call, Observer, Outcome, QueryEvent, observe};
pub use pools::Pools;
pub use statements::Statements;
