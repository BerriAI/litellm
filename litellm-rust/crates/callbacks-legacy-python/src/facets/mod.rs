//! [`LegacyHooks`](crate::LegacyHooks) on each facet of the call, one file per facet, so the
//! stage-to-callback mapping is the file layout. Every facet file also lists the Python
//! callbacks its stages reach, which `mapping.rs` collects.

mod arguments;
mod outcome;
mod provider;
mod response;

pub(crate) use arguments::CALLBACKS as ARGUMENT_CALLBACKS;
pub(crate) use outcome::CALLBACKS as OUTCOME_CALLBACKS;
pub(crate) use provider::CALLBACKS as PROVIDER_CALLBACKS;
pub(crate) use response::CALLBACKS as RESPONSE_CALLBACKS;
