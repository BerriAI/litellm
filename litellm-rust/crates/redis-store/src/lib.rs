#![expect(
    dead_code,
    reason = "the store families built on these tiers and codecs land in the next PR"
)]
#![cfg_attr(
    not(test),
    expect(
        unused_imports,
        reason = "the store families built on these tiers and codecs land in the next PR"
    )
)]

mod codec;
mod dual_cache;
mod error;
mod failure_log;
mod memory;
mod number;
mod remote;
mod request;
mod value;

pub use codec::DecodeError;
pub use dual_cache::TierPolicy;
pub use error::{Error, ErrorKind};
pub use memory::{Clock, system_clock};
pub use number::Number;
pub use remote::Remote;
pub use request::{Pending, RequestRedis, Send};
pub use value::{FromValue, IntoValue};
