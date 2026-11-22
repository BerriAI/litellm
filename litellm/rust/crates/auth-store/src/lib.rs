//! The database reads of auth objects for `gateway-auth-core`'s store ports, shaped into what the
//! auth cache holds for them, one implementation for every host. `AGENTS.md` states the boundary.

mod error;
mod key;
mod object_permission;

pub use error::Error;
pub use key::KeyDatabase;
