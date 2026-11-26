//! The database reads of auth objects for `gateway-auth-core`'s store ports, shaped into what the
//! auth cache holds for them, one implementation for every host. `AGENTS.md` states the boundary.

mod budget_limit;
mod column;
mod error;
mod key;
mod member;
mod object_permission;
mod team;

pub use error::Error;
pub use key::KeyDatabase;
pub use team::{AliasMatch, TeamDatabase};
