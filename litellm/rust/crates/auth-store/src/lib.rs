//! The database reads of auth objects for `gateway-auth-core`'s store ports, shaped into what the
//! auth cache holds for them, one implementation for every host. `AGENTS.md` states the boundary.

mod budget;
mod budget_limit;
mod column;
mod error;
mod key;
mod member;
mod object_permission;
mod project;
mod team;
mod team_membership;

pub use error::Error;
pub use key::KeyDatabase;
pub use project::ProjectDatabase;
pub use team::{AliasMatch, TeamDatabase};
pub use team_membership::TeamMembershipDatabase;
