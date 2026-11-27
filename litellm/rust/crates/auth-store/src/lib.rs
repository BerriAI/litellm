//! The database reads and writes of auth objects for `gateway-auth-core`'s store ports, shaped
//! into what the auth cache holds for them, one implementation for every host. `AGENTS.md` states the boundary.

mod budget;
mod budget_limit;
mod column;
mod end_user;
mod error;
mod key;
mod member;
mod object_permission;
mod organization;
mod project;
mod team;
mod team_membership;
mod user;

pub use end_user::EndUserDatabase;
pub use error::Error;
pub use key::KeyDatabase;
pub use organization::OrganizationDatabase;
pub use project::ProjectDatabase;
pub use team::{AliasMatch, TeamDatabase};
pub use team_membership::{MemberAdded, MemberRemoved, TeamMembershipDatabase};
pub use user::{SignedIn, UserDatabase};
