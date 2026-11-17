//! Rust repositories for the auth objects the proxy reads from Postgres when its auth cache misses.
//!
//! Each repository is a folder that owns its queries, the shaping of their rows into what the auth flow consumes,
//! and any memory tier private to that object. Queries are private to their folder, so every read goes through a
//! repository and its `Database::run`. The shared auth cache in front of them stays with the host. See README.md
//! for the layout.

macro_rules_attribute::attribute_alias! {
    #[apply(db_row)] =
        #[derive(Clone, Debug, PartialEq, sqlx::FromRow)];
}

mod budget;
mod database;
mod end_user;
mod jwt_key_mapping;
mod key;
mod model_access_group;
mod raw_json;
#[cfg(test)]
mod test_support;
mod user;
mod utc_timestamp;

pub use budget::{Budget, BudgetRepository};
pub use database::{Database, Side};
pub use end_user::{EndUserRecord, EndUserRepository};
pub use jwt_key_mapping::{JwtKeyMappingRepository, MappedClaim};
pub use key::{FetchedKey, KeyRepository};
pub use model_access_group::{ModelAccessGroupBudget, ModelAccessGroupRepository};
use raw_json::RawJson;
pub use user::{SignIn, User, UserRepository};
