//! Fixtures for tests: key rows any crate's tests can hold in memory, and the database fixtures this crate's
//! own tests share.

#[cfg(test)]
mod database;
mod keys;

#[cfg(test)]
pub(crate) use database::{OneTransaction, database_pool, insert_budget, leave_utc};
pub use keys::{HeldKeys, PLAIN_KEY_JSON, plain_key};
