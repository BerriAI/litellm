//! How an insert that another writer can beat ended.

/// `Created` holds what this call wrote. `Conflict` holds what was already stored under the same unique key, or
/// `None` when that row no longer answers the read that follows the insert.
#[derive(Clone, Debug, PartialEq)]
pub enum Inserted<T> {
    Created(T),
    Conflict(Option<T>),
}
