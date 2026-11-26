//! A roster entry as the auth cache holds it, nested in a cached key.

use litellm_python_compat::{Value, json};
use litellm_redis_store::{DecodeError, FromValue, auth_cache::MemberEntry};
use litellm_repositories::RawJson;

use crate::{Error, error::within};

/// [`member`] for the key's `team_member` column.
pub(crate) fn team_member_column(raw: &RawJson) -> Result<MemberEntry, Error> {
    json::loads(raw.get())
        .map_err(|_| Error::row("team_member"))
        .and_then(|entry| member("team_member", &entry))
}

/// The roster entry under `field` as `Member` dumps it: its three fields, whatever else the entry
/// holds. A refused field is named under `field`, and an entry the model refuses as a whole, one
/// that is no object or names neither a user id nor an email, is `field` itself.
pub(crate) fn member(field: &'static str, entry: &Value) -> Result<MemberEntry, Error> {
    match MemberEntry::from_value(entry) {
        Ok(member) => Ok(MemberEntry {
            unread: Vec::new(),
            ..member
        }),
        Err(DecodeError::Shape(column @ ("user_id" | "user_email" | "role"))) => {
            Err(within(field)(Error::row(column)))
        }
        Err(_) => Err(Error::row(field)),
    }
}
