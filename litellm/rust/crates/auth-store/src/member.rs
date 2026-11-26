//! A roster entry as the auth cache holds it, nested in a cached key.

use litellm_redis_store::auth_cache::CachedMember;
use litellm_repositories::RawJson;

use crate::{Error, column::column};

/// The roster entry as `Member` dumps it: its three fields, whatever else the entry holds. The
/// model refuses an entry that names neither a user id nor an email.
pub(crate) fn member(entry: &RawJson) -> Result<CachedMember, Error> {
    let member: CachedMember = column("team_member", entry)?;
    if member.user_id.is_none() && member.user_email.is_none() {
        return Err(Error::Row {
            column: "team_member",
        });
    }
    Ok(CachedMember {
        unread: Vec::new(),
        ..member
    })
}
