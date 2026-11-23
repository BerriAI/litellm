//! Callers for the inline tests, over rows that hold a key, which the native host's do not yet.

use litellm_gateway_auth_core::{
    Caller, KeyHash, Rows,
    authn::key::KeyRecord,
    identity::key::{KeyContext, TeamGrant},
};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Held;

impl Rows for Held {
    type Key = KeyRecord;
    type User = ();
    type Team = ();
    type EndUser = ();
    type ObjectPermission = ();
    type Project = ();
    type Membership = ();
    type Org = ();
    type Agent = ();
}

/// A key whose row is stored under `stored` and says `is_proxy_admin`.
pub(crate) fn key(stored: &str, is_proxy_admin: bool) -> Caller<Held> {
    held(KeyRecord {
        token: Some(KeyHash::new(stored)),
        is_proxy_admin,
        ..KeyRecord::default()
    })
}

/// A key whose row is `record`.
pub(crate) fn held(record: KeyRecord) -> Caller<Held> {
    Caller::Key(Box::new(KeyContext {
        key: record,
        end_user: (),
        owner: None,
        team: TeamGrant::NoTeam,
        project: None,
    }))
}
