//! Callers for the inline tests, over rows that hold a key, which the native host's do not yet.

use litellm_gateway_auth_core::{
    Caller, KeyHash, Rows,
    authn::key::{KeyRecord, Keyed},
    identity::key::{KeyContext, TeamGrant},
};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Held;

impl Rows for Held {
    type Key = ();
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
    let record = KeyRecord {
        is_proxy_admin,
        ..KeyRecord::default()
    };
    Caller::Key(Box::new(KeyContext {
        hash: KeyHash::new(stored),
        key: Keyed { record, row: () },
        end_user: (),
        owner: None,
        team: TeamGrant::NoTeam,
        project: None,
    }))
}
