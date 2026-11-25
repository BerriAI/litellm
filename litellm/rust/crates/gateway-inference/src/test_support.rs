//! Callers for the inline tests, over rows that hold a key, which the native host's do not yet.

use litellm_gateway_auth_core::{
    Belongs, Caller, Credential, KeyHash, Rows, authn::key::KeyRecord,
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
    Caller {
        credential: Credential::Key {
            key: KeyRecord {
                token: Some(KeyHash::new(stored)),
                is_proxy_admin,
                ..KeyRecord::default()
            },
        },
        belongs: Belongs::nobody(),
        end_user: (),
    }
}
