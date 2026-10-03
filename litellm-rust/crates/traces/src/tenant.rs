use serde::Deserialize;

/// Who sent a batch of spans. Always taken from the caller's authentication, never from span
/// attributes.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq)]
pub struct Tenant {
    pub team_id: String,
    pub api_key_hash: String,
    pub org_id: String,
    pub user_id: String,
}
