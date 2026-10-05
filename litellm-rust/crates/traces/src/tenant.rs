/// Who sent a batch of spans. Always taken from the caller's authentication, never from span
/// attributes.
#[macro_rules_attribute::apply(request_type)]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Tenant {
    pub team_id: String,
    pub api_key_hash: String,
    #[serde(default)]
    pub org_id: String,
    #[serde(default)]
    pub user_id: String,
}
