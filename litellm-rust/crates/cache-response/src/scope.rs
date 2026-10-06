#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CacheScope {
    Shared,
    Caller(String),
}

impl CacheScope {
    pub fn caller(authority: &str, subject: &str, credential_id: &str) -> Self {
        Self::Caller(serde_json::json!([authority, subject, credential_id]).to_string())
    }
}
