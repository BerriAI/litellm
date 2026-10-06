#[derive(Clone, Debug, Default)]
pub struct CacheScope {
    pub credential: Option<CacheCredential>,
    pub model_group: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheCredential(String);

impl CacheCredential {
    pub fn new(authority: &str, subject: &str, credential_id: &str) -> Self {
        Self(serde_json::json!([authority, subject, credential_id]).to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}
