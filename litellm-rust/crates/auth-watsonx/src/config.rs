use std::fmt;

use litellm_auth_types::SecretValue;

const IAM_ENDPOINT: &str = "https://iam.cloud.ibm.com/identity/token";

#[derive(Clone)]
pub struct WatsonxIamConfig {
    pub(crate) api_key: SecretValue,
    pub(crate) endpoint: String,
}

impl fmt::Debug for WatsonxIamConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WatsonxIamConfig")
            .field("api_key", &"[REDACTED]")
            .field("endpoint", &"[REDACTED]")
            .finish()
    }
}

impl WatsonxIamConfig {
    pub fn new(api_key: SecretValue) -> Self {
        Self::with_endpoint(api_key, IAM_ENDPOINT.into())
    }

    pub fn with_endpoint(api_key: SecretValue, endpoint: String) -> Self {
        Self { api_key, endpoint }
    }

    pub(crate) fn api_key(&self) -> &SecretValue {
        &self.api_key
    }

    pub(crate) fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_redacts_credentials_and_endpoint() {
        let config = WatsonxIamConfig::with_endpoint(
            SecretValue::new("private-key"),
            "https://private.example/token?secret=value".into(),
        );
        let debug = format!("{config:?}");
        assert!(!debug.contains("private-key"));
        assert!(!debug.contains("private.example"));
        assert!(!debug.contains("secret=value"));
    }
}
