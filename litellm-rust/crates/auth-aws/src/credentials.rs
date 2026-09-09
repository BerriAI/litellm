use std::time::SystemTime;

use aws_credential_types::Credentials as SdkCredentials;
use veil::Redact;

#[derive(Clone, Redact)]
pub struct Credentials(#[redact(fixed = 8)] pub(crate) SdkCredentials);

impl Credentials {
    pub fn new(
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        session_token: Option<String>,
        expires_after: Option<SystemTime>,
        provider_name: &'static str,
    ) -> Self {
        Self(SdkCredentials::new(
            access_key_id,
            secret_access_key,
            session_token,
            expires_after,
            provider_name,
        ))
    }

    pub fn access_key_id(&self) -> &str {
        self.0.access_key_id()
    }

    pub fn session_token(&self) -> Option<&str> {
        self.0.session_token()
    }
}

impl PartialEq for Credentials {
    fn eq(&self, other: &Self) -> bool {
        self.0.access_key_id() == other.0.access_key_id()
            && self.0.secret_access_key() == other.0.secret_access_key()
            && self.0.session_token() == other.0.session_token()
    }
}

impl Eq for Credentials {}

pub fn static_credentials(
    access_key_id: impl Into<String>,
    secret_access_key: impl Into<String>,
) -> Credentials {
    Credentials::new(
        access_key_id,
        secret_access_key,
        None,
        None,
        "litellm-static",
    )
}

pub fn session_credentials(
    access_key_id: impl Into<String>,
    secret_access_key: impl Into<String>,
    session_token: impl Into<String>,
    provider_name: &'static str,
) -> Credentials {
    Credentials::new(
        access_key_id,
        secret_access_key,
        Some(session_token.into()),
        None,
        provider_name,
    )
}

#[cfg(test)]
mod tests {
    use super::static_credentials;

    #[test]
    fn credentials_are_redacted() {
        let credentials = static_credentials("visible-id", "never-print-secret");
        let debug = format!("{credentials:?}");
        assert!(!debug.contains("visible-id"));
        assert!(!debug.contains("never-print-secret"));
    }
}
