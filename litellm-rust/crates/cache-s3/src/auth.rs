use aws_credential_types::{
    Credentials as AwsCredentials,
    provider::{ProvideCredentials, error::CredentialsError, future},
};
use litellm_auth_aws::{AwsAuthConfig, resolve_credentials};

#[derive(Clone)]
pub(crate) struct Credentials {
    config: AwsAuthConfig,
    env: fn(&str) -> Option<String>,
}

impl Credentials {
    pub(crate) fn new(config: AwsAuthConfig) -> Self {
        Self::with_env(config, |name| std::env::var(name).ok())
    }

    pub(crate) fn with_env(config: AwsAuthConfig, env: fn(&str) -> Option<String>) -> Self {
        Self { config, env }
    }
}

impl ProvideCredentials for Credentials {
    fn provide_credentials<'a>(&'a self) -> future::ProvideCredentials<'a>
    where
        Self: 'a,
    {
        future::ProvideCredentials::new(async {
            if let (Some(access_key_id), Some(secret_access_key)) = (
                self.config.access_key_id.clone(),
                self.config.secret_access_key.clone(),
            ) {
                return Ok(AwsCredentials::new(
                    access_key_id,
                    secret_access_key,
                    self.config.session_token.clone(),
                    None,
                    "litellm-s3-cache",
                ));
            }
            resolve_credentials(self.config.clone(), &self.env)
                .await
                .map_err(|_| CredentialsError::provider_error("S3 cache authentication failed"))
        })
    }
}

impl std::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credentials").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn explicit_keys_ignore_an_ambient_session_token() {
        let provider = Credentials::with_env(
            AwsAuthConfig {
                access_key_id: Some("key".to_string()),
                secret_access_key: Some("secret".to_string()),
                region_name: Some("us-east-1".to_string()),
                ..Default::default()
            },
            |name| (name == "AWS_SESSION_TOKEN").then(|| "ambient".to_string()),
        );
        let credentials = provider.provide_credentials().await.unwrap();
        assert_eq!(credentials.access_key_id(), "key");
        assert_eq!(credentials.secret_access_key(), "secret");
        assert_eq!(credentials.session_token(), None);
    }

    #[tokio::test]
    async fn explicit_keys_keep_their_session_token() {
        let provider = Credentials::new(AwsAuthConfig {
            access_key_id: Some("key".to_string()),
            secret_access_key: Some("secret".to_string()),
            session_token: Some("t".to_string()),
            region_name: Some("us-east-1".to_string()),
            ..Default::default()
        });
        let credentials = provider.provide_credentials().await.unwrap();
        assert_eq!(credentials.session_token(), Some("t"));
    }

    #[tokio::test]
    async fn environment_keys_resolve_with_their_session_token() {
        let provider = Credentials::with_env(AwsAuthConfig::default(), |name| match name {
            "AWS_ACCESS_KEY_ID" => Some("env-key".to_string()),
            "AWS_SECRET_ACCESS_KEY" => Some("env-secret".to_string()),
            "AWS_SESSION_TOKEN" => Some("env-token".to_string()),
            _ => None,
        });
        let credentials = provider.provide_credentials().await.unwrap();
        assert_eq!(credentials.access_key_id(), "env-key");
        assert_eq!(credentials.secret_access_key(), "env-secret");
        assert_eq!(credentials.session_token(), Some("env-token"));
    }
}
