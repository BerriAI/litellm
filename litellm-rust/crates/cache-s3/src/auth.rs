use aws_credential_types::{
    Credentials as AwsCredentials,
    provider::{ProvideCredentials, error::CredentialsError, future},
};
use litellm_auth_aws::{AwsAuthConfig, AwsAuthService};

#[derive(Clone)]
pub struct S3Credentials {
    auth: AwsAuthService,
    config: AwsAuthConfig,
    env: fn(&str) -> Option<String>,
}

impl S3Credentials {
    pub fn new(config: AwsAuthConfig) -> Self {
        Self::with_env(config, |name| std::env::var(name).ok())
    }

    pub fn with_env(config: AwsAuthConfig, env: fn(&str) -> Option<String>) -> Self {
        Self {
            auth: AwsAuthService::default(),
            config,
            env,
        }
    }
}

impl ProvideCredentials for S3Credentials {
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
                    access_key_id.expose(),
                    secret_access_key.expose(),
                    self.config
                        .session_token
                        .as_ref()
                        .map(|value| value.expose().to_owned()),
                    None,
                    "litellm-s3-cache",
                ));
            }
            self.auth
                .resolve_credentials(self.config.clone(), &self.env)
                .await
                .map_err(CredentialsError::provider_error)
        })
    }
}

impl std::fmt::Debug for S3Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3Credentials").finish_non_exhaustive()
    }
}
