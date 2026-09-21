use aws_credential_types::provider::{ProvideCredentials, error::CredentialsError, future};
use litellm_auth_aws::{AwsAuthConfig, resolve_credentials};

#[derive(Clone)]
pub(crate) struct Credentials {
    config: AwsAuthConfig,
}

impl Credentials {
    pub(crate) fn new(config: AwsAuthConfig) -> Self {
        Self { config }
    }
}

impl ProvideCredentials for Credentials {
    fn provide_credentials<'a>(&'a self) -> future::ProvideCredentials<'a>
    where
        Self: 'a,
    {
        future::ProvideCredentials::new(async {
            resolve_credentials(self.config.clone(), &|name| std::env::var(name).ok())
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
