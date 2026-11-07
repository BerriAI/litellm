use std::sync::Arc;

use aws_credential_types::Credentials;
use aws_credential_types::provider::{ProvideCredentials, error::CredentialsError, future};

use crate::{AwsAuthConfig, AwsAuthService};

type Environment = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// AWS credentials for an SDK client. The host picks where they come from and supplies the environment they read
#[derive(Clone)]
pub struct AwsCredentialsProvider {
    auth: AwsAuthService,
    source: Source,
}

#[derive(Clone)]
enum Source {
    DefaultChain {
        region: String,
    },
    Configured {
        config: Box<AwsAuthConfig>,
        environment: Environment,
    },
}

impl AwsCredentialsProvider {
    /// What botocore's default chain finds in `region`, without LiteLLM's Bedrock settings
    pub fn default_chain(auth: AwsAuthService, region: String) -> Self {
        Self {
            auth,
            source: Source::DefaultChain { region },
        }
    }

    /// Explicit keys in `config` as given, otherwise LiteLLM's `AWS_*` settings read through `environment`
    pub fn configured(
        auth: AwsAuthService,
        config: AwsAuthConfig,
        environment: impl Fn(&str) -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            auth,
            source: Source::Configured {
                config: Box::new(config),
                environment: Arc::new(environment),
            },
        }
    }

    async fn credentials(&self) -> Result<Credentials, crate::Error> {
        match &self.source {
            Source::DefaultChain { region } => self.auth.default_chain_credentials(region).await,
            Source::Configured {
                config,
                environment,
            } => match (&config.access_key_id, &config.secret_access_key) {
                (Some(access_key_id), Some(secret_access_key)) => Ok(Credentials::new(
                    access_key_id.expose(),
                    secret_access_key.expose(),
                    config
                        .session_token
                        .as_ref()
                        .map(|token| token.expose().to_owned()),
                    None,
                    "litellm-configured",
                )),
                _ => {
                    self.auth
                        .resolve_credentials(AwsAuthConfig::clone(config), environment.as_ref())
                        .await
                }
            },
        }
    }
}

impl ProvideCredentials for AwsCredentialsProvider {
    fn provide_credentials<'a>(&'a self) -> future::ProvideCredentials<'a>
    where
        Self: 'a,
    {
        future::ProvideCredentials::new(async {
            self.credentials()
                .await
                .map_err(CredentialsError::provider_error)
        })
    }
}

impl std::fmt::Debug for AwsCredentialsProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AwsCredentialsProvider")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn explicit(session_token: Option<&str>) -> AwsAuthConfig {
        AwsAuthConfig {
            access_key_id: Some("key".into()),
            secret_access_key: Some("secret".into()),
            session_token: session_token.map(Into::into),
            region_name: Some("us-east-1".to_string()),
            ..Default::default()
        }
    }

    fn ambient_token(name: &str) -> Option<String> {
        (name == "AWS_SESSION_TOKEN").then(|| "ambient".to_string())
    }

    fn environment_keys(name: &str) -> Option<String> {
        match name {
            "AWS_ACCESS_KEY_ID" => Some("env-key".to_string()),
            "AWS_SECRET_ACCESS_KEY" => Some("env-secret".to_string()),
            "AWS_SESSION_TOKEN" => Some("env-token".to_string()),
            _ => None,
        }
    }

    #[rstest]
    #[case::explicit_keys_ignore_an_ambient_session_token(
        explicit(None), ambient_token, ("key", "secret", None)
    )]
    #[case::explicit_keys_keep_their_session_token(
        explicit(Some("t")), ambient_token, ("key", "secret", Some("t"))
    )]
    #[case::environment_keys_resolve_with_their_session_token(
        AwsAuthConfig::default(), environment_keys, ("env-key", "env-secret", Some("env-token"))
    )]
    #[tokio::test]
    async fn configured_credentials_read_the_environment_they_are_given(
        #[case] config: AwsAuthConfig,
        #[case] environment: fn(&str) -> Option<String>,
        #[case] expected: (&str, &str, Option<&str>),
    ) {
        let credentials =
            AwsCredentialsProvider::configured(AwsAuthService::default(), config, environment)
                .provide_credentials()
                .await
                .unwrap();
        assert_eq!(
            (
                credentials.access_key_id(),
                credentials.secret_access_key(),
                credentials.session_token(),
            ),
            expected
        );
    }
}
