use std::sync::Arc;

use litellm_auth::AuthServices;
use litellm_auth_aws::{AwsAuthService, WebIdentityTokenFuture, WebIdentityTokenSource};
use litellm_auth_gcp::vertex::VertexLoginService;
use litellm_config::Config;
use litellm_config_load::{ConfigEnvironment, environment_values};
use litellm_env::{Lookup, ProcessEnvironment};
use litellm_http::{
    ClientVariant, HttpClientConfig, HttpClientPool, HttpSettings, HttpSettingsLayer, Resolution,
    SslVerify, media::PublicDnsResolver,
};
use litellm_inference::resources::CoreResources;
use litellm_secrets::SecretResolver;

use crate::Error;

/// What inference, Redis and the database are all built from, made once per config.
#[derive(Clone)]
pub struct Resources {
    /// The config's `environment_variables` in front of the process environment, as the proxy exports them.
    pub environment: Arc<dyn Lookup + Send + Sync>,
    /// `get_secret` over `environment`: OIDC, the secret manager, then the environment.
    pub secrets: Arc<SecretResolver>,
    pub http: HttpClientConfig,
    pub core: CoreResources,
}

impl Resources {
    pub fn build(config: &Config) -> Result<Self, Error> {
        Self::over(config, Arc::new(ProcessEnvironment))
    }

    /// The resources of `config` over `process` as the process environment.
    ///
    /// Provider auth resolves an `oidc/` AWS web identity token, for Bedrock and for Claude on
    /// Vertex AI's AWS federation, through the secrets of this first config, as Python's
    /// `get_secret` does. Like the client pool, it lives as long as the process.
    pub fn over(config: &Config, process: Arc<dyn Lookup + Send + Sync>) -> Result<Self, Error> {
        let pool = Arc::new(HttpClientPool::new(Arc::new(PublicDnsResolver)));
        let unauthenticated = Self::sharing(config, process, CoreResources::new(pool.clone()))?;
        let aws = AwsAuthService::with_web_identity_tokens(Arc::new(SecretWebIdentityTokens(
            unauthenticated.secrets.clone(),
        )));
        let auth = AuthServices {
            vertex_login: VertexLoginService::default().with_aws(aws.clone()),
            aws,
            ..AuthServices::default()
        };
        Ok(Self {
            core: CoreResources {
                pool,
                auth: Arc::new(auth),
            },
            ..unauthenticated
        })
    }

    /// The resources of `config` over `process`, sharing the client pool and provider auth of
    /// `core`, which another config's resources made, so a new config reuses their connections.
    pub(crate) fn sharing(
        config: &Config,
        process: Arc<dyn Lookup + Send + Sync>,
        core: CoreResources,
    ) -> Result<Self, Error> {
        let environment: Arc<dyn Lookup + Send + Sync> = Arc::new(ConfigEnvironment::new(
            environment_values(&config.environment_variables)?,
            process,
        ));
        let lookup = |name: &str| environment.get(name);
        let settings = &config.litellm_settings;
        let http_settings = HttpSettings::from_layers([
            HttpSettingsLayer::from_environment(&lookup),
            HttpSettingsLayer {
                ssl_verify: settings.ssl_verify.as_ref().map(|value| match value {
                    litellm_config::Spelled::Value(true) => SslVerify::Enabled,
                    litellm_config::Spelled::Value(false) => SslVerify::Disabled,
                    litellm_config::Spelled::Text(value) => SslVerify::parse(value),
                }),
                ssl_certificate: settings.ssl_certificate.as_ref().map(Into::into),
                ssl_security_level: settings.ssl_security_level.clone(),
                ssl_ecdh_curve: settings.ssl_ecdh_curve.clone(),
                force_ipv4: settings.force_ipv4,
                http2: settings.http2,
                aiohttp_trust_env: settings.aiohttp_trust_env,
                disable_aiohttp_trust_env: settings.disable_aiohttp_trust_env,
                disable_aiohttp_transport: settings.disable_aiohttp_transport,
                ..Default::default()
            },
        ]);
        let http = Resolution::from(&http_settings).config;
        let client = core.pool.client(&http, ClientVariant::Provider)?;
        let secrets = Arc::new(crate::environment_secrets(environment.clone(), client));
        Ok(Self {
            environment,
            secrets,
            http,
            core,
        })
    }
}

/// An AWS web identity token reference read as Python's `get_secret` reads it, so an `oidc/`
/// reference names the token's source rather than the token.
struct SecretWebIdentityTokens(Arc<SecretResolver>);

impl std::fmt::Debug for SecretWebIdentityTokens {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SecretWebIdentityTokens")
            .finish_non_exhaustive()
    }
}

impl WebIdentityTokenSource for SecretWebIdentityTokens {
    fn web_identity_token<'a>(&'a self, reference: &'a str) -> WebIdentityTokenFuture<'a> {
        Box::pin(async move {
            self.0
                .get_secret_str(reference, None)
                .await
                .map_err(litellm_error::Source::new)
        })
    }
}

#[cfg(test)]
mod tests {
    use litellm_config::Config;
    use litellm_env::Lookup;
    use rstest::rstest;

    use super::*;

    struct NoProcessEnvironment;

    impl Lookup for NoProcessEnvironment {
        fn get(&self, _: &str) -> Option<String> {
            None
        }
    }

    #[rstest]
    #[tokio::test]
    async fn an_oidc_web_identity_token_reads_the_source_it_names() {
        let config =
            Config::from_yaml("environment_variables:\n  WEB_TOKEN: federated-token\n").unwrap();
        let resources = Resources::over(&config, Arc::new(NoProcessEnvironment)).unwrap();
        let tokens = SecretWebIdentityTokens(resources.secrets.clone());

        let token = tokens
            .web_identity_token("oidc/env/WEB_TOKEN")
            .await
            .unwrap();

        assert_eq!(
            token.map(|token| token.expose().to_owned()).as_deref(),
            Some("federated-token")
        );
    }
}
