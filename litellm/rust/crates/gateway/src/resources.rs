use std::sync::Arc;

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
    pub fn over(config: &Config, process: Arc<dyn Lookup + Send + Sync>) -> Result<Self, Error> {
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
        let pool = Arc::new(HttpClientPool::new(Arc::new(PublicDnsResolver)));
        let client = pool.client(&http, ClientVariant::Provider)?;
        let secrets = Arc::new(crate::environment_secrets(environment.clone(), client));
        Ok(Self {
            environment,
            secrets,
            http,
            core: CoreResources::new(pool),
        })
    }
}
