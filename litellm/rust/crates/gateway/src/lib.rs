mod admit;
mod db;
mod error;
mod health;
mod license;
pub mod middleware;
mod redis;
mod resources;
mod state;
pub use admit::{Admit, Refused};
pub use error::Error;
pub use redis::RedisClients;
pub use resources::Resources;
pub use state::AppState;

use std::{path::Path, sync::Arc};

use axum::Router;

use litellm_config::Config;
use litellm_env::{Lookup, ProcessEnvironment};
use litellm_gateway_auth::{Auth, BearerOrXApiKey};
use litellm_gateway_inference::{Gateway, ModelRouter};
use litellm_http::{
    ClientVariant, HttpClientPool, HttpSettings, HttpSettingsLayer, Resolution,
    media::PublicDnsResolver,
};
use litellm_model_catalog::{BUNDLED_REGISTRY, Catalog};
use litellm_secrets::{OidcResolver, SecretManagerState, SecretResolver};

/// The config at `path`, its `os.environ/` references read from the process environment.
pub async fn load_config(path: impl AsRef<Path>) -> Result<Config, Error> {
    let pool = HttpClientPool::new(Arc::new(PublicDnsResolver));
    let lookup = |name: &str| std::env::var(name).ok();
    let settings = HttpSettings::from_layers([HttpSettingsLayer::from_environment(&lookup)]);
    let client = pool.client(&Resolution::from(&settings).config, ClientVariant::Provider)?;
    let secrets = environment_secrets(Arc::new(ProcessEnvironment), client);
    Ok(litellm_config_load::load(path, secrets).await?)
}

pub(crate) fn environment_secrets(
    environment: Arc<dyn Lookup + Send + Sync>,
    client: litellm_http::Client,
) -> SecretResolver {
    SecretResolver::new_python_compatible(
        Arc::new(SecretManagerState::default()),
        environment,
        OidcResolver::new(client),
    )
}

pub fn build_inference(config: &Config, resources: &Resources) -> Result<Arc<Gateway>, Error> {
    let catalog = Catalog::parse(BUNDLED_REGISTRY)?.value;
    let router = ModelRouter::from_config(config, &catalog);
    Ok(Arc::new(
        Gateway::new(
            resources.core.clone(),
            resources.http.clone(),
            resources.secrets.clone(),
            router,
        )?
        .with_client_side_credentials(config.general_settings.allow_client_side_credentials),
    ))
}

pub fn install_crypto_provider() {
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }
}

pub fn router(state: AppState, config: &Config) -> Router {
    let auth = Auth::from_config(config).with_extractor(Arc::new(BearerOrXApiKey));
    let inference = litellm_gateway_inference::router(state.inference.clone()).route_layer(
        axum::middleware::from_fn_with_state(auth, litellm_gateway_auth::authenticate),
    );
    middleware::inference(inference).merge(health::router(state))
}
