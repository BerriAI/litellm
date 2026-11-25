mod admit;
mod boot;
mod callbacks;
mod db;
mod error;
mod gateway_requests;
mod health;
pub mod host;
mod license;
pub mod middleware;
mod periodic;
mod redis;
mod resources;
mod state;
#[cfg(test)]
mod test_support;
pub use admit::{Admit, Reason, Refused};
pub use callbacks::{
    Callbacks, Registered, Unregistrable,
    model_capacity::{self, ModelCapacity},
    parallel_request_limiter::{self, KeyLimiter},
};
pub use error::Error;
pub use gateway_requests::{Category, GatewayRequests};
pub use periodic::PeriodicJob;
pub use redis::RedisClients;
pub use resources::Resources;
pub use state::AppState;

/// The gateway served through the native host.
pub type Gateway = litellm_gateway_inference::Gateway<host::NativeHost>;

use std::{path::Path, sync::Arc};

use axum::Router;

use litellm_config::{Config, Environment};
use litellm_env::{Lookup, ProcessEnvironment};
use litellm_gateway_inference::RoutingCache;
use litellm_http::{
    ClientVariant, HttpClientPool, HttpSettings, HttpSettingsLayer, Resolution,
    media::PublicDnsResolver,
};
use litellm_model_catalog::{BUNDLED_REGISTRY, Catalog};
use litellm_redis_store::{Remote, active_teams::ActiveTeams};
use litellm_router_core::registry::DeploymentFilter;
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

/// The inference routes of a config, and the limiter that counts their calls against each key's
/// own limits.
pub struct Inference {
    pub gateway: Arc<Gateway>,
    pub key_limiter: Arc<KeyLimiter>,
}

/// Whether a license is set at all is all the proxy asks before it honors a reserved priority.
const LICENSE: &str = "LITELLM_LICENSE";

/// The inference routes of `config` with the callbacks the gateway runs around their calls, in
/// the order the proxy registers them: the config's `callbacks`, of which only the dynamic rate
/// limiter is built, then the proxy's own key limiter. The keys' limits and the model groups'
/// capacity are counted in `redis`'s coordination client when there is one, and in this process's
/// memory otherwise.
pub async fn build_inference(
    config: &Config,
    environment: &Environment,
    resources: &Resources,
    routing: RoutingCache,
    redis: &RedisClients,
) -> Result<Inference, Error> {
    let catalog = Catalog::parse(BUNDLED_REGISTRY)?.value;
    let (router, shaping) = boot::router(config, &catalog).await?;
    let deployments = router.registry().deployments(&DeploymentFilter::default());
    let windows = host::ContextWindows::read(&catalog, &deployments).await;
    let limits = parallel_request_limiter::Settings::read(config, environment);
    let counters = parallel_request_limiter::counters(redis.coordination.clone());
    let key_limiter = KeyLimiter::new(limits, counters.clone(), router.clone());
    let licensed = resources.environment.get(LICENSE).is_some();
    let model_capacity = model_capacity::Settings::read(config, limits, licensed).map(|read| {
        let active = ActiveTeams::new(redis.coordination.clone().map(|client| Remote {
            client,
            default_ttl: parallel_request_limiter::REDIS_DEFAULT_TTL,
        }));
        ModelCapacity::new(read, counters, active, router.clone())
    });
    let callbacks = Callbacks::new(
        model_capacity
            .iter()
            .map(|capacity| capacity.registered())
            .chain([key_limiter.registered()])
            .collect(),
    )?;
    let gateway = Gateway::new(
        resources.core.clone(),
        resources.http.clone(),
        resources.secrets.clone(),
        router,
        shaping,
        routing,
        host::NativeHost::new(Arc::new(callbacks), Arc::new(windows)),
    )?;
    Ok(Inference {
        gateway: Arc::new(gateway),
        key_limiter,
    })
}

pub fn install_crypto_provider() {
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        let _ = process_crypto_provider().install_default();
    }
}

#[cfg(feature = "fips")]
fn process_crypto_provider() -> rustls::crypto::CryptoProvider {
    rustls::crypto::default_fips_provider()
}

#[cfg(not(feature = "fips"))]
fn process_crypto_provider() -> rustls::crypto::CryptoProvider {
    rustls::crypto::aws_lc_rs::default_provider()
}

pub fn router(state: AppState, config: &Config) -> Router {
    let inference = middleware::rate_limited(
        litellm_gateway_inference::router(state.inference.clone()),
        state.key_limiter.clone(),
    )
    .route_layer(axum::middleware::from_fn_with_state(
        state.auth(config),
        litellm_gateway_auth::authenticate::<host::NativeRows>,
    ));
    middleware::inference(inference).merge(health::router(state))
}
