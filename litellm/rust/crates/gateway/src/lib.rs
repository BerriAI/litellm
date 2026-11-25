mod admit;
mod boot;
mod callbacks;
mod config_sync;
mod db;
mod error;
mod gateway_requests;
mod health;
pub mod host;
mod license;
mod live;
pub mod middleware;
mod periodic;
mod redis;
mod reload;
mod resources;
mod state;
mod stored;
#[cfg(test)]
mod test_support;
pub use admit::{Admit, Admitted, Reason, Refusal, Refused};
pub use callbacks::{
    Callbacks, Registered, Unregistrable,
    model_capacity::{self, ModelCapacity},
    parallel_request_limiter::{self, KeyLimiter},
    spend_log::{SpendLog, SpendLogs},
};
pub use error::Error;
pub use gateway_requests::{Category, GatewayRequests};
pub use live::{Live, Served};
pub use periodic::PeriodicJob;
pub use redis::RedisClients;
pub use reload::{Poller, Source, reload_every};
pub use resources::Resources;
pub use state::AppState;
pub use stored::{Loaded, StoredConfig};

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
use litellm_redis_store::{Remote, active_teams::ActiveTeams, rate_limits::RateLimits};
use litellm_router_core::registry::DeploymentFilter;
use litellm_secrets::{OidcResolver, SecretManagerState, SecretResolver};

/// The config at `path`, its `os.environ/` references read from the process environment, with
/// the config the proxy stores in its database layered in when `store_model_in_db` is on.
pub async fn load_config(path: impl AsRef<Path>) -> Result<Loaded, Error> {
    let pool = HttpClientPool::new(Arc::new(PublicDnsResolver));
    let lookup = |name: &str| std::env::var(name).ok();
    let settings = HttpSettings::from_layers([HttpSettingsLayer::from_environment(&lookup)]);
    let client = pool.client(&Resolution::from(&settings).config, ClientVariant::Provider)?;
    Box::pin(stored::load(
        path.as_ref(),
        Arc::new(move || environment_secrets(Arc::new(ProcessEnvironment), client.clone())),
    ))
    .await
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
/// memory otherwise. `variables` holds the config's `environment_variables` in front of the
/// process's, as the limiter reads its window from them on every call. With a `spend_log`, the
/// caller's budgets are checked and then held first, as Python does both while it authenticates,
/// and the proxy's spend log callback runs last and writes each served call's row to the database.
pub async fn build_inference(
    config: &Config,
    variables: &Environment,
    resources: &Resources,
    routing: RoutingCache,
    redis: &RedisClients,
    spend_log: Option<&SpendLog>,
) -> Result<Inference, Error> {
    let counters = parallel_request_limiter::counters(redis.coordination.clone());
    inference_over(
        config, variables, resources, routing, redis, counters, spend_log,
    )
    .await
}

/// [`build_inference`] over the routing state and the limiters' counters a previous config's
/// inference counted in, and its spend log callback, so a new config keeps its cooldowns, rate
/// limit windows and queued spend logs.
pub(crate) async fn inference_over(
    config: &Config,
    variables: &Environment,
    resources: &Resources,
    routing: impl Into<Arc<RoutingCache>>,
    redis: &RedisClients,
    counters: Arc<RateLimits>,
    spend_log: Option<&SpendLog>,
) -> Result<Inference, Error> {
    let catalog = Catalog::parse(BUNDLED_REGISTRY)?.value;
    let (router, shaping) = boot::router(config, &catalog).await?;
    let deployments = router.registry().deployments(&DeploymentFilter::default());
    let windows = host::ContextWindows::read(&catalog, &deployments).await;
    let prices = Arc::new(host::HoldPrices::read(&catalog, &deployments).await);
    let limits = parallel_request_limiter::Settings::read(config, variables);
    let key_limiter = KeyLimiter::new(limits, counters.clone(), router.clone());
    let licensed = resources.environment.get(LICENSE).is_some();
    let model_capacity = model_capacity::Settings::read(config, limits, licensed).map(|read| {
        let active = ActiveTeams::new(redis.coordination.clone().map(|client| Remote {
            client,
            default_ttl: parallel_request_limiter::REDIS_DEFAULT_TTL,
        }));
        ModelCapacity::new(read, counters, active, router.clone())
    });
    let budgets = spend_log.map(|spend_log| {
        [
            spend_log.budget_check().registered(),
            spend_log
                .reserver(config.general_settings.disable_budget_reservation)
                .registered(),
        ]
    });
    let callbacks = Callbacks::new(
        budgets
            .into_iter()
            .flatten()
            .chain(model_capacity.iter().map(|capacity| capacity.registered()))
            .chain([key_limiter.registered()])
            .chain(spend_log.map(SpendLog::registered))
            .collect(),
    )?;
    let gateway = Gateway::new(
        resources.core.clone(),
        resources.http.clone(),
        resources.secrets.clone(),
        router,
        shaping,
        routing,
        host::NativeHost::new(Arc::new(callbacks), Arc::new(windows)).with_prices(prices),
    )?
    .with_body_check(litellm_gateway_inference::BodyCheck::new(
        &config.general_settings,
    ));
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
    let inference = middleware::allowlisted(
        inference,
        middleware::request_allowlist::Settings::from_general(&config.general_settings),
    );
    middleware::inference(inference).merge(health::router(state))
}

#[cfg(test)]
mod tests {
    use litellm_gateway_auth::Authenticated;
    use litellm_gateway_auth_core::{
        Belongs, Caller, Credential, Token,
        record::{Budget, EndUserRecord},
    };
    use litellm_gateway_inference::{
        CallHooks, HookError, Host, RouterTiers,
        record::{self, CallRecord},
    };
    use litellm_redis_store::spend_counters::SpendCounter;
    use rstest::rstest;
    use serde_json::{Map, Value, json};

    use super::*;
    use crate::host::{Hold, Key, auth::keys::Held};

    const TOKEN: &str = "sk-spent";
    const MAX_BUDGET: f64 = 10.0;

    /// A caller whose cached key still reads no spend, while the key's row holds its whole budget.
    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_caller_at_its_budget_is_refused_by_the_check_before_anything_is_held() {
        let held = Held::holding(
            TOKEN,
            json!({"spend": MAX_BUDGET, "max_budget": MAX_BUDGET}),
        )
        .await;
        let config = Config::default();
        let spend_log = SpendLog::start(
            &held.repositories(),
            &RedisClients::default(),
            &Environment::default(),
        );
        let inference = build_inference(
            &config,
            &Environment::default(),
            &Resources::build(&config).unwrap(),
            RoutingCache::new(None, RouterTiers::default()),
            &RedisClients::default(),
            Some(&spend_log),
        )
        .await
        .unwrap();
        let hash = Token::new(TOKEN).hash().to_string();
        let caller = Authenticated::from(Caller {
            credential: Credential::Key {
                key: Key::stored(&hash, Some("spent")).with_budget(Budget {
                    max: Some(MAX_BUDGET),
                    spend: 0.0,
                }),
            },
            belongs: Belongs::nobody(),
            end_user: EndUserRecord::default(),
        });
        let admitted = Admitted {
            hold: Some(Hold {
                reservation_cost: 1.0,
                input_cost: 0.5,
                input_tokens: 9,
            }),
            free: false,
        };
        let body: Map<String, Value> = Map::from_iter([("model".to_owned(), json!("held-model"))]);

        let refused = record::scope(CallRecord::new(), async {
            let call = inference.gateway.host().hooks(&caller, Some(&admitted));
            match call.pre_call(body).await {
                Ok(_) => None,
                Err(HookError::Refused(refusal)) => Some(refusal.message),
                Err(HookError::Failed(error)) => panic!("the call failed: {error}"),
            }
        })
        .await;
        let counter = SpendCounter::Key {
            token: hash,
            window: None,
        };
        let spend = spend_log
            .spend_counters()
            .get_local(&counter)
            .unwrap()
            .map(|spend| spend.as_f64());

        assert_eq!(
            (refused.as_deref(), spend),
            (
                Some("Budget has been exceeded! Key=spent Current cost: 10.0, Max budget: 10.0"),
                Some(MAX_BUDGET)
            )
        );
    }
}
