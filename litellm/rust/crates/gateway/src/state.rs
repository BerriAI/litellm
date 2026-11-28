use std::{num::NonZeroUsize, sync::Arc, time::Duration};

use axum::extract::FromRef;
use futures_util::StreamExt;
use litellm_config::{Config, Environment, GeneralSettings};
use litellm_db::Pools;
use litellm_env::{Lookup, ProcessEnvironment};
use litellm_gateway_auth::Auth;
use litellm_license::Entitlements;
use litellm_redis_client::RedisClient;
use litellm_redis_store::{
    Clock, Remote,
    auth_cache::{AuthCache, Settings},
    cache_invalidation::{self, CacheInvalidation},
    rate_limits::RateLimits,
    spend_counters::SpendCounters,
    system_clock,
};
use litellm_repositories::{Database, GatewayRequestRepository};
use tokio::runtime::Handle;

use crate::{
    Error, Gateway, Inference, KeyLimiter, RedisClients, Resources, SpendLog,
    callbacks::parallel_request_limiter,
    db,
    gateway_requests::GatewayRequests,
    host::{self, NativeRows, auth::AuthStore},
    license::build_license,
    periodic::{self, PeriodicJob},
};

#[derive(Clone)]
pub struct AppState {
    pub inference: Arc<Gateway>,
    pub redis: RedisClients,
    pub database: Option<Pools>,
    /// True when the config names a database the gateway cannot sign in to (token auth), so the
    /// proxy holds rows the gateway cannot read.
    pub database_unread: bool,
    /// Where a virtual key is read from: the auth cache in front of the proxy's database. `None`
    /// without a database.
    pub auth_store: Option<Arc<AuthStore>>,
    pub entitlements: Arc<Entitlements>,
    /// The proxy's `parallel_request_limiter_v3`, which the inference routes' callback stage runs
    /// and the limit check before the body asks.
    pub key_limiter: Arc<KeyLimiter>,
    /// The spend log callback the inference routes' callback stage runs, and the writer it queues
    /// to, `None` without a database.
    pub spend_logs: Option<Arc<SpendLog>>,
    /// The config's `environment_variables` in front of the process environment.
    pub environment: Arc<dyn Lookup + Send + Sync>,
    /// What the key limits and model group capacity count in, in this process's memory when there
    /// is no coordination Redis.
    pub counters: Arc<RateLimits>,
    /// The variables only the environment holds, read through `environment`, for the readers
    /// Python runs after its config exported them, such as the key limiter and admit.
    pub variables: Environment,
    /// The gateway request counts the request volume middleware records. `None` without a
    /// database.
    pub gateway_requests: Option<Arc<GatewayRequests>>,
    /// The job that writes `gateway_requests` every `PROXY_BATCH_WRITE_AT` seconds.
    pub gateway_requests_flush: Option<Arc<PeriodicJob>>,
}

impl AppState {
    pub async fn build(config: &Config, environment: &Environment) -> Result<Self, Error> {
        Self::over(config, environment, &Resources::build(config)?).await
    }

    /// The state `config` describes, built from the `resources` made for it. `environment` is
    /// the process's alone, for the settings Python reads before it loads its config.
    pub async fn over(
        config: &Config,
        environment: &Environment,
        resources: &Resources,
    ) -> Result<Self, Error> {
        let variables = Environment::read(&*resources.environment)?;
        let redis = crate::redis::connect(config, resources).await?;
        let routing = crate::redis::routing(&redis, environment);
        let counters = parallel_request_limiter::counters(redis.coordination.clone());
        let connected = db::connect(config, &resources.secrets).await?;
        let database_unread = matches!(connected, db::Connected::Unread);
        let connected = connected.pools();
        let spend_logs = connected
            .as_ref()
            .map(|(_, repositories)| Arc::new(SpendLog::start(repositories, &redis, &variables)));
        let Inference {
            gateway: inference,
            key_limiter,
        } = crate::inference_over(
            config,
            &variables,
            resources,
            routing,
            &redis,
            counters.clone(),
            spend_logs.as_deref(),
        )
        .await?;
        let (database, connected) = connected
            .zip(spend_logs.as_ref())
            .map(|((pools, repositories), spend_logs)| {
                let requests = Arc::new(GatewayRequests::new());
                let flush = gateway_requests_flush(
                    requests.clone(),
                    GatewayRequestRepository::new(repositories.clone()),
                    environment,
                );
                let spend_counters = spend_logs.spend_counters().clone();
                let auth_store = Arc::new(auth_store(config, &redis, repositories, spend_counters));
                (pools, (auth_store, (requests, Arc::new(flush))))
            })
            .unzip();
        let (auth_store, gateway_requests) = connected.unzip();
        let (gateway_requests, gateway_requests_flush) = gateway_requests.unzip();
        let entitlements = Arc::new(build_license(config, environment, resources).await);
        Ok(Self {
            inference,
            redis,
            database,
            database_unread,
            auth_store,
            entitlements,
            key_limiter,
            spend_logs,
            environment: resources.environment.clone(),
            counters,
            variables,
            gateway_requests,
            gateway_requests_flush,
        })
    }

    /// The state `config` describes, built over what this one keeps for the life of the process:
    /// the Redis clients, the database pools and the keys read through them, the license, the
    /// HTTP client pool, the deployments' routing state and the limiters' counters. Only what
    /// `config` decides is built again, its environment, router and callbacks, and the kept auth
    /// cache takes its `user_api_key_cache_max_size`, as the proxy's reload resizes it. Its TTL
    /// stays the one boot read, as the proxy reads `user_api_key_cache_ttl` only at startup.
    pub async fn next(&self, config: &Config) -> Result<Self, Error> {
        let resources = Resources::sharing(
            config,
            Arc::new(ProcessEnvironment),
            self.inference.resources.clone(),
        )?;
        let variables = Environment::read(&*resources.environment)?;
        let Inference {
            gateway: inference,
            key_limiter,
        } = crate::inference_over(
            config,
            &variables,
            &resources,
            self.inference.shared_routing(),
            &self.redis,
            self.counters.clone(),
            self.spend_logs.as_deref(),
        )
        .await?;
        if let Some(auth_store) = &self.auth_store {
            auth_store.resize(auth_cache_settings(&config.general_settings).memory_entries);
        }
        inference.install((*self.inference.cost_map()).clone());
        Ok(Self {
            inference,
            key_limiter,
            environment: resources.environment,
            variables,
            ..self.clone()
        })
    }

    /// Who the inference routes and `admit` authenticate, which the core decides over the native
    /// host: the master key of `config`, and with a database a virtual key it holds. A key's own
    /// budget is checked only with the spend logs, whose counters the budget checks read.
    pub(crate) fn auth(&self, config: &Config) -> Auth<NativeRows> {
        host::auth::auth(config, self.auth_store.clone(), self.spend_logs.is_some())
    }

    /// Drains the spend logs still queued, writes the gateway request counts still in memory,
    /// then closes the database pools.
    pub async fn close(&self) {
        self.close_with_accounting(None).await;
    }

    /// Standalone owns the applier. Python owns it when the gateway is running in front.
    pub fn start_standalone_accounting(&self) -> Option<PeriodicJob> {
        let logs = self.spend_logs.as_ref()?;
        Some(crate::accounting::start(
            logs.repository.clone(),
            self.auth_store.as_ref()?.clone(),
            &self.variables,
            &*self.environment,
        ))
    }

    /// Only the standalone entry point starts this scheduler.
    pub fn start_standalone_budget_reset(&self, config: &Config) -> Option<PeriodicJob> {
        let logs = self.spend_logs.as_ref()?;
        crate::budget_reset::start(
            config,
            &*self.environment,
            logs.budget_resets.clone(),
            self.auth_store.clone()?,
            logs.spend_counters().clone(),
            self.redis.coordination.clone()?,
        )
    }

    /// Runs the applier's last pass after all locally queued deliveries are durable.
    pub async fn close_with_accounting(&self, accounting: Option<PeriodicJob>) {
        if let Some(spend_logs) = &self.spend_logs {
            spend_logs.drain().await;
        }
        if let Some(accounting) = accounting {
            accounting.close().await;
        }
        if let Some(flush) = &self.gateway_requests_flush {
            flush.close().await;
        }
        if let Some(database) = &self.database {
            database.close().await;
        }
    }
}

/// `update_gateway_requests_job`: flushes `requests` through `repository` every batch write
/// interval and once more on shutdown.
fn gateway_requests_flush(
    requests: Arc<GatewayRequests>,
    repository: GatewayRequestRepository,
    environment: &Environment,
) -> PeriodicJob {
    let repository = Arc::new(repository);
    PeriodicJob::start(periodic::batch_write_interval(environment), move || {
        let (requests, repository) = (requests.clone(), repository.clone());
        async move { requests.flush(&repository).await }
    })
}

/// Where the gateway reads a virtual key: through the auth cache Python's `UserApiKeyCache` is,
/// this process's memory in front of the database, with the coordination Redis between them when
/// `enable_redis_auth_cache` attaches it. Every pod's evictions reach the memory whenever a
/// coordination Redis carries them, as Python subscribes with the flag on or off, and so do its
/// spend resets, to `spend_counters`, the counters the spend log charges.
pub(crate) fn auth_store(
    config: &Config,
    redis: &RedisClients,
    database: Database,
    spend_counters: Arc<SpendCounters>,
) -> AuthStore {
    let cache = Arc::new(auth_cache(config, redis, system_clock()));
    if let Some(client) = &redis.coordination {
        hear_evictions(&cache, spend_counters, client);
    }
    AuthStore::new(cache, database)
}

/// Drops from this process's memory every key a pod evicts, Python or native, for as long as
/// `cache` lives. The store keeps a copy of each key it reads there, and without this a blocked
/// key would keep answering from it until the copy expired. A spend reset's counter and floor
/// reach `spend_counters`, so a spend read never answers the spend from before the reset.
fn hear_evictions(
    cache: &Arc<AuthCache>,
    spend_counters: Arc<SpendCounters>,
    client: &RedisClient,
) {
    let channel = CacheInvalidation::new(
        client.clone(),
        client.namespace().as_str(),
        Handle::current(),
    );
    let mut evictions = channel.subscribe();
    let cache = Arc::downgrade(cache);
    litellm_call_scope::spawn_detached(async move {
        while let Some(invalidation) = evictions.next().await {
            let Some(cache) = cache.upgrade() else {
                return;
            };
            cache_invalidation::apply(&invalidation, &cache, &spend_counters);
        }
    });
}

/// Python's `UserApiKeyCache`, with the coordination Redis attached when
/// `enable_redis_auth_cache` is on, sized as `general_settings` sizes it (see [`auth_cache_settings`]).
/// Every write the gateway makes takes the cache's TTL, so no Redis write takes the client's default.
pub(crate) fn auth_cache(config: &Config, redis: &RedisClients, now: Clock) -> AuthCache {
    const NEVER_TAKEN: Duration = Duration::ZERO;
    let remote = redis
        .coordination
        .clone()
        .filter(|_| config.litellm_settings.enable_redis_auth_cache == Some(true))
        .map(|client| Remote {
            client,
            default_ttl: NEVER_TAKEN,
        });
    AuthCache::with_clock(remote, auth_cache_settings(&config.general_settings), now)
}

/// `UserAPIKeyCacheTTLEnum.in_memory_cache_ttl`, what the proxy keeps an auth object for unless
/// `user_api_key_cache_ttl` says otherwise.
const KEY_TTL: Duration = Duration::from_secs(60);

/// What the proxy's startup makes of `user_api_key_cache` (`load_config` in `proxy_server.py`):
/// `user_api_key_cache_ttl` seconds for every entry, in memory and in Redis, else
/// `UserAPIKeyCacheTTLEnum`'s 60, and `user_api_key_cache_max_size` entries in each partition, else
/// `InMemoryCache`'s 200.
fn auth_cache_settings(settings: &GeneralSettings) -> Settings {
    Settings {
        memory_entries: settings
            .user_api_key_cache_max_size
            .map_or(Settings::default().memory_entries, NonZeroUsize::get),
        memory_ttl: Some(settings.user_api_key_cache_ttl.unwrap_or(KEY_TTL)),
    }
}

impl FromRef<AppState> for RedisClients {
    fn from_ref(state: &AppState) -> Self {
        state.redis.clone()
    }
}

impl FromRef<AppState> for Option<Pools> {
    fn from_ref(state: &AppState) -> Self {
        state.database.clone()
    }
}

#[cfg(test)]
mod tests {
    use litellm_gateway_auth_core::Token;
    use litellm_redis_store::{
        Number, cache_invalidation::Invalidation, spend_counters::SpendCounter,
    };
    use litellm_redis_testing::{RedisServer, eventually};
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::host::auth::keys::Held;

    #[rstest]
    #[case::unset("model_list: []\n", 60)]
    #[case::from_the_config_environment(
        "model_list: []\nenvironment_variables:\n  LITELLM_RATE_LIMIT_WINDOW_SIZE: \"3600\"\n",
        3600
    )]
    #[tokio::test]
    async fn the_key_limiter_reads_its_window_from_the_config_environment(
        #[case] yaml: &str,
        #[case] window_size: i64,
    ) {
        let config = Config::from_yaml(yaml).unwrap();

        let state = AppState::build(&config, &Environment::default())
            .await
            .unwrap();

        assert_eq!(state.key_limiter.window_size(), window_size);
        state.close().await;
    }

    #[rstest]
    #[case::shrunk(
        "  user_api_key_cache_max_size: 2\n",
        "  user_api_key_cache_max_size: 1\n",
        false
    )]
    #[case::back_to_the_default("  user_api_key_cache_max_size: 1\n", "", true)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_reload_resizes_the_auth_cache_it_keeps(
        #[case] booted: &str,
        #[case] reloaded: &str,
        #[case] first_kept: bool,
    ) {
        let held = Held::start().await;
        held.key("sk-first", json!({})).await;
        held.key("sk-second", json!({})).await;
        let config = |general_settings: &str| {
            Config::from_yaml(&format!(
                "model_list: []\ngeneral_settings:\n  database_url: {}\n{general_settings}",
                held.url()
            ))
            .unwrap()
        };
        let hash = |token: &str| Token::new(token).hash();
        let state = AppState::build(&config(booted), &Environment::default())
            .await
            .unwrap();
        let keys = state.auth_store.clone().unwrap();
        assert!(keys.loaded(&hash("sk-first")).await.unwrap().is_some());

        let next = state.next(&config(reloaded)).await.unwrap();
        assert!(keys.loaded(&hash("sk-second")).await.unwrap().is_some());

        assert_eq!(
            (
                keys.cached(&hash("sk-first")).await.is_some(),
                keys.cached(&hash("sk-second")).await.is_some()
            ),
            (first_kept, true)
        );
        next.close().await;
    }
    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_spend_reset_another_pod_announces_reaches_the_counters_the_spend_log_charges() {
        let held = Held::start().await;
        let redis = RedisServer::start().await.unwrap();
        let config = Config::from_yaml(&format!(
            "general_settings:\n  database_url: {}\n  coordination_redis:\n    host: 127.0.0.1\n    port: {}\n",
            held.url(),
            redis.port()
        ))
        .unwrap();
        let state = AppState::build(&config, &Environment::default())
            .await
            .unwrap();
        let counters = state.spend_logs.as_ref().unwrap().spend_counters().clone();
        let channel = CacheInvalidation::new(
            state.redis.coordination.clone().unwrap(),
            None,
            Handle::current(),
        );
        let team = SpendCounter::Team {
            team_id: "t".to_owned(),
            window: None,
        };
        let reset = |cache_key: String| Invalidation::Set {
            cache_key,
            value: Number::Float(0.0),
            ttl: Some(Number::Int(60)),
        };
        counters.set_local(&team, Number::Float(7.0), None).unwrap();

        eventually(async || {
            channel.publish(&reset(team.key()));
            channel.publish(&reset(team.floor_key()));
            let held = (
                counters.get_local(&team).unwrap(),
                counters.floor(&team).unwrap(),
            );
            (held == (Some(Number::Float(0.0)), Some(Number::Float(0.0)))).then_some(())
        })
        .await;
        state.close().await;
    }
}
