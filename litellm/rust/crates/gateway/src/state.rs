use std::{sync::Arc, time::Duration};

use axum::extract::FromRef;
use futures_util::StreamExt;
use litellm_auth_store::KeyDatabase;
use litellm_config::{Config, Environment};
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
use litellm_repositories::{
    Database, GatewayRequestRepository, KeyRepository, ObjectPermissionRepository,
};
use tokio::runtime::Handle;

use crate::{
    Error, Gateway, Inference, KeyLimiter, RedisClients, Resources, SpendLog,
    callbacks::parallel_request_limiter,
    db,
    gateway_requests::GatewayRequests,
    host::{
        self, NativeRows,
        auth::{AuthStore, KEY_TTL},
    },
    license::build_license,
    periodic::{self, PeriodicJob},
};

#[derive(Clone)]
pub struct AppState {
    pub inference: Arc<Gateway>,
    pub redis: RedisClients,
    pub database: Option<Pools>,
    /// Where a virtual key is read from: the auth cache in front of the proxy's database. `None`
    /// without a database.
    pub keys: Option<Arc<AuthStore>>,
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
                let keys = Arc::new(keys(config, &redis, repositories, spend_counters));
                (pools, (keys, (requests, Arc::new(flush))))
            })
            .unzip();
        let (keys, gateway_requests) = connected.unzip();
        let (gateway_requests, gateway_requests_flush) = gateway_requests.unzip();
        let entitlements = Arc::new(build_license(config, environment, resources).await);
        Ok(Self {
            inference,
            redis,
            database,
            keys,
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
    /// `config` decides is built again, its environment, router and callbacks.
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
        Ok(Self {
            inference,
            key_limiter,
            environment: resources.environment,
            variables,
            ..self.clone()
        })
    }

    /// Who the inference routes and `admit` authenticate, which the core decides over the native
    /// host: the master key of `config`, and with a database a virtual key it holds.
    pub(crate) fn auth(&self, config: &Config) -> Auth<NativeRows> {
        host::auth::auth(config, self.keys.clone())
    }

    /// Drains the spend logs still queued, writes the gateway request counts still in memory,
    /// then closes the database pools.
    pub async fn close(&self) {
        if let Some(spend_logs) = &self.spend_logs {
            spend_logs.drain().await;
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
pub(crate) fn keys(
    config: &Config,
    redis: &RedisClients,
    database: Database,
    spend_counters: Arc<SpendCounters>,
) -> AuthStore {
    let database = KeyDatabase::new(
        Arc::new(KeyRepository::new(database.clone())),
        Arc::new(ObjectPermissionRepository::new(database)),
    );
    let cache = Arc::new(auth_cache(config, redis, system_clock()));
    if let Some(client) = &redis.coordination {
        hear_evictions(&cache, spend_counters, client);
    }
    AuthStore::new(cache, database)
}

/// Drops from this process's memory every key a pod evicts, Python or native, for as long as
/// `cache` lives. The store keeps a copy of each key it reads there, and without this a blocked
/// key would keep answering from it until the copy expired. A spend reset's counter and floor
/// reach `spend_counters`, so a reservation never holds against the spend from before the reset.
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
    tokio::spawn(async move {
        while let Some(invalidation) = evictions.next().await {
            let Some(cache) = cache.upgrade() else {
                return;
            };
            cache_invalidation::apply(&invalidation, &cache, &spend_counters);
        }
    });
}

/// Python's `UserApiKeyCache`, with the coordination Redis attached when
/// `enable_redis_auth_cache` is on. Its memory keeps a Redis hit for Python's
/// `default_in_memory_ttl`, and every write the gateway makes names its own TTL, so no Redis
/// write takes the default.
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
    let settings = Settings {
        memory_ttl: Some(KEY_TTL),
        ..Settings::default()
    };
    AuthCache::with_clock(remote, settings, now)
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
    use litellm_redis_store::{
        Number, cache_invalidation::Invalidation, spend_counters::SpendCounter,
    };
    use litellm_redis_testing::{RedisServer, eventually};
    use rstest::rstest;

    use super::*;

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
    #[tokio::test(flavor = "multi_thread")]
    async fn a_spend_reset_another_pod_announces_reaches_the_counters_a_reservation_holds_against()
    {
        let redis = RedisServer::start().await.unwrap();
        let config = Config::from_yaml(&format!(
            "general_settings:\n  database_url: postgresql://postgres:postgres@127.0.0.1:1/postgres\n  coordination_redis:\n    host: 127.0.0.1\n    port: {}\n",
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
