//! LiteLLM AI Gateway — a minimal Axum server fronting the Rust router.
//!
//! Flow: client → `POST /v1/realtime` → `router.realtime()` selects a deployment
//! (simple-shuffle) → `io::realtime::realtime()` invokes OpenAI. The
//! server owns transport + config; routing lives in the `router` crate.
//!
//! The binary requires the `server` feature (declared in `Cargo.toml` via
//! `required-features`), so cargo skips it unless that feature is on. Everything
//! the binary needs lives in the library (`litellm_ai_gateway`); `main` just
//! wires startup.

use std::sync::Arc;
use std::time::Duration;

use litellm_ai_gateway::auth::circuit_breaker::{CircuitBreakerConfig, CircuitBreakerRegistry};
use litellm_ai_gateway::hardening::GlobalRateLimiter;
use litellm_ai_gateway::io::realtime_pool::{PoolConfig, RealtimePool, upstream_key};
use litellm_ai_gateway::metrics::GatewayMetrics;
use litellm_ai_gateway::routes;
use litellm_ai_gateway::state::{AppState, GatewayConfig};
use litellm_core::auth::KeyCache;
use litellm_core::persistence::{PostgresStore, RedisPostgresSpendFlush, RedisStore};
use litellm_core::router::{Deployment, LiteLLMParams, Router};
use litellm_core::spend_tracking::SpendWorker;

use litellm_ai_gateway::integrations::custom_logger::CustomLogger;
use litellm_ai_gateway::integrations::litellm_python_proxy_api::LiteLLMPythonProxyAPILogger;

/// Bind to localhost by default so the gateway is not a public, unauthenticated
/// provider proxy out of the box. Override with `HOST` (e.g. `0.0.0.0`).
const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 4001;

#[tokio::main]
async fn main() {
    // Initialize structured logging (JSON format for log aggregation)
    // Set RUST_LOG env var to control log level (e.g., RUST_LOG=info)
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    tracing::info!("starting litellm-ai-gateway");

    // Trim before storing so it matches the trimmed bearer token in `auth`
    // (avoids a silent auth failure when the env var has surrounding whitespace).
    let master_key: Option<Arc<str>> = std::env::var("LITELLM_MASTER_KEY")
        .ok()
        .map(|key| key.trim().to_string())
        .filter(|key| !key.is_empty())
        .map(Arc::from);
    if master_key.is_none() {
        eprintln!(
            "warning: LITELLM_MASTER_KEY is not set; /v1/realtime will reject all requests (fail closed)"
        );
    }

    // Spawn the realtime-logging worker (drains a channel → POSTs batches to the
    // Python proxy's /v1/callbacks/logs). Built here so the spawn lands on the
    // tokio runtime. `from_env` reads LITELLM_PROXY_BASE_URL + LITELLM_MASTER_KEY.
    let proxy_logger = LiteLLMPythonProxyAPILogger::from_env();
    let loggers: Vec<Arc<dyn CustomLogger>> = vec![proxy_logger];

    let router = Arc::new(build_router());
    eprintln!("router has {} deployments:", router.deployments().len());
    for d in router.deployments() {
        eprintln!(
            "  model_name={}, model={}",
            d.model_name, d.litellm_params.model
        );
    }

    // Build the pre-warmed realtime pool and register each deployment's upstream
    // so the background replenisher starts warming it. `REALTIME_POOL_SIZE=0`
    // yields a disabled pool → every connect fresh-dials (original behavior).
    let pool_config = PoolConfig::from_env();
    let realtime_pool = RealtimePool::spawn(pool_config);
    if pool_config.enabled() {
        register_deployments(&router, &realtime_pool);
        eprintln!(
            "realtime connection pool enabled: target {} warm sockets/key, max idle {}s",
            pool_config.target_size,
            pool_config.max_idle.as_secs()
        );
    } else {
        eprintln!(
            "realtime connection pool disabled (REALTIME_POOL_SIZE=0); fresh-dialing each connect"
        );
    }

    // Key cache: in-memory LRU for API key auth objects.
    let key_cache = Arc::new(KeyCache::new(Duration::from_secs(600), 10_000));

    // Redis: optional, for spend counters and rate limiting.
    let redis = match std::env::var("REDIS_URL") {
        Ok(url) => match RedisStore::connect(&url).await {
            Ok(store) => {
                eprintln!("Redis connected: {url}");
                Some(Arc::new(store))
            }
            Err(err) => {
                eprintln!("warning: Redis connection failed ({err}); spend counters disabled");
                None
            }
        },
        Err(_) => {
            eprintln!("REDIS_URL not set; Redis spend counters disabled");
            None
        }
    };

    // PostgreSQL: optional, for spend logs and key lookups.
    let postgres = match std::env::var("DATABASE_URL") {
        Ok(url) => match PostgresStore::connect(&url).await {
            Ok(store) => {
                eprintln!("PostgreSQL connected");
                Some(Arc::new(store))
            }
            Err(err) => {
                eprintln!("warning: PostgreSQL connection failed ({err}); spend logs disabled");
                None
            }
        },
        Err(_) => {
            eprintln!("DATABASE_URL not set; PostgreSQL spend logs disabled");
            None
        }
    };

    // Spend worker: background task that batches spend entries and flushes them.
    let spend_worker = match (&redis, &postgres) {
        (Some(redis), Some(postgres)) => {
            let flush = RedisPostgresSpendFlush::new(
                RedisStore::from_manager(redis.clone_manager()),
                PostgresStore::from_pool(postgres.clone_pool()),
            );
            let worker = SpendWorker::spawn(100, Duration::from_millis(100), flush);
            eprintln!("spend tracking enabled: Redis + PostgreSQL");
            Some(Arc::new(worker))
        }
        _ => {
            eprintln!("spend tracking disabled (need both REDIS_URL and DATABASE_URL)");
            None
        }
    };

    let state = AppState {
        router,
        master_key,
        loggers: Arc::new(loggers),
        realtime_pool,
        key_cache,
        redis,
        postgres,
        spend_worker,
        http_client: Arc::new(AppState::new_http_client()),
        circuit_breakers: Arc::new(CircuitBreakerRegistry::new(CircuitBreakerConfig::default())),
        metrics: Arc::new(GatewayMetrics::new()),
        config: GatewayConfig::from_env(),
        global_rate_limiter: Arc::new(GlobalRateLimiter::new(
            std::env::var("GLOBAL_RATE_LIMIT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10_000),
            60, // 60-second window
        )),
    };

    let host = std::env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port = resolve_port();

    let listener = tokio::net::TcpListener::bind((host.as_str(), port))
        .await
        .expect("failed to bind listener");
    eprintln!("litellm-ai-gateway listening on {host}:{port}");

    // Graceful shutdown: wait for SIGTERM or SIGINT, then drain in-flight requests
    let shutdown = async {
        let ctrl_c = async {
            tokio::signal::ctrl_c()
                .await
                .expect("failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .await
                .expect("failed to install SIGTERM handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }

        eprintln!("shutdown signal received, draining in-flight requests...");
    };

    let app = routes::app(state);

    let tls_cert = std::env::var("TLS_CERT").ok();
    let tls_key = std::env::var("TLS_KEY").ok();

    match (tls_cert, tls_key) {
        (Some(cert_path), Some(key_path)) => {
            use axum_server::tls_rustls::RustlsConfig;

            eprintln!("TLS enabled: cert={cert_path}, key={key_path}");
            let rustls_config = RustlsConfig::from_pem_file(&cert_path, &key_path)
                .await
                .expect("failed to load TLS certificate and key");

            let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
            axum_server::bind_rustls(addr, rustls_config)
                .serve(app.into_make_service())
                .await
                .expect("TLS server error");
        }
        _ => {
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown)
                .await
                .expect("server error");
        }
    }

    eprintln!("shutdown complete");
}

/// Register every deployment's upstream key with the pool so the replenisher
/// pre-warms it. Mirrors `service::run`'s key derivation (strip `openai/`, resolve
/// api_key); deployments whose key can't be resolved are skipped (they fresh-dial
/// and surface the auth error on the request path, as before).
fn register_deployments(router: &Router, pool: &RealtimePool) {
    for deployment in router.deployments() {
        let params = &deployment.litellm_params;
        let provider_model = params
            .model
            .strip_prefix("openai/")
            .unwrap_or(&params.model);
        if let Some(key) = upstream_key(
            provider_model,
            params.api_key.as_deref(),
            params.api_base.as_deref(),
        ) {
            pool.register(key);
        }
    }
}

/// Resolve `PORT`, warning (rather than silently defaulting) on an invalid value.
fn resolve_port() -> u16 {
    match std::env::var("PORT") {
        Ok(raw) => raw.parse().unwrap_or_else(|_| {
            eprintln!("warning: PORT={raw:?} is not a valid port; using {DEFAULT_PORT}");
            DEFAULT_PORT
        }),
        Err(_) => DEFAULT_PORT,
    }
}

/// Build the router. Tries config sources in order:
/// 1. Native YAML loader (`LITELLM_YAML_CONFIG` env var)
/// 2. Env-based fallback (single deployment from `OPENAI_REALTIME_MODEL`)
fn build_router() -> Router {
    if let Ok(config_path) = std::env::var("LITELLM_YAML_CONFIG") {
        match litellm_ai_gateway::config::load_router_from_yaml(&config_path) {
            Ok(router) => {
                eprintln!("loaded model_list from {config_path} via native YAML loader");
                return router;
            }
            Err(err) => {
                eprintln!("YAML config load failed ({err}); falling back to env deployment");
            }
        }
    }
    build_router_from_env()
}

/// Build a minimal single-deployment `model_list` from the environment.
///
/// A real deployment loads `model_list` from config; this is the minimal stand-in
/// so the gateway has one OpenAI deployment to route to.
fn build_router_from_env() -> Router {
    let model =
        std::env::var("OPENAI_REALTIME_MODEL").unwrap_or_else(|_| "gpt-realtime".to_string());
    let api_key = std::env::var("OPENAI_API_KEY").ok();
    if api_key.is_none() {
        eprintln!(
            "warning: OPENAI_API_KEY is not set; realtime requests will fail with auth errors"
        );
    }
    let deployment = Deployment {
        model_name: model.clone(),
        litellm_params: LiteLLMParams {
            model,
            api_key,
            api_base: None,
        },
    };
    Router::new(vec![deployment])
}
