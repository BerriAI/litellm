use std::sync::Arc;
use std::time::Duration;

use litellm_core::integrations::custom_logger::CustomLogger;
use litellm_core::router::{Deployment, LiteLLMParams, Router};

use crate::io::realtime_pool::{PoolConfig, RealtimePool, upstream_key};
use crate::proxy_logger::{LiteLLMPythonProxyAPILogger, LogEgressConfig};
use crate::routes;
use crate::state::{AppState, GatewayMessagesServices};

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 4001;
const DEFAULT_PROXY_BASE_URL: &str = "http://localhost:4000";
const DEFAULT_CHANNEL_CAPACITY: usize = 4096;
const DEFAULT_MAX_BATCH_SIZE: usize = 256;
const DEFAULT_FLUSH_INTERVAL_MS: u64 = 500;

pub async fn run() {
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

    let proxy_logger = configured_proxy_logger();
    let loggers: Vec<Arc<dyn CustomLogger>> = vec![proxy_logger];
    let router = Arc::new(build_router());
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

    let state = AppState {
        router,
        master_key,
        loggers: Arc::new(loggers),
        realtime_pool,
        messages_client: litellm_core::runtime::LiteLlm::from_services(
            GatewayMessagesServices::new(|key| std::env::var(key).ok()),
        ),
    };
    let host = std::env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port = resolve_port();
    let listener = tokio::net::TcpListener::bind((host.as_str(), port))
        .await
        .expect("failed to bind listener");
    eprintln!("litellm listening on {host}:{port}");
    axum::serve(listener, routes::app(state))
        .await
        .expect("server error");
}

fn configured_proxy_logger() -> Arc<LiteLLMPythonProxyAPILogger> {
    let base = std::env::var("LITELLM_PROXY_BASE_URL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_PROXY_BASE_URL.to_string());
    let key = std::env::var("LITELLM_MASTER_KEY").unwrap_or_default();
    LiteLLMPythonProxyAPILogger::start(
        base,
        key,
        LogEgressConfig {
            channel_capacity: positive_env(
                "LITELLM_LOG_CHANNEL_CAPACITY",
                DEFAULT_CHANNEL_CAPACITY,
            ),
            max_batch_size: positive_env("LITELLM_LOG_BATCH_SIZE", DEFAULT_MAX_BATCH_SIZE),
            flush_interval: Duration::from_millis(positive_env(
                "LITELLM_LOG_FLUSH_INTERVAL_MS",
                DEFAULT_FLUSH_INTERVAL_MS,
            )),
        },
    )
}

fn positive_env<T>(name: &str, default: T) -> T
where
    T: std::str::FromStr + PartialOrd + From<u8>,
{
    let zero = T::from(0u8);
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<T>().ok())
        .filter(|value| *value > zero)
        .unwrap_or(default)
}

fn register_deployments(router: &Router, pool: &RealtimePool) {
    for deployment in router.deployments() {
        let params = &deployment.litellm_params;
        if let Some(key) = upstream_key(
            &params.model,
            params.api_key.as_deref(),
            params.api_base.as_deref(),
        ) {
            pool.register(key);
        }
    }
}

fn resolve_port() -> u16 {
    match std::env::var("PORT") {
        Ok(raw) => raw.parse().unwrap_or_else(|_| {
            eprintln!("warning: PORT={raw:?} is not a valid port; using {DEFAULT_PORT}");
            DEFAULT_PORT
        }),
        Err(_) => DEFAULT_PORT,
    }
}

fn build_router() -> Router {
    #[cfg(feature = "python-config")]
    if let Ok(config_path) = std::env::var("LITELLM_CONFIG_PATH") {
        match litellm_config::load_model_list(std::path::Path::new(&config_path)) {
            Ok(deployments) => {
                eprintln!("loaded model_list from {config_path} via python config reader");
                return Router::new(deployments);
            }
            Err(err) => {
                eprintln!("config load failed ({err}); falling back to env deployment");
            }
        }
    }
    build_router_from_env()
}

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
