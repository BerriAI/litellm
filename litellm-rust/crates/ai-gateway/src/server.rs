use std::sync::Arc;

use crate::constants::{DEFAULT_HOST, DEFAULT_PORT};
use crate::integrations::litellm_python_proxy_api::LiteLLMPythonProxyAPILogger;
#[cfg(feature = "python-config")]
use crate::python;
use crate::routes;
use crate::routes::realtime::pool::{PoolConfig, RealtimePool, upstream_key};
use crate::state::AppState;
use litellm_core::callbacks::custom_logger::CustomLogger;
use litellm_core::router::{Deployment, LiteLLMParams, Router};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ServerOptions {
    pub config_path: Option<String>,
}

impl ServerOptions {
    pub fn from_env() -> Self {
        Self {
            config_path: std::env::var("LITELLM_CONFIG_PATH").ok(),
        }
    }
}

pub async fn run(options: ServerOptions) -> std::io::Result<()> {
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

    let proxy_logger: Arc<LiteLLMPythonProxyAPILogger> = LiteLLMPythonProxyAPILogger::from_env();
    let loggers: Vec<Arc<dyn CustomLogger>> = vec![proxy_logger];
    let router: Arc<Router> = Arc::new(build_router(options.config_path.as_deref()));
    let pool_config: PoolConfig = PoolConfig::from_env();
    let realtime_pool: Arc<RealtimePool> = RealtimePool::spawn(pool_config);
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

    let state: AppState = AppState {
        router,
        master_key,
        loggers: Arc::new(loggers),
        realtime_pool,
    };
    let host: String = std::env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port: u16 = resolve_port();
    let listener: tokio::net::TcpListener =
        tokio::net::TcpListener::bind((host.as_str(), port)).await?;
    eprintln!("litellm-ai-gateway listening on {host}:{port}");
    axum::serve(listener, routes::app(state)).await
}

fn register_deployments(router: &Router, pool: &RealtimePool) {
    for deployment in router.deployments() {
        let params: &LiteLLMParams = &deployment.litellm_params;
        let provider_model: &str = params
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

fn resolve_port() -> u16 {
    match std::env::var("PORT") {
        Ok(raw) => raw.parse().unwrap_or_else(|_| {
            eprintln!("warning: PORT={raw:?} is not a valid port; using {DEFAULT_PORT}");
            DEFAULT_PORT
        }),
        Err(_) => DEFAULT_PORT,
    }
}

fn build_router(config_path: Option<&str>) -> Router {
    #[cfg(feature = "python-config")]
    if let Some(config_path) = config_path {
        match python::config::load_router_from_config(config_path) {
            Ok(router) => {
                eprintln!("loaded model_list from {config_path} via python config reader");
                return router;
            }
            Err(err) => {
                eprintln!("config load failed ({err}); falling back to env deployment");
            }
        }
    }
    #[cfg(not(feature = "python-config"))]
    if config_path.is_some() {
        eprintln!("warning: config path ignored because python-config support is disabled");
    }
    build_router_from_env()
}

fn build_router_from_env() -> Router {
    let model: String =
        std::env::var("OPENAI_REALTIME_MODEL").unwrap_or_else(|_| "gpt-realtime".to_string());
    let api_key: Option<String> = std::env::var("OPENAI_API_KEY").ok();
    if api_key.is_none() {
        eprintln!(
            "warning: OPENAI_API_KEY is not set; realtime requests will fail with auth errors"
        );
    }
    let deployment: Deployment = Deployment {
        model_name: model.clone(),
        litellm_params: LiteLLMParams {
            model,
            api_key,
            api_base: None,
        },
    };
    Router::new(vec![deployment])
}
