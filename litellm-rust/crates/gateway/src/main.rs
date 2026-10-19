use std::{
    error::Error,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use litellm_config::Config;
use litellm_gateway_auth::UiBackend;
use litellm_tracing::{Level, Logger, Metadata, Record, Sink};
use serde_json::json;

mod settings;

use settings::{Settings, UiSettings};

struct StderrSink {
    level: Level,
}

impl Sink for StderrSink {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        *metadata.level() <= self.level && metadata.target().starts_with("litellm")
    }

    fn emit(&self, record: &Record) {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        eprintln!(
            "{}",
            json!({
                "timestamp_ms": timestamp_ms,
                "level": record.metadata.level().as_str(),
                "target": record.metadata.target(),
                "message": record.message,
                "fields": record.fields,
            })
        );
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let settings: Settings = settings::from_iter(std::env::vars_os())?;
    let level = settings.log_level();
    Logger::new(StderrSink { level }).install_global()?;
    let config = Config::load(&settings.litellm_config)?;
    let inference = litellm_gateway::build_inference(&config)?;
    let ui = match std::env::var_os("LITELLM_UI_PATH") {
        Some(directory) => {
            let ui_settings: UiSettings = settings::from_iter(std::env::vars_os())?;
            let backend = UiBackend::new(ui_settings.ui_username, ui_settings.ui_password)?;
            Some(
                litellm_gateway_ui::router(
                    backend,
                    tower_sessions_moka_store::MokaStore::new(Some(10_000)),
                    ui_settings.litellm_ui_secure_cookies,
                )
                .merge(litellm_gateway_ui::dashboard_assets(
                    std::path::PathBuf::from(directory),
                )),
            )
        }
        None => None,
    };
    let shutdown = tokio_util::sync::CancellationToken::new();
    let _shutdown_guard = shutdown.clone().drop_guard();
    let mcp = litellm_gateway::build_mcp(
        &config,
        inference.secrets.clone(),
        shutdown.clone(),
        &inference.resources.pool,
        &inference.http,
    )
    .await?;
    let mcp_router = mcp.as_ref().map(|gateway| {
        let defaults = litellm_gateway_mcp::HttpConfig::default();
        litellm_gateway_mcp::router(
            gateway.operations.clone(),
            litellm_gateway_mcp::HttpConfig {
                allowed_hosts: config
                    .general_settings
                    .mcp_allowed_hosts
                    .as_deref()
                    .map(Vec::from)
                    .unwrap_or(defaults.allowed_hosts),
                allowed_origins: config.general_settings.mcp_allowed_origins.to_vec(),
                cancellation_token: shutdown.clone(),
                server_info: defaults.server_info,
            },
        )
    });
    let database = litellm_gateway::build_database(&config, inference.secrets.as_ref()).await?;
    let keys = database
        .clone()
        .map(|pool| litellm_gateway::build_keys_on(&config, pool))
        .transpose()?
        .map(|keys| litellm_gateway::key_cache(keys, &config));
    let virtual_keys = keys.is_some();
    let coordination =
        litellm_gateway::coordination::RedisSettings::resolve(&config, inference.secrets.as_ref())
            .await?
            .map(litellm_gateway::coordination::Coordination::new)
            .transpose()?;
    if let Some(coordination) = &coordination {
        inference
            .models
            .share_health(Arc::new(litellm_gateway::coordination::RedisHealth(
                coordination.clone(),
            )));
        coordination.start(inference.clone(), keys.clone(), shutdown.clone());
    }
    // Built-in hooks and observers are added here as the gateway gains them.
    let mut hooks: Vec<Arc<dyn litellm_gateway_hooks::Hook>> = Vec::new();
    if database.is_some() {
        hooks.push(Arc::new(litellm_gateway::budgets::Budgets::for_gateway(
            &config,
            &inference,
            coordination.clone(),
            database.clone(),
        )));
        hooks.push(Arc::new(
            litellm_gateway::rate_limits::RateLimiter::for_gateway(
                &config,
                coordination.clone(),
                database.clone(),
            ),
        ));
    }
    hooks.extend(litellm_gateway::request_hooks::from_config(&config)?);
    let mut observers: Vec<Arc<dyn litellm_gateway_hooks::Observer>> = Vec::new();
    let spend = database.zip(litellm_gateway::spend::SpendSettings::from_config(&config));
    let spend_tracking = spend.is_some();
    if let Some((pool, settings)) = spend {
        observers.push(litellm_gateway::spend::SpendWriter::start(pool, settings));
    }
    litellm_gateway::hooks::install(&inference, &config, hooks, observers).await?;
    let listener = tokio::net::TcpListener::bind((settings.host.as_str(), settings.port)).await?;
    let address = listener.local_addr()?;
    // Without TCP_NODELAY, Nagle holds small SSE chunks until the client's delayed ACK (~40 ms per stream).
    let listener = axum::serve::ListenerExt::tap_io(listener, |stream| {
        if let Err(error) = stream.set_nodelay(true) {
            tracing::warn!(%error, "failed to set TCP_NODELAY");
        }
    });

    let shared_state = coordination.is_some();
    tracing::info!(%address, models = config.model_list.len(), virtual_keys, shared_state, spend_tracking, log_level = %level, "gateway listening");

    let draining = litellm_gateway::health::Draining::default();
    let metrics = litellm_gateway::metrics::Metrics::new(&config, draining.flag());
    let metrics_auth = litellm_gateway::metrics_auth(&config, inference.secrets.clone());
    let app = litellm_gateway::with_ops(
        litellm_gateway::router_with_key_cache(inference.clone(), &config, ui, mcp_router, keys),
        metrics,
        draining.clone(),
        metrics_auth,
    );
    let result = litellm_gateway::drain::serve(
        listener,
        app,
        draining,
        settings.drain(),
        shutdown.clone(),
        shutdown_signal,
    )
    .await;
    shutdown.cancel();
    litellm_gateway::hooks::close(&inference).await;
    if let Some(mcp) = mcp {
        mcp.close().await;
    }
    result?;
    Ok(())
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut terminate) => {
                tokio::select! { _ = tokio::signal::ctrl_c() => (), _ = terminate.recv() => () }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
