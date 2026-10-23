use std::{
    error::Error,
    time::{SystemTime, UNIX_EPOCH},
};

use litellm_config::Config;
use litellm_gateway_auth::UiBackend;
use litellm_tracing::{Level, Logger, Metadata, Record, Sink};
use serde_json::json;

mod settings;

use litellm_layered::Env;
use settings::{BootKeysPartial, Settings, UiKeysPartial, UiSettings};

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
    let env = Env::from_process();
    let settings = Settings::from(BootKeysPartial::from_env(&env)?);
    let level = settings.log_level();
    Logger::new(StderrSink { level }).install_global()?;
    let config = Config::load(settings.litellm_config)?;
    let inference = litellm_gateway::build_inference(&config, &env)?;
    let ui = match &settings.ui_path {
        Some(directory) => {
            let ui_settings = UiSettings::try_from(UiKeysPartial::from_env(&env)?)?;
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
    let signal_shutdown = shutdown.clone();
    tokio::spawn(async move {
        shutdown_signal().await;
        signal_shutdown.cancel();
    });
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
    let listener = tokio::net::TcpListener::bind((settings.host.as_str(), settings.port)).await?;

    tracing::info!(address = %listener.local_addr()?, models = config.model_list.len(), log_level = %level, "gateway listening");

    let result = axum::serve(
        listener,
        litellm_gateway::router(inference, &config, ui, mcp_router),
    )
    .with_graceful_shutdown(shutdown.clone().cancelled_owned())
    .await;
    shutdown.cancel();
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
