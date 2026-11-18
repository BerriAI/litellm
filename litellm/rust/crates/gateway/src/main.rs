use std::{
    error::Error,
    time::{SystemTime, UNIX_EPOCH},
};

use litellm_tracing::{Level, Logger, Metadata, Record, Sink};
use serde_json::json;

mod settings;

use settings::{LogSettings, Settings};

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
    litellm_gateway::install_crypto_provider();
    if let Some(forward) = litellm_gateway_forward::Settings::from_env()? {
        let log: LogSettings = settings::from_iter(std::env::vars_os())?;
        Logger::new(StderrSink {
            level: log.log_level(),
        })
        .install_global()?;
        let stdin_closed = litellm_gateway_forward::input_closed(std::io::stdin());
        litellm_gateway_forward::run(forward, async {
            tokio::select! { () = shutdown_signal() => (), () = stdin_closed => () }
        })
        .await?;
        return Ok(());
    }
    let settings: Settings = settings::from_iter(std::env::vars_os())?;
    let level = settings.log_level();
    Logger::new(StderrSink { level }).install_global()?;
    let config = litellm_gateway::load_config(settings.litellm_config).await?;
    let state = litellm_gateway::AppState::build(&config).await?;
    let listener = tokio::net::TcpListener::bind((settings.host.as_str(), settings.port)).await?;

    tracing::info!(address = %listener.local_addr()?, models = config.model_list.len(), log_level = %level, "gateway listening");

    let served = axum::serve(listener, litellm_gateway::router(state.clone(), &config))
        .with_graceful_shutdown(shutdown_signal())
        .await;
    state.close().await;
    Ok(served?)
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
    // Windows support exists for local CLI runs only, the proxy is never deployed on Windows. The supervisor
    // stops the gateway with CTRL_BREAK_EVENT, which has to drain the same way SIGTERM does.
    #[cfg(windows)]
    {
        match tokio::signal::windows::ctrl_break() {
            Ok(mut ctrl_break) => {
                tokio::select! { _ = tokio::signal::ctrl_c() => (), _ = ctrl_break.recv() => () }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
