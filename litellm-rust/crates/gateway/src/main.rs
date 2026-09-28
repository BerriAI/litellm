use std::{
    error::Error,
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
    let config = Config::load(settings.litellm_config)?;
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
    let listener = tokio::net::TcpListener::bind((settings.host.as_str(), settings.port)).await?;

    tracing::info!(address = %listener.local_addr()?, models = config.model_list.len(), log_level = %level, "gateway listening");

    axum::serve(listener, litellm_gateway::router(inference, &config, ui)).await?;
    Ok(())
}
