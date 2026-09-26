use std::{
    error::Error,
    time::{SystemTime, UNIX_EPOCH},
};

use litellm_config::Config;
use litellm_tracing::{Level, Logger, Metadata, Record, Sink};
use serde_json::json;

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
    let level = std::env::var("RUST_LOG")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(Level::INFO);
    Logger::new(StderrSink { level }).install_global()?;
    let config_path = std::env::var("LITELLM_CONFIG").unwrap_or_else(|_| "config.yaml".into());
    let config = Config::load(config_path)?;
    let inference = litellm_gateway::build_inference(&config)?;
    let host = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".into());
    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "4000".into())
        .parse::<u16>()?;
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;

    tracing::info!(address = %listener.local_addr()?, models = config.model_list.len(), log_level = %level, "gateway listening");

    axum::serve(listener, litellm_gateway::router(inference, &config)).await?;
    Ok(())
}
