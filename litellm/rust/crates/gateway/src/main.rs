use std::{
    convert::Infallible,
    error::Error,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{ServiceExt, extract::Request, response::Response};
use litellm_config::{Config, Environment};
use litellm_gateway::{Admit, AppState};
use litellm_gateway_inference::Gateway;
use litellm_license::Entitlements;
use litellm_tracing::{Level, Logger, Metadata, Record, Sink};
use serde_json::json;
use tower::Service;

mod process;

use process::{Mode, Process, Standalone};

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
    let process = Process::from_env(std::env::vars_os())?;
    Logger::new(StderrSink {
        level: process.log_level,
    })
    .install_global()?;
    match process.mode {
        Mode::InFrontOfPython(forward) => in_front_of_python(forward, &process.environment).await,
        Mode::Standalone(settings) => {
            standalone(settings, &process.environment, process.log_level).await
        }
    }
}

/// Serves every request natively on its own listener.
async fn standalone(
    settings: Standalone,
    environment: &Environment,
    log_level: Level,
) -> Result<(), Box<dyn Error>> {
    let config = litellm_gateway::load_config(settings.litellm_config).await?;
    let state = AppState::build(&config).await?;
    let entitlements = build_license(&state.inference, &config, environment).await?;
    tracing::info!(
        enterprise = entitlements.is_enterprise(),
        "license verified"
    );
    let app = native(state.clone(), &config);
    let listener = tokio::net::TcpListener::bind((settings.host.as_str(), settings.port)).await?;

    tracing::info!(address = %listener.local_addr()?, models = config.model_list.len(), log_level = %log_level, "gateway listening");

    let served = axum::serve(listener, ServiceExt::<Request>::into_make_service(app))
        .with_graceful_shutdown(shutdown_signal())
        .await;
    state.close().await;
    Ok(served?)
}

/// The config Python runs with. `None` when Python loads no local file, and when the gateway cannot
/// load that file, because failing here would stop Python too.
async fn pythons_config(path: Option<&Path>) -> Option<Config> {
    match litellm_gateway::load_config(path?).await {
        Ok(config) => {
            tracing::info!(
                models = config.model_list.len(),
                "loaded the config file python runs with"
            );
            Some(config)
        }
        Err(error) => {
            tracing::error!(
                %error,
                "could not load the config file python runs with, so python answers every request"
            );
            None
        }
    }
}

/// Serves natively the requests the gateway admits, in front of the Python proxy, which answers
/// everything else.
async fn in_front_of_python(
    forward: litellm_gateway_forward::Settings,
    environment: &Environment,
) -> Result<(), Box<dyn Error>> {
    let stdin_closed = litellm_gateway_forward::input_closed(std::io::stdin());
    let loaded = pythons_config(forward.config.as_deref()).await;
    let admit = loaded.as_ref().map_or_else(Admit::unconfigured, Admit::new);
    let config = loaded.unwrap_or_default();
    let state = AppState::build(&config).await?;
    let entitlements = build_license(&state.inference, &config, environment).await?;
    tracing::info!(
        enterprise = entitlements.is_enterprise(),
        "license verified"
    );
    let ran = litellm_gateway_forward::run(
        forward,
        native(state.clone(), &config),
        move |request| admit.clone().admit(request),
        async move {
            tokio::select! { () = shutdown_signal() => (), () = stdin_closed => () }
        },
    )
    .await;
    state.close().await;
    Ok(ran?)
}

async fn build_license(
    inference: &Gateway,
    config: &Config,
    environment: &Environment,
) -> Result<Entitlements, litellm_gateway::Error> {
    litellm_gateway::build_license(
        config,
        environment,
        &inference.resources.pool,
        &inference.http,
    )
    .await
}

/// The native app, which both modes build here so they cannot differ in it. In front of Python,
/// `admit` goes around all of it.
fn native(
    state: AppState,
    config: &Config,
) -> impl Service<Request, Response = Response, Error = Infallible, Future: Send + 'static>
+ Clone
+ Send
+ Sync
+ 'static
+ use<> {
    litellm_gateway::router(state, config)
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

#[cfg(test)]
mod tests {
    use std::io::Write;

    use rstest::rstest;
    use tempfile::NamedTempFile;

    use super::*;

    fn yaml(contents: &str) -> NamedTempFile {
        let mut file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        file
    }

    #[rstest]
    #[tokio::test]
    async fn the_config_file_python_runs_with_is_loaded() {
        litellm_gateway::install_crypto_provider();
        let file = yaml(
            "model_list:\n  - model_name: claude\n    litellm_params:\n      model: anthropic/claude-haiku-5-5\n",
        );

        let config = pythons_config(Some(file.path())).await.unwrap();

        assert_eq!(config.model_list.len(), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn a_file_the_gateway_cannot_parse_is_not_loaded() {
        litellm_gateway::install_crypto_provider();
        let file = yaml("model_list: [unclosed\n");

        assert!(pythons_config(Some(file.path())).await.is_none());
    }

    #[rstest]
    #[case::no_local_file(None)]
    #[case::a_missing_file(Some("/nonexistent/litellm/config.yaml"))]
    #[tokio::test]
    async fn without_a_file_to_load_there_is_no_config(#[case] path: Option<&str>) {
        litellm_gateway::install_crypto_provider();

        assert!(pythons_config(path.map(Path::new)).await.is_none());
    }
}
