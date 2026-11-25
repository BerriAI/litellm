use std::{
    convert::Infallible,
    error::Error,
    path::Path,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, ServiceExt,
    extract::{Request, State},
    http::StatusCode,
    middleware::{Next, from_fn_with_state},
    response::{IntoResponse, Response},
};
use litellm_config::{Config, Environment};
use litellm_env::Lookup;
use litellm_gateway::{Admit, AppState, Loaded, Refused, Resources, StoredConfig};
use litellm_tracing::{KeyLength, Level, Logger, Metadata, Record, Scrubbed, Sink};
use serde_json::json;
use tower::{Layer, Service};

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
    let scrubbed = Scrubbed::new(
        StderrSink {
            level: process.log_level,
        },
        process.minimum_custom_key_length,
    );
    let key_length = scrubbed.key_length();
    Logger::new(scrubbed).install_global()?;
    match process.mode {
        Mode::InFrontOfPython(forward) => {
            in_front_of_python(forward, &process.environment, &key_length).await
        }
        Mode::Standalone(settings) => {
            standalone(
                settings,
                &process.environment,
                process.log_level,
                &key_length,
            )
            .await
        }
    }
}

/// The log is installed before the config is read, so it starts with the process's key length. The config's
/// environment can set it too, and authentication reads it from there, so the log follows once the config is in.
fn scrub_at_the_configs_key_length(key_length: &KeyLength, environment: &dyn Lookup) {
    if let Some(length) = environment
        .get("MINIMUM_CUSTOM_KEY_LENGTH")
        .as_deref()
        .and_then(process::key_length)
    {
        key_length.set(length);
    }
}

/// Serves natively the requests the gateway admits, on its own listener with nothing behind it.
async fn standalone(
    settings: Standalone,
    environment: &Environment,
    log_level: Level,
    key_length: &KeyLength,
) -> Result<(), Box<dyn Error>> {
    let Loaded { config, stored } = litellm_gateway::load_config(settings.litellm_config).await?;
    let state = AppState::build(&config, environment).await?;
    scrub_at_the_configs_key_length(key_length, &*state.environment);
    let admit = Arc::new(Admit::new(&config, &state, stored));
    let app = alone(native(state.clone(), &config, environment), admit);
    let listener = tokio::net::TcpListener::bind((settings.host.as_str(), settings.port)).await?;

    tracing::info!(address = %listener.local_addr()?, models = config.model_list.len(), log_level = %log_level, "gateway listening");

    let served = axum::serve(listener, ServiceExt::<Request>::into_make_service(app))
        .with_graceful_shutdown(shutdown_signal())
        .await;
    state.close().await;
    Ok(served?)
}

/// `native` with nothing behind it. `admit` sees every request first, as it does in front of
/// Python, and a request it refuses is answered 501 with why, since nothing else can serve it.
fn alone<S>(
    native: S,
    admit: Arc<Admit>,
) -> impl Service<Request, Response = Response, Error = Infallible, Future: Send + 'static>
+ Clone
+ Send
+ 'static
where
    S: Service<Request, Response = Response, Error = Infallible> + Clone + Send + Sync + 'static,
    S::Future: Send + 'static,
{
    from_fn_with_state(admit, admitted_or_not_served).layer(native)
}

async fn admitted_or_not_served(
    State(admit): State<Arc<Admit>>,
    request: Request,
    native: Next,
) -> Response {
    match admit.admit(request).await {
        Ok(request) => native.run(request).await,
        Err((_, reason)) => not_served(reason.why),
    }
}

fn not_served(why: Refused) -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({"error": {
            "message": why.to_string(),
            "type": "not_implemented",
            "code": <&str>::from(why),
        }})),
    )
        .into_response()
}

/// The config Python runs with. `None` when Python loads no local file, and when the gateway cannot
/// load that file, because failing here would stop Python too.
async fn pythons_config(path: Option<&Path>) -> Option<Loaded> {
    match litellm_gateway::load_config(path?).await {
        Ok(loaded) => {
            tracing::info!(
                models = loaded.config.model_list.len(),
                stored = ?loaded.stored,
                "loaded the config file python runs with"
            );
            Some(loaded)
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

/// Python answers every request when the gateway is not running the config Python runs with.
async fn admitted(
    admit: Option<Arc<Admit>>,
    request: Request,
) -> Result<Request, (Box<Request>, Refused)> {
    match admit {
        Some(admit) => admit
            .admit(request)
            .await
            .map_err(|(request, reason)| (request, reason.why)),
        None => Err((Box::new(request), Refused::Configuration)),
    }
}

/// The native app's state for the config Python runs with, and which requests it answers. When
/// that config cannot be loaded, or the native app cannot be built from it, the state is built
/// from an empty config and Python answers every request, because failing here would stop Python
/// too.
async fn native_state(
    loaded: Option<Loaded>,
    environment: &Environment,
) -> Result<(Config, AppState, Option<Arc<Admit>>), litellm_gateway::Error> {
    if let Some(Loaded { config, stored }) = loaded {
        match admitted_state(&config, environment, stored).await {
            Ok((state, admit)) => return Ok((config, state, Some(Arc::new(admit)))),
            Err(error) => tracing::error!(
                error = %litellm_error::Report::new(&error),
                "could not build the native app from the config python runs with, so python answers every request"
            ),
        }
    }
    let config = Config::default();
    let state = AppState::build(&config, environment).await?;
    Ok((config, state, None))
}

async fn admitted_state(
    config: &Config,
    environment: &Environment,
    stored: StoredConfig,
) -> Result<(AppState, Admit), litellm_gateway::Error> {
    let resources = Resources::build(config)?;
    let state = AppState::over(config, environment, &resources).await?;
    let admit = Admit::new(config, &state, stored);
    Ok((state, admit))
}

/// Serves natively the requests the gateway admits, in front of the Python proxy, which answers
/// everything else.
async fn in_front_of_python(
    forward: litellm_gateway_forward::Settings,
    environment: &Environment,
    key_length: &KeyLength,
) -> Result<(), Box<dyn Error>> {
    let stdin_closed = litellm_gateway_forward::input_closed(std::io::stdin());
    let loaded = pythons_config(forward.config.as_deref()).await;
    let (config, state, admit) = native_state(loaded, environment).await?;
    scrub_at_the_configs_key_length(key_length, &*state.environment);
    let ran = litellm_gateway_forward::run(
        forward,
        native(state.clone(), &config, environment),
        move |request| admitted(admit.clone(), request),
        async move {
            tokio::select! { () = shutdown_signal() => (), () = stdin_closed => () }
        },
    )
    .await;
    state.close().await;
    Ok(ran?)
}

/// The native app, which both modes build here so they cannot differ in its layers. `admit` goes
/// around all of it in both.
fn native(
    state: AppState,
    config: &Config,
    environment: &Environment,
) -> impl Service<Request, Response = Response, Error = Infallible, Future: Send + 'static>
+ Clone
+ Send
+ Sync
+ 'static
+ use<> {
    use litellm_gateway::middleware::{InFlight, global, root_path};
    let entitlements = state.entitlements.clone();
    let requests = state.gateway_requests.clone();
    let settings = middleware_settings(config, environment, &*state.environment);
    root_path::layer(root_path::Settings::from_environment(environment)).layer(global(
        litellm_gateway::router(state, config),
        &settings,
        entitlements,
        &InFlight::new(),
        requests,
    ))
}

fn middleware_settings(
    config: &Config,
    environment: &Environment,
    config_environment: &dyn Lookup,
) -> litellm_gateway::middleware::Settings {
    use litellm_gateway::middleware::{
        admission_control, body_limit, cors, decompression, security_headers,
    };
    litellm_gateway::middleware::Settings {
        admission_control: admission_control::Settings::from_general(&config.general_settings),
        security_headers: security_headers::Settings::from_environment(config_environment),
        cors: cors::Settings::from_environment(environment),
        body_limit: body_limit::Settings::from_general(&config.general_settings),
        decompression: decompression::Settings::from_environment(environment),
    }
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

        let loaded = pythons_config(Some(file.path())).await.unwrap();

        assert_eq!(loaded.config.model_list.len(), 1);
        assert_eq!(loaded.stored, StoredConfig::Unused);
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

    /// Why a master key caller's `/v1/messages` request is left to Python.
    async fn refused(admit: Option<Arc<Admit>>) -> Option<&'static str> {
        let request = Request::post("/v1/messages")
            .header("authorization", "Bearer gateway-key")
            .body(axum::body::Body::empty())
            .unwrap();
        admitted(admit, request)
            .await
            .err()
            .map(|(_, why)| why.into())
    }

    #[rstest]
    #[case::a_setting_the_native_app_does_not_honor(
        "litellm_settings:\n  cache: true\n",
        StoredConfig::Unused,
        "settings"
    )]
    #[case::a_config_the_native_app_cannot_be_built_from(
        "general_settings:\n  coordination_redis:\n    host: coord\n    port: not-a-port\n",
        StoredConfig::Unused,
        "configuration"
    )]
    #[case::a_stored_config_the_gateway_could_not_read(
        "general_settings:\n  store_model_in_db: true\n",
        StoredConfig::Unread,
        "db_config_not_loaded"
    )]
    #[tokio::test]
    async fn python_answers_every_request_of_a_config_the_native_app_does_not_run(
        #[case] settings: &str,
        #[case] stored: StoredConfig,
        #[case] why: &str,
    ) {
        litellm_gateway::install_crypto_provider();
        let config = Config::from_yaml(&format!(
            "model_list:\n  - model_name: claude\n    litellm_params:\n      model: anthropic/claude-haiku-5-5\n{settings}"
        ))
        .unwrap();

        let (_, state, admit) =
            native_state(Some(Loaded { config, stored }), &Environment::default())
                .await
                .unwrap();

        assert_eq!(refused(admit).await, Some(why));
        state.close().await;
    }

    #[rstest]
    #[case::an_admitted_request("/v1/messages", StoredConfig::Unused, (StatusCode::OK, None))]
    #[case::a_stored_config_layered_in(
        "/v1/messages",
        StoredConfig::Loaded,
        (StatusCode::OK, None)
    )]
    #[case::a_refused_request(
        "/v1/chat/completions",
        StoredConfig::Unused,
        (StatusCode::NOT_IMPLEMENTED, Some("route"))
    )]
    #[case::a_stored_config_the_gateway_could_not_read(
        "/v1/messages",
        StoredConfig::Unread,
        (StatusCode::NOT_IMPLEMENTED, Some("db_config_not_loaded"))
    )]
    #[tokio::test]
    async fn standalone_answers_a_request_admit_refuses_with_a_501_that_says_why(
        #[case] path: &str,
        #[case] stored: StoredConfig,
        #[case] expected: (StatusCode, Option<&str>),
    ) {
        litellm_gateway::install_crypto_provider();
        let config = Config::from_yaml(
            "model_list:\n  - model_name: claude\n    litellm_params:\n      model: anthropic/claude-haiku-5-5\n      api_key: provider-key\ngeneral_settings:\n  master_key: gateway-key\n  supported_db_objects: [models]\n",
        )
        .unwrap();
        let (_, state, admit) =
            native_state(Some(Loaded { config, stored }), &Environment::default())
                .await
                .unwrap();
        let native = tower::service_fn(|_: Request| async {
            Ok::<_, Infallible>(Response::new(axum::body::Body::empty()))
        });
        let request = Request::post(path)
            .header("authorization", "Bearer gateway-key")
            .body(axum::body::Body::from(
                r#"{"model":"claude","max_tokens":8,"messages":[]}"#,
            ))
            .unwrap();

        let response = tower::ServiceExt::oneshot(alone(native, admit.unwrap()), request)
            .await
            .unwrap();

        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let code = serde_json::from_slice::<serde_json::Value>(&body)
            .ok()
            .and_then(|body| body["error"]["code"].as_str().map(str::to_owned));
        assert_eq!((status, code.as_deref()), expected);
        state.close().await;
    }

    #[rstest]
    #[tokio::test]
    async fn without_a_config_python_answers_every_request() {
        litellm_gateway::install_crypto_provider();

        let (_, state, admit) = native_state(None, &Environment::default()).await.unwrap();

        assert_eq!(refused(admit).await, Some("configuration"));
        state.close().await;
    }

    struct Lines(std::sync::mpsc::Sender<String>);

    impl Sink for Lines {
        fn enabled(&self, _: &Metadata<'_>) -> bool {
            true
        }

        fn emit(&self, record: &Record) {
            self.0.send(record.message.clone()).unwrap();
        }
    }

    #[rstest]
    #[case::unset("model_list: []\n", "key sk-abcde rejected")]
    #[case::config_environment_variables(
        "model_list: []\nenvironment_variables:\n  MINIMUM_CUSTOM_KEY_LENGTH: \"8\"\n",
        "key REDACTED rejected"
    )]
    #[tokio::test]
    async fn the_log_scrubs_keys_at_the_length_the_config_exports(
        #[case] yaml: &str,
        #[case] logged: &str,
    ) {
        litellm_gateway::install_crypto_provider();
        let config = Config::from_yaml(yaml).unwrap();
        let state = AppState::build(&config, &Environment::default())
            .await
            .unwrap();
        let (sender, lines) = std::sync::mpsc::channel();
        let scrubbed = Scrubbed::new(Lines(sender), 16);
        let key_length = scrubbed.key_length();
        let logger = Logger::new(scrubbed);

        scrub_at_the_configs_key_length(&key_length, &*state.environment);
        logger.scope(|| tracing::warn!("key sk-abcde rejected"));

        state.close().await;
        assert_eq!(lines.try_recv().unwrap(), logged);
    }

    #[rstest]
    #[case::unset("model_list: []\n", None)]
    #[case::config_environment_variables(
        "model_list: []\nenvironment_variables:\n  LITELLM_ENABLE_HSTS: \"true\"\n",
        Some("max-age=31536000; includeSubDomains")
    )]
    #[tokio::test]
    async fn hsts_is_read_after_the_config_exports_its_environment_variables(
        #[case] yaml: &str,
        #[case] expected: Option<&str>,
    ) {
        use tower::ServiceExt;
        litellm_gateway::install_crypto_provider();
        let config = Config::from_yaml(yaml).unwrap();
        let state = AppState::build(&config, &Environment::default())
            .await
            .unwrap();

        let response = native(state.clone(), &config, &Environment::default())
            .oneshot(
                Request::get("/health/readiness")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        state.close().await;
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::STRICT_TRANSPORT_SECURITY)
                .map(|value| value.to_str().unwrap()),
            expected
        );
    }
}
