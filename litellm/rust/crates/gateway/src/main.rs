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
use litellm_gateway::{Admit, AppState, Live, Refused, Resources, Served, middleware::InFlight};
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
    let config = litellm_gateway::load_config(settings.litellm_config).await?;
    let state = AppState::build(&config, environment).await?;
    scrub_at_the_configs_key_length(key_length, &*state.environment);
    let admit = Arc::new(Admit::new(&config, environment, &state));
    let live = Arc::new(Live::new(served(
        state,
        Some(admit),
        &config,
        environment,
        &InFlight::new(),
    )));
    let app = alone(live.clone());
    let listener = tokio::net::TcpListener::bind((settings.host.as_str(), settings.port)).await?;

    tracing::info!(address = %listener.local_addr()?, models = config.model_list.len(), log_level = %log_level, "gateway listening");

    let served = axum::serve(listener, ServiceExt::<Request>::into_make_service(app))
        .with_graceful_shutdown(shutdown_signal())
        .await;
    live.current().state.close().await;
    Ok(served?)
}

/// The native app with nothing behind it. `admit` sees every request first, as it does in front
/// of Python, and a request it refuses is answered 501 with why, since nothing else can serve it.
fn alone(
    live: Arc<Live>,
) -> impl Service<Request, Response = Response, Error = Infallible, Future: Send + 'static>
+ Clone
+ Send
+ 'static {
    from_fn_with_state(live.clone(), admitted_or_not_served).layer(live.native())
}

async fn admitted_or_not_served(
    State(live): State<Arc<Live>>,
    request: Request,
    native: Next,
) -> Response {
    match live.admitted(request).await {
        Ok(request) => native.run(request).await,
        Err((_, why)) => not_served(why),
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

/// The native app's state for the config Python runs with, and which requests it answers. When
/// that config cannot be loaded, or the native app cannot be built from it, the state is built
/// from an empty config and Python answers every request, because failing here would stop Python
/// too.
async fn native_state(
    loaded: Option<Config>,
    environment: &Environment,
) -> Result<(Config, AppState, Option<Arc<Admit>>), litellm_gateway::Error> {
    if let Some(config) = loaded {
        match admitted_state(&config, environment).await {
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
) -> Result<(AppState, Admit), litellm_gateway::Error> {
    let resources = Resources::build(config)?;
    let state = AppState::over(config, environment, &resources).await?;
    let admit = Admit::new(config, environment, &state);
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
    let live = Arc::new(Live::new(served(
        state,
        admit,
        &config,
        environment,
        &InFlight::new(),
    )));
    let admitting = live.clone();
    let ran = litellm_gateway_forward::run(
        forward,
        live.clone().native(),
        move |request| admitting.clone().admitted(request),
        async move {
            tokio::select! { () = shutdown_signal() => (), () = stdin_closed => () }
        },
    )
    .await;
    live.current().state.close().await;
    Ok(ran?)
}

/// What both modes serve from `config`: the native app built from `state`, with `admit` deciding
/// which requests it answers. `in_flight` counts requests across every config the process serves.
fn served(
    state: AppState,
    admit: Option<Arc<Admit>>,
    config: &Config,
    environment: &Environment,
    in_flight: &InFlight,
) -> Served {
    let native = native(state.clone(), config, environment, in_flight);
    Served::new(state, admit, native)
}

/// The native app, which both modes build here so they cannot differ in its layers. `admit` goes
/// around all of it in both.
fn native(
    state: AppState,
    config: &Config,
    environment: &Environment,
    in_flight: &InFlight,
) -> impl Service<Request, Response = Response, Error = Infallible, Future: Send + 'static>
+ Clone
+ Send
+ Sync
+ 'static
+ use<> {
    use litellm_gateway::middleware::{global, root_path};
    let entitlements = state.entitlements.clone();
    let settings = middleware_settings(config, environment, &*state.environment);
    root_path::layer(root_path::Settings::from_environment(environment)).layer(global(
        litellm_gateway::router(state, config),
        &settings,
        entitlements,
        in_flight,
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

    fn live(state: AppState, admit: Option<Arc<Admit>>, config: &Config) -> Arc<Live> {
        let served = served(
            state,
            admit,
            config,
            &Environment::default(),
            &InFlight::new(),
        );
        Arc::new(Live::new(served))
    }

    /// Why a master key caller's `/v1/messages` request is left to Python.
    async fn refused(state: &AppState, admit: Option<Arc<Admit>>) -> Option<&'static str> {
        let request = Request::post("/v1/messages")
            .header("authorization", "Bearer gateway-key")
            .body(axum::body::Body::empty())
            .unwrap();
        live(state.clone(), admit, &Config::default())
            .admitted(request)
            .await
            .err()
            .map(|(_, why)| why.into())
    }

    /// Why `live` leaves a master key caller's `/v1/messages` request for `model` to Python.
    async fn refused_for(live: &Arc<Live>, model: &str) -> Option<&'static str> {
        let request = Request::post("/v1/messages")
            .header("authorization", "Bearer gateway-key")
            .body(axum::body::Body::from(format!(
                r#"{{"model":"{model}","max_tokens":8,"messages":[]}}"#
            )))
            .unwrap();
        live.clone()
            .admitted(request)
            .await
            .err()
            .map(|(_, why)| why.into())
    }

    /// What the gateway in front of Python serves from the config file holding `yaml`.
    async fn from_the_file(yaml_text: &str) -> Served {
        let file = yaml(yaml_text);
        let environment = Environment::default();
        let (config, state, admit) =
            native_state(pythons_config(Some(file.path())).await, &environment)
                .await
                .unwrap();
        served(state, admit, &config, &environment, &InFlight::new())
    }

    fn deploying(model: &str) -> String {
        format!(
            "model_list:\n  - model_name: {model}\n    litellm_params:\n      model: anthropic/claude-haiku-5-5\n      api_key: provider-key\ngeneral_settings:\n  master_key: gateway-key\n"
        )
    }

    #[rstest]
    #[case::a_model_the_new_file_adds("sonnet", (Some("model"), None))]
    #[case::a_model_the_new_file_drops("claude", (None, Some("model")))]
    #[case::a_model_neither_file_deploys("gpt", (Some("model"), Some("model")))]
    #[tokio::test]
    async fn after_a_swap_admit_and_the_router_both_follow_the_new_config(
        #[case] model: &str,
        #[case] expected: (Option<&str>, Option<&str>),
    ) {
        litellm_gateway::install_crypto_provider();
        let live = Arc::new(Live::new(from_the_file(&deploying("claude")).await));
        let before = refused_for(&live, model).await;

        let next = from_the_file(&deploying("sonnet")).await;
        let replaced = live
            .swap(|_| async { Ok::<_, Infallible>(next) })
            .await
            .unwrap();
        let after = refused_for(&live, model).await;

        let current = live.current();
        let routed = current.state.inference.router.reach(model).is_some();
        assert_eq!(
            (before, after, routed),
            (expected.0, expected.1, after.is_none())
        );
        replaced.state.close().await;
        current.state.close().await;
    }

    #[rstest]
    #[case::a_setting_the_native_app_does_not_honor(
        "litellm_settings:\n  cache: true\n",
        "settings"
    )]
    #[case::a_config_the_native_app_cannot_be_built_from(
        "general_settings:\n  coordination_redis:\n    host: coord\n    port: not-a-port\n",
        "configuration"
    )]
    #[tokio::test]
    async fn python_answers_every_request_of_a_config_the_native_app_does_not_run(
        #[case] settings: &str,
        #[case] why: &str,
    ) {
        litellm_gateway::install_crypto_provider();
        let config = Config::from_yaml(&format!(
            "model_list:\n  - model_name: claude\n    litellm_params:\n      model: anthropic/claude-haiku-5-5\n{settings}"
        ))
        .unwrap();

        let (_, state, admit) = native_state(Some(config), &Environment::default())
            .await
            .unwrap();

        assert_eq!(refused(&state, admit).await, Some(why));
        state.close().await;
    }

    #[rstest]
    #[case::an_admitted_request("/v1/messages", (StatusCode::OK, None))]
    #[case::a_refused_request(
        "/v1/chat/completions",
        (StatusCode::NOT_IMPLEMENTED, Some("route"))
    )]
    #[tokio::test]
    async fn standalone_answers_a_request_admit_refuses_with_a_501_that_says_why(
        #[case] path: &str,
        #[case] expected: (StatusCode, Option<&str>),
    ) {
        litellm_gateway::install_crypto_provider();
        let config = Config::from_yaml(
            "model_list:\n  - model_name: claude\n    litellm_params:\n      model: anthropic/claude-haiku-5-5\n      api_key: provider-key\ngeneral_settings:\n  master_key: gateway-key\n",
        )
        .unwrap();
        let (_, state, admit) = native_state(Some(config), &Environment::default())
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

        let live = Arc::new(Live::new(Served::new(state.clone(), admit, native)));
        let response = tower::ServiceExt::oneshot(alone(live), request)
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

        assert_eq!(refused(&state, admit).await, Some("configuration"));
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

        let response = native(
            state.clone(),
            &config,
            &Environment::default(),
            &InFlight::new(),
        )
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
