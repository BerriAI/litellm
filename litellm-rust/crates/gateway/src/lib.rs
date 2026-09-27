mod auth;
mod error;
mod secrets;
pub use error::Error;

use std::{sync::Arc, time::Instant};

use axum::{
    Router,
    body::{Body, Bytes},
    extract::Request,
    middleware::Next,
    response::Response,
};
use http_body_util::BodyExt;

use litellm_config::Config;
use litellm_core::{CoreClient, resources::CoreResources};
use litellm_gateway_auth::Auth;
use litellm_gateway_inference::{Gateway, ModelRouter};
use litellm_http::{
    ClientVariant, HttpClientPool, HttpSettings, HttpSettingsLayer, Resolution, SslVerify,
    media::PublicDnsResolver,
};
use litellm_secrets::source::EnvironmentSecrets;
use litellm_tracing::ByteChunk;
use uuid::Uuid;

pub fn build_inference(config: &Config) -> Result<Arc<Gateway>, Error> {
    let pool = Arc::new(HttpClientPool::new(Arc::new(PublicDnsResolver)));
    let environment = secrets::environment_values(&config.environment_variables)?;
    let lookup = |name: &str| {
        environment
            .get(name)
            .map(|value| value.expose().to_owned())
            .or_else(|| std::env::var(name).ok())
    };
    let settings = &config.litellm_settings;
    let http_settings = HttpSettings::from_layers([
        HttpSettingsLayer::from_environment(&lookup),
        HttpSettingsLayer {
            ssl_verify: settings.ssl_verify.as_ref().map(|value| match value {
                litellm_config::Flag::Boolean(true) => SslVerify::Enabled,
                litellm_config::Flag::Boolean(false) => SslVerify::Disabled,
                litellm_config::Flag::String(value) => SslVerify::parse(value),
            }),
            ssl_certificate: settings.ssl_certificate.as_ref().map(Into::into),
            ssl_security_level: settings.ssl_security_level.clone(),
            ssl_ecdh_curve: settings.ssl_ecdh_curve.clone(),
            force_ipv4: settings.force_ipv4,
            http2: settings.http2,
            aiohttp_trust_env: settings.aiohttp_trust_env,
            disable_aiohttp_trust_env: settings.disable_aiohttp_trust_env,
            disable_aiohttp_transport: settings.disable_aiohttp_transport,
            ..Default::default()
        },
    ]);
    let http = Resolution::from(&http_settings).config;
    let client = pool.client(&http, ClientVariant::Provider)?;
    let secrets = Arc::new(secrets::ConfigSecrets::new(
        environment,
        Arc::new(EnvironmentSecrets::python_compatible(client)),
    ));
    let resources = CoreResources::new(pool);

    Ok(Arc::new(Gateway {
        core: CoreClient::new(resources, http, secrets),
        models: ModelRouter::from_model_list(&config.model_list),
    }))
}

pub async fn build_mcp(
    config: &Config,
    secrets: Arc<dyn litellm_secrets::source::SecretSource>,
    shutdown: tokio_util::sync::CancellationToken,
    pool: &HttpClientPool,
    http: &litellm_http::HttpClientConfig,
) -> Result<Option<litellm_gateway_mcp::ConfiguredGateway>, Error> {
    if !config.mcp_tools.is_empty() {
        return Err(Error::McpSetting("mcp_tools".into()));
    }
    if config.mcp_servers.is_empty() {
        return Ok(None);
    }
    if let Some(setting) = config
        .general_settings
        .additional_fields
        .keys()
        .chain(config.litellm_settings.additional_fields.keys())
        .find(|key| key.starts_with("mcp_"))
    {
        return Err(Error::McpSetting(setting.clone()));
    }
    if !config.guardrails.is_empty()
        || !config.policies.is_empty()
        || !config.policy_attachments.is_empty()
    {
        return Err(Error::McpSetting("guardrails or policies".into()));
    }
    Auth::from_config(config, secrets.clone())
        .validate()
        .await?;
    let client = pool.mcp_client(http)?;
    Ok(Some(
        litellm_gateway_mcp::ConfiguredGateway::connect(
            &config.mcp_servers,
            client,
            secrets.as_ref(),
            shutdown,
        )
        .await?,
    ))
}

pub fn router(
    inference: Arc<Gateway>,
    config: &Config,
    ui: Option<Router>,
    mcp: Option<Router>,
) -> Router {
    let auth = Auth::from_config(config, inference.core.secret_source().clone());
    let inference = litellm_gateway_inference::router(inference)
        .merge(mcp.unwrap_or_default())
        .route_layer(axum::middleware::from_fn(auth::bind_session_owner))
        .route_layer(axum::middleware::from_fn_with_state(
            auth,
            litellm_gateway_auth::authenticate,
        ))
        .layer(axum::middleware::from_fn(log_request));
    match ui {
        Some(ui) => inference.merge(ui),
        None => inference,
    }
}

async fn log_request(request: Request, next: Next) -> Response {
    let request_id = Uuid::new_v4().to_string();
    let log_body_chunks = tracing::enabled!(tracing::Level::DEBUG);
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    let started = Instant::now();
    let request = if log_body_chunks {
        request.map(|body| logged_body(body, request_id.clone(), "input"))
    } else {
        request
    };
    let response = next.run(request).await;
    tracing::info!(
        %request_id,
        %method,
        %path,
        status = response.status().as_u16(),
        time_to_headers_ms = started.elapsed().as_secs_f64() * 1000.0,
        "response headers"
    );
    if log_body_chunks {
        response.map(|body| logged_body(body, request_id, "output"))
    } else {
        response
    }
}

fn logged_body(body: Body, request_id: String, direction: &'static str) -> Body {
    Body::new(body.map_frame(move |frame| {
        if let Some(data) = frame.data_ref() {
            log_chunk(&request_id, direction, data);
        }
        frame
    }))
}

fn log_chunk(request_id: &str, direction: &str, data: &Bytes) {
    let chunk = ByteChunk::new(data);
    tracing::debug!(request_id, direction, encoding = chunk.encoding(), chunk = %chunk, "body chunk");
}

#[cfg(test)]
mod tests {
    use std::{convert::Infallible, sync::mpsc};

    use axum::{body::to_bytes, http::StatusCode, routing::post};
    use futures_util::stream;
    use litellm_tracing::{Logger, Metadata, Record, Sink};
    use rstest::rstest;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    use super::*;

    struct LogSink(mpsc::Sender<Value>);

    impl Sink for LogSink {
        fn enabled(&self, _: &Metadata<'_>) -> bool {
            true
        }

        fn emit(&self, record: &Record) {
            self.0
                .send(json!({"message": record.message, "fields": record.fields}))
                .unwrap();
        }
    }

    #[rstest]
    #[tokio::test]
    async fn logs_each_body_chunk_without_changing_streamed_bytes() {
        let app = Router::new()
            .route(
                "/stream",
                post(|_: Bytes| async {
                    (
                        StatusCode::OK,
                        Body::from_stream(stream::iter([
                            Ok::<_, Infallible>(Bytes::from_static(b"event: first\n\n")),
                            Ok(Bytes::from_static(b"event: second\n\n")),
                        ])),
                    )
                }),
            )
            .layer(axum::middleware::from_fn(log_request));
        let request_chunks = [
            Ok::<_, Infallible>(Bytes::from_static(b"hello")),
            Ok(Bytes::from_static(b" world")),
        ];
        let request = Request::post("/stream")
            .body(Body::from_stream(stream::iter(request_chunks)))
            .unwrap();
        let (sender, receiver) = mpsc::channel();
        let logger = Logger::new(LogSink(sender));

        let output = logger
            .instrument(async {
                let response = app.oneshot(request).await.unwrap();
                to_bytes(response.into_body(), 1024).await.unwrap()
            })
            .await;

        assert_eq!(output, "event: first\n\nevent: second\n\n");
        let records: Vec<Value> = receiver.try_iter().collect();
        assert_eq!(records.len(), 5);
        assert_eq!(records[0]["fields"]["chunk"], "hello");
        assert_eq!(records[1]["fields"]["chunk"], " world");
        assert_eq!(records[2]["fields"]["status"], 200);
        assert_eq!(records[3]["fields"]["chunk"], "event: first\n\n");
        assert_eq!(records[4]["fields"]["chunk"], "event: second\n\n");
        let request_id = &records[2]["fields"]["request_id"];
        assert!(request_id.as_str().is_some());
        assert!(
            records
                .iter()
                .all(|record| &record["fields"]["request_id"] == request_id)
        );
    }
}
