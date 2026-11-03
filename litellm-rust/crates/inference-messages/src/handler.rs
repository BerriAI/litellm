use std::time::Duration;

use bytes::Bytes;
use futures_util::{FutureExt, StreamExt, TryStreamExt, stream::BoxStream};
use litellm_auth::AuthServices;
use litellm_secret_value::SecretValue;

use litellm_core_utils::{
    dot_notation_indexing::delete_nested_value,
    get_provider_specific_headers::get_provider_specific_headers, settings::Lookup,
};
use litellm_host::interceptors::{
    Cost, ExecutionFacts, Interceptors, ProviderIdentity, RawResponse, RequestContext,
    ResponseHead, ResultSource, WireRequest,
};
use litellm_llms::base_llm::{
    auth::{Authenticated, ValidatedEnvironment, resolve_auth},
    messages::{
        context::MessagesTransformContext,
        streaming::{self, ByteStream, StreamDecoder},
        transformation::BaseMessagesConfig,
    },
};
use litellm_llms_types::formats::messages::{
    MessagesOptionalParams, MessagesRequest, MessagesResponse,
};
use litellm_secrets::source::Secrets;
use litellm_tracing::ByteChunk;
use serde_json::Value;

use super::{
    Error, MessagesCall, MessagesCallResponse, MessagesRoute,
    common_utils::{MessagesProvider, merge_messages_header_sources, string_headers},
    constants::MESSAGES_TIMEOUT_SECS,
    route::Messages,
    types::invalid_request,
};
use litellm_inference::{caching::CallCache, context::CallContext, outbound::outbound_request};

struct ProviderCall {
    identity: ProviderIdentity,
    wire: WireRequest,
    provider: MessagesProvider,
    signer: Option<litellm_auth_aws::SigV4Signer>,
    timeout: Option<Duration>,
    stream: bool,
    /// Relay the stream in whole SSE frames, as Python's pass-through chunk processor does
    /// while it injects cost into streamed usage.
    whole_frames: bool,
}

impl ProviderCall {
    fn cacheable(&self) -> bool {
        self.signer.is_none()
    }
}

pub(super) async fn execute(
    route: &MessagesRoute,
    provider: MessagesProvider,
    model: String,
    call: MessagesCall,
    secrets: Secrets,
    context: &CallContext<'_, impl Interceptors<Error>>,
) -> Result<MessagesCallResponse, Error> {
    let request = provider_call(
        &route.auth,
        provider,
        model,
        call,
        secrets.as_ref(),
        context,
    )
    .boxed()
    .await?;
    let cache = CallCache::<Messages>::from_wire(
        route.cache.as_ref().filter(|_| request.cacheable()),
        context.cache,
        &request.identity,
        &request.wire,
    );
    let identity = request.identity.clone();
    let (output, source) = match cache.lookup().await {
        Some(hit) => hit,
        None => (
            call_provider(&route.http, request, context).await?,
            ResultSource::Provider,
        ),
    };
    let reported = match &output {
        MessagesCallResponse::Complete(response) => provider.config().reported_cost(response),
        MessagesCallResponse::Stream { .. } => None,
    };
    context
        .result_ready(ExecutionFacts {
            provider: identity,
            cost: Cost::settle(&source, reported),
            source: source.clone(),
        })
        .await?;
    Ok(cache.finish(output, &source).await)
}

/// The errors Python's `mock_response` raises for the `mock_response` text that names them.
const MOCKED_ERRORS: [&str; 3] = [
    "litellm.InternalServerError",
    "litellm.ContextWindowExceededError",
    "litellm.RateLimitError",
];

/// Python's Messages `mock_response`: the canned message carrying `mock` as its text, or the
/// error it names, answered before any provider work or logging update.
pub(super) fn mock_response(mock: &str) -> Result<MessagesResponse, Error> {
    if MOCKED_ERRORS.contains(&mock) {
        return Err(Error::Mocked(mock.to_string()));
    }
    let response = serde_json::from_value(serde_json::json!({
        "content": [{"text": mock, "type": "text"}],
        "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
        "model": "claude-sonnet-4-20250514",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "type": "message",
        "usage": {"input_tokens": 2095, "output_tokens": 503},
    }))
    .map_err(|err| {
        Error::invalid_response(litellm_llms::ErrorDetail::invalid("mock response", err))
    })?;
    Ok(response)
}

async fn provider_call(
    auth: &AuthServices,
    provider: MessagesProvider,
    model: String,
    call: MessagesCall,
    secrets: &(dyn Lookup + Send + Sync),
    context: &CallContext<'_, impl Interceptors<Error>>,
) -> Result<ProviderCall, Error> {
    let MessagesCall {
        body,
        api_key,
        api_base,
        litellm_params,
        extra_headers,
        provider_specific_header,
        timeout,
        shaping,
        ..
    } = call;
    let connection_params = litellm_router_types::LitellmParams {
        api_base: api_base.clone().or(litellm_params.api_base.clone()),
        ..litellm_params
    };
    let config = provider.config();
    let env_lookup = |key: &str| secrets.get(key);
    // Python takes `stream` from the caller, not from the body that `additional_drop_params`
    // may have trimmed, for the URL, the response handling and logging.
    let caller_stream = body.params.stream == Some(true);

    let sanitized = config.shape_request(
        MessagesRequest { model, ..body },
        shaping.settings.reasoning_auto_summary,
    )?;
    let transform_context =
        MessagesTransformContext::new(shaping.capabilities, shaping.settings.drop_params);
    let filtered = requested_optional_params(sanitized, provider, &transform_context)?;
    let trimmed =
        without_additional_drop_params(filtered, &shaping.settings.additional_drop_params)?;

    let scoped =
        get_provider_specific_headers(provider_specific_header.as_ref(), provider.as_str());
    let forwarded = string_headers(Some(merge_messages_header_sources([
        extra_headers.unwrap_or_default(),
        scoped,
    ])))?;
    let validated = config.validate_environment(
        forwarded,
        api_key.as_deref(),
        &trimmed.model,
        &trimmed.params,
        &connection_params,
        &env_lookup,
    )?;

    let body = config.request_body(
        config.transform_anthropic_messages_request(trimmed, &transform_context)?,
        &validated.headers,
        shaping.settings.drop_params,
    )?;
    let environment = ValidatedEnvironment {
        headers: config.request_headers(
            with_python_default_headers(validated.headers, config.default_headers()),
            &body,
        ),
        auth: validated.auth,
    };

    let url = config.get_complete_url(
        api_base.as_deref(),
        &body.model,
        &connection_params,
        caller_stream,
        &env_lookup,
    )?;

    let request_context = RequestContext {
        model: body.model.clone(),
        custom_llm_provider: provider.as_str().to_string(),
        optional_params: serde_json::to_value(&body.params).map_err(serialize_failure)?,
        secret_fields: Vec::new(),
        api_key: api_key.map(SecretValue::new),
    };
    let authenticated = resolve_auth(auth, environment, &|key| std::env::var(key).ok()).await?;
    let identity = ProviderIdentity {
        model: request_context.model.clone(),
        provider: request_context.custom_llm_provider.clone(),
    };
    let wire = context
        .interceptors
        .before_provider_request(
            WireRequest {
                url,
                headers: joined_header_values(authenticated.headers),
                body: config.wire_body(serde_json::to_value(&body).map_err(serialize_failure)?),
            },
            request_context,
        )
        .await?;
    let stream = match wire.body.get("stream") {
        None => caller_stream,
        Some(Value::Null) => false,
        Some(Value::Bool(stream)) => *stream,
        Some(value) => {
            return Err(Error::invalid_request(
                litellm_llms::ErrorDetail::InvalidValue {
                    field: "stream",
                    expected: "a boolean",
                    actual: value.clone(),
                },
            ));
        }
    };
    Ok(ProviderCall {
        identity,
        wire,
        provider,
        signer: authenticated.signer,
        timeout: timeout.or_else(|| shaping.settings.request_timeout()),
        stream,
        whole_frames: shaping.settings.include_cost_in_streaming_usage,
    })
}

/// Python sends its header dict through `httpx.Headers`, which folds names that differ only in
/// case into one lowercase header whose values are joined with `", "`.
fn joined_header_values(headers: Vec<(String, String)>) -> Vec<(String, String)> {
    let mut joined: Vec<(String, String)> = Vec::with_capacity(headers.len());
    let mut positions: std::collections::HashMap<String, usize> =
        std::collections::HashMap::with_capacity(headers.len());
    for (name, value) in headers {
        match positions.entry(name.to_ascii_lowercase()) {
            std::collections::hash_map::Entry::Occupied(position) => {
                let present = &mut joined[*position.get()].1;
                present.push_str(", ");
                present.push_str(&value);
            }
            std::collections::hash_map::Entry::Vacant(position) => {
                position.insert(joined.len());
                joined.push((name, value));
            }
        }
    }
    joined
}

/// Python's `finalize_messages_headers` adds a default only when the exact lowercase name is
/// absent, so a caller's `Anthropic-Version` is sent alongside the default, not instead of it.
fn with_python_default_headers(
    headers: Vec<(String, String)>,
    defaults: &[(&str, &str)],
) -> Vec<(String, String)> {
    let missing: Vec<(String, String)> = defaults
        .iter()
        .filter(|(name, _)| !headers.iter().any(|(present, _)| present == name))
        .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
        .collect();
    headers.into_iter().chain(missing).collect()
}

/// Python's `get_requested_anthropic_messages_optional_param`, which gates speed, service tier
/// and sampling params before `additional_drop_params` and the environment check run.
fn requested_optional_params(
    request: MessagesRequest,
    provider: MessagesProvider,
    context: &MessagesTransformContext,
) -> Result<MessagesRequest, Error> {
    let request = litellm_llms::anthropic::messages::transformation::drop_unsupported_params(
        request, context,
    )?;
    if provider.accepts_service_tier() {
        return Ok(request);
    }
    Ok(MessagesRequest {
        params: MessagesOptionalParams {
            service_tier: None,
            ..request.params
        },
        ..request
    })
}

fn without_additional_drop_params(
    request: MessagesRequest,
    paths: &[String],
) -> Result<MessagesRequest, Error> {
    if paths.is_empty() {
        return Ok(request);
    }
    let params = serde_json::to_value(request.params).map_err(invalid_request)?;
    let trimmed = paths
        .iter()
        .fold(params, |params, path| delete_nested_value(params, path));
    Ok(MessagesRequest {
        params: serde_json::from_value(trimmed).map_err(invalid_request)?,
        ..request
    })
}

async fn call_provider(
    http: &litellm_http::Client,
    request: ProviderCall,
    context: &CallContext<'_, impl Interceptors<Error>>,
) -> Result<MessagesCallResponse, Error> {
    let ProviderCall {
        identity,
        wire,
        provider,
        signer,
        timeout,
        stream,
        whole_frames,
    } = request;
    let provider_name = provider.as_str();
    let config = provider.config();
    let timeout = timeout.unwrap_or(Duration::from_secs(MESSAGES_TIMEOUT_SECS));
    let mut body = wire.body;
    let mut attempts = 1;
    let response = loop {
        log_request_body(provider_name, stream, &body);
        let response = send(
            http,
            Authenticated {
                headers: wire.headers.clone(),
                signer: signer.clone(),
            },
            &wire.url,
            &body,
            timeout,
            stream,
        )
        .await?;
        if response.status().is_success() {
            break response;
        }
        let rejection = provider_rejection(response, timeout).await?;
        // Python's `_async_post_anthropic_messages_with_http_error_retry`: a recoverable
        // rejection is sent once more with the body the config rewrites, under the same headers.
        let retried = (attempts < litellm_llms::base_llm::messages::http_error_retry::MAX_ATTEMPTS)
            .then(|| config.body_after_http_error(rejection.status, &rejection.text, &body))
            .flatten();
        match retried {
            Some(rewritten) => {
                tracing::debug!(
                    attempt = attempts + 1,
                    "anthropic messages: retrying after a recoverable HTTP error"
                );
                body = rewritten;
                attempts += 1;
            }
            None => return Err(rejection.into_error()),
        }
    };
    if stream {
        return Ok(streaming_response(
            response,
            config.stream_decoder(),
            provider_name,
            timeout,
            whole_frames,
        ));
    }
    let head = ResponseHead {
        status: response.status().as_u16(),
        headers: litellm_http::request::header_pairs(response.headers()),
    };
    let status = head.status;
    let text = response.text().await.map_err(litellm_http::Error::from)?;
    log_response_body(&text);
    context
        .raw_response_received(RawResponse::new(text.clone()).with_head(head))
        .await?;
    if serde_json::from_str::<serde::de::IgnoredAny>(&text).is_err() {
        // Python raises `AnthropicError(status_code=<the response's>, message=<its text>)` for a
        // body it cannot parse, and the public mapping classifies it by that status.
        return Err(litellm_http::Error::upstream(status, text).into());
    }
    decode_response(config, &identity.model, &text)
        .map(|message| MessagesCallResponse::Complete(Box::new(message)))
}

fn serialize_failure(err: serde_json::Error) -> Error {
    Error::invalid_request(litellm_llms::ErrorDetail::failed(
        "Anthropic messages request serialization",
        err,
    ))
}

/// Python's httpx timeout bounds each phase and each read, never the whole exchange. A complete
/// response keeps the client's overall deadline; a stream is bounded until its head arrives and
/// then per read (see [`timed_chunks`]), so a long stream that keeps sending is not cut off.
async fn send(
    http: &litellm_http::Client,
    authenticated: Authenticated,
    url: &str,
    body: &Value,
    timeout: Duration,
    stream: bool,
) -> Result<reqwest::Response, Error> {
    let request = outbound_request(authenticated, url, body, (!stream).then_some(timeout))?;
    if !stream {
        return Ok(litellm_inference::outbound::send(request, http).await?);
    }
    tokio::time::timeout(timeout, litellm_inference::outbound::send(request, http))
        .await
        .map_err(litellm_http::Error::read_timeout)?
        .map_err(Error::from)
}

/// The stream's body, each read bounded by `timeout` as httpx bounds each read.
fn timed_chunks(
    response: reqwest::Response,
    timeout: Duration,
) -> impl futures_util::Stream<Item = Result<Bytes, litellm_http::Error>> + Send + 'static {
    futures_util::stream::try_unfold(response, move |mut response| async move {
        let chunk = tokio::time::timeout(timeout, response.chunk())
            .await
            .map_err(litellm_http::Error::body_read_timeout)?
            .map_err(litellm_http::Error::from)?;
        Ok(chunk.map(|chunk| (chunk, response)))
    })
}

/// An upstream rejection whose body was read.
struct Rejection {
    status: u16,
    text: String,
    headers: Vec<(String, String)>,
}

impl Rejection {
    fn into_error(self) -> Error {
        log_error_body(self.status, &self.text);
        litellm_http::Error::upstream_with_headers(self.status, self.text, self.headers).into()
    }
}

/// The provider's error, its body read under `timeout`: a stream has no overall deadline once
/// its head arrives, so an error body that stalls would otherwise hold the call forever.
async fn provider_rejection(
    response: reqwest::Response,
    timeout: Duration,
) -> Result<Rejection, Error> {
    let status = response.status().as_u16();
    let headers = litellm_http::request::header_pairs(response.headers());
    match tokio::time::timeout(timeout, response.text()).await {
        Ok(Ok(text)) => Ok(Rejection {
            status,
            text,
            headers,
        }),
        Ok(Err(error)) => Err(litellm_http::Error::from(error).into()),
        Err(elapsed) => Err(litellm_http::Error::read_timeout(elapsed).into()),
    }
}

fn decode_response(
    config: &dyn BaseMessagesConfig,
    model: &str,
    text: &str,
) -> Result<MessagesResponse, Error> {
    let response = serde_json::from_str(text).map_err(|err| {
        Error::invalid_response(litellm_llms::ErrorDetail::invalid(
            "messages response JSON",
            err,
        ))
    })?;
    config
        .transform_anthropic_messages_response(model, response)
        .map_err(Error::from)
        .map_err(Error::after_send)
}

fn streaming_response(
    response: reqwest::Response,
    decoder: Option<StreamDecoder>,
    provider: &'static str,
    timeout: Duration,
    whole_frames: bool,
) -> MessagesCallResponse {
    let headers = response
        .headers()
        .iter()
        .filter_map(|(name, value)| Some((name.to_string(), value.to_str().ok()?.to_string())))
        .collect();
    let chunks = timed_chunks(response, timeout)
        .inspect_ok(move |chunk| log_chunk(provider, "provider_response", chunk));
    let chunks = match decoder {
        // A decoded stream is re-encoded one whole event per chunk, and Python's decoded
        // hosts relay it without the pass-through cost injection.
        None if whole_frames => in_whole_frames(chunks.map_err(Error::from).boxed()),
        None => chunks.map_err(Error::from).boxed(),
        Some(decode) => decoded_chunks(
            chunks.map_err(std::io::Error::other).boxed(),
            decode,
            provider,
        ),
    };
    MessagesCallResponse::Stream {
        head: super::route::MessagesStreamHead { headers },
        chunks,
    }
}

const SSE_FRAME_DELIMITERS: [&[u8]; 3] = [b"\r\n\r\n", b"\n\n", b"\r\r"];

/// Python's `split_complete_sse_frames`: the buffered bytes up to the last frame boundary, and
/// the unterminated tail. `pending[..scanned]` holds no whole delimiter, so only the bytes a new
/// delimiter could end in are searched.
fn split_complete_sse_frames_after(pending: &[u8], scanned: usize) -> (&[u8], &[u8]) {
    let longest = SSE_FRAME_DELIMITERS
        .iter()
        .map(|delimiter| delimiter.len())
        .max()
        .unwrap_or(0);
    let from = scanned.saturating_sub(longest - 1);
    let boundary_end = SSE_FRAME_DELIMITERS
        .iter()
        .filter_map(|delimiter| {
            pending[from..]
                .windows(delimiter.len())
                .rposition(|window| window == *delimiter)
                .map(|start| from + start + delimiter.len())
        })
        .max()
        .unwrap_or(0);
    pending.split_at(boundary_end)
}

/// The stream regrouped into whole SSE frames, as Python's pass-through chunk processor yields
/// it while cost injection is on: a read that completes no frame yields nothing, and an
/// unterminated tail is yielded once the upstream ends.
fn in_whole_frames(
    chunks: BoxStream<'static, Result<Bytes, Error>>,
) -> BoxStream<'static, Result<Bytes, Error>> {
    futures_util::stream::try_unfold(
        (chunks, Vec::new(), false),
        |(mut chunks, mut pending, ended)| async move {
            if ended {
                return Ok(None);
            }
            loop {
                match chunks.try_next().await? {
                    Some(chunk) => {
                        let scanned = pending.len();
                        pending.extend_from_slice(&chunk);
                        let (complete, tail) = split_complete_sse_frames_after(&pending, scanned);
                        if complete.is_empty() {
                            continue;
                        }
                        let frames = Bytes::copy_from_slice(complete);
                        pending = tail.to_vec();
                        return Ok(Some((frames, (chunks, pending, false))));
                    }
                    None if pending.is_empty() => return Ok(None),
                    None => {
                        let tail = Bytes::from(std::mem::take(&mut pending));
                        return Ok(Some((tail, (chunks, pending, true))));
                    }
                }
            }
        },
    )
    .boxed()
}

fn decoded_chunks(
    bytes: ByteStream,
    decode: StreamDecoder,
    provider: &'static str,
) -> BoxStream<'static, Result<Bytes, Error>> {
    futures_util::stream::try_unfold(decode(bytes), move |mut events| async move {
        let Some(event) = events.try_next().await? else {
            return Ok(None);
        };
        let chunk = streaming::encode(&event)?;
        log_chunk(provider, "client_response", &chunk);
        Ok(Some((chunk, events)))
    })
    .boxed()
}

fn log_request_body(provider: &str, stream: bool, body: &serde_json::Value) {
    tracing::debug!(provider, stream, body = %body, "provider request");
}

fn log_response_body(body: &str) {
    tracing::debug!(body, "provider response body");
}

fn log_error_body(status: u16, body: &str) {
    tracing::debug!(status, body, "provider error body");
}

fn log_chunk(provider: &str, stage: &str, data: &bytes::Bytes) {
    let chunk = ByteChunk::new(data);
    tracing::debug!(provider, stage, encoding = chunk.encoding(), chunk = %chunk, "stream chunk");
}

#[cfg(test)]
mod tests {
    use litellm_llms_types::headers::ProviderSpecificHeaders;
    use rstest::{fixture, rstest};
    use serde_json::{Map, Value, json};
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::any};

    use super::*;
    use crate::{MessagesSettings, MessagesShaping, common_utils::messages_provider};
    use litellm_inference::provider::resolve_llm_provider;

    #[fixture]
    fn shaping() -> MessagesShaping {
        MessagesShaping::default()
    }

    fn body(value: Value) -> MessagesRequest {
        serde_json::from_value(value).unwrap()
    }

    async fn provider_call_for(call: MessagesCall) -> Result<ProviderCall, Error> {
        provider_call_with_secrets(call, &|_: &str| None).await
    }

    async fn provider_call_with_secrets(
        call: MessagesCall,
        secrets: &(dyn Lookup + Send + Sync),
    ) -> Result<ProviderCall, Error> {
        let provider = resolve_llm_provider(
            &call.body.model,
            call.custom_llm_provider.as_deref(),
            "messages",
        )?;
        let config = messages_provider(provider.provider, provider.model)
            .ok_or_else(|| Error::InvalidProvider(<&str>::from(provider.provider).to_string()))?;
        let model = provider.model.to_owned();
        let interceptors = ();
        let context = CallContext::new(&interceptors, litellm_inference::CallOptions::default());
        provider_call(
            &AuthServices::default(),
            config,
            model,
            call,
            secrets,
            &context,
        )
        .await
    }

    async fn wire_body(fields: Value, shaping: MessagesShaping) -> Result<Value, Error> {
        provider_call_for(MessagesCall {
            mock_response: None,
            body: body(fields),
            api_key: Some("sk-test".into()),
            api_base: Some("https://anthropic.test".into()),
            custom_llm_provider: Some("anthropic".into()),
            litellm_params: Default::default(),
            extra_headers: None,
            provider_specific_header: None,
            timeout: None,
            shaping,
        })
        .await
        .map(|call| call.wire.body)
    }

    #[rstest]
    #[case::api_key(
        &[("ANTHROPIC_API_KEY", "sk-secret")],
        &[("x-api-key", "sk-secret")],
        "https://api.anthropic.com/v1/messages"
    )]
    #[case::auth_token(
        &[("ANTHROPIC_AUTH_TOKEN", "token")],
        &[("authorization", "Bearer token")],
        "https://api.anthropic.com/v1/messages"
    )]
    #[case::api_base(
        &[("ANTHROPIC_API_KEY", "sk-secret"), ("ANTHROPIC_API_BASE", "https://gateway.test")],
        &[("x-api-key", "sk-secret")],
        "https://gateway.test/v1/messages"
    )]
    #[case::sdk_base_url(
        &[("ANTHROPIC_API_KEY", "sk-secret"), ("ANTHROPIC_BASE_URL", "https://sdk.test")],
        &[("x-api-key", "sk-secret")],
        "https://sdk.test/v1/messages"
    )]
    #[tokio::test]
    async fn credentials_and_base_come_from_the_resolved_secrets(
        shaping: MessagesShaping,
        #[case] secrets: &[(&str, &str)],
        #[case] expected_auth: &[(&str, &str)],
        #[case] expected_url: &str,
    ) {
        let lookup = |name: &str| {
            secrets
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        };
        let call = provider_call_with_secrets(
            MessagesCall { mock_response: None,
                body: body(
                    json!({"model": "claude-test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}),
                ),
                api_key: None,
                api_base: None,
                custom_llm_provider: Some("anthropic".into()),
                litellm_params: Default::default(),
                extra_headers: None,
                provider_specific_header: None,
                timeout: None,
                shaping,
            },
            &lookup,
        )
        .await
        .unwrap();
        let auth: Vec<(&str, &str)> = call
            .wire
            .headers
            .iter()
            .filter(|(name, _)| matches!(name.as_str(), "x-api-key" | "authorization"))
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        assert_eq!(
            (auth.as_slice(), call.wire.url.as_str()),
            (expected_auth, expected_url)
        );
    }

    fn vertex_call_requesting(
        shaping: MessagesShaping,
        requested: &str,
        drop_params: bool,
    ) -> MessagesCall {
        MessagesCall {
            mock_response: None,
            body: body(
                json!({"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}),
            ),
            api_key: None,
            api_base: Some("https://vertex.test".into()),
            custom_llm_provider: Some("vertex_ai".into()),
            litellm_params: Default::default(),
            extra_headers: Some(Map::from_iter([
                ("authorization".to_string(), json!("Bearer vertex-token")),
                ("anthropic-version".to_string(), json!(requested)),
            ])),
            provider_specific_header: None,
            timeout: None,
            shaping: MessagesShaping {
                settings: MessagesSettings {
                    drop_params,
                    ..shaping.settings
                },
                ..shaping
            },
        }
    }

    fn vertex_project(name: &str) -> Option<String> {
        (name == "VERTEXAI_PROJECT").then(|| "version-project".to_string())
    }

    #[rstest]
    #[case::sdk_2("2026-10-21", false, "2026-10-21")]
    #[case::sdk_1("2023-06-01", false, "vertex-2023-10-16")]
    #[case::unknown_under_drop_params("latest", true, "vertex-2023-10-16")]
    #[tokio::test]
    async fn vertex_gets_the_requested_version_in_the_body_alone(
        shaping: MessagesShaping,
        #[case] requested: &str,
        #[case] drop_params: bool,
        #[case] body_version: &str,
    ) {
        let call = provider_call_with_secrets(
            vertex_call_requesting(shaping, requested, drop_params),
            &vertex_project,
        )
        .await
        .unwrap();
        let version_headers: Vec<&str> = call
            .wire
            .headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-version"))
            .map(|(_, value)| value.as_str())
            .collect();
        assert_eq!(
            (version_headers, &call.wire.body["anthropic_version"]),
            (Vec::<&str>::new(), &json!(body_version))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn vertex_refuses_an_unsupported_version_without_drop_params(shaping: MessagesShaping) {
        assert!(matches!(
            provider_call_with_secrets(
                vertex_call_requesting(shaping, "latest", false),
                &vertex_project
            )
            .await,
            Err(Error::Llm(litellm_llms::Error::InvalidRequest(
                litellm_llms::ErrorDetail::UnsupportedParameter { .. }
            )))
        ));
    }

    #[rstest]
    #[tokio::test]
    async fn missing_credentials_are_reported_before_the_request_is_transformed(
        shaping: MessagesShaping,
    ) {
        let result = provider_call_for(MessagesCall {
            mock_response: None,
            body: body(
                json!({"model": "claude-test", "messages": [{"role": "user", "content": "hi"}]}),
            ),
            api_key: None,
            api_base: None,
            custom_llm_provider: Some("anthropic".into()),
            litellm_params: Default::default(),
            extra_headers: None,
            provider_specific_header: None,
            timeout: None,
            shaping,
        })
        .await;
        let Err(error) = result else {
            panic!("a call without credentials fails");
        };
        assert!(error.is_missing_credential(), "{error:?}");
    }

    #[rstest]
    #[case::nothing_configured(None, None, None)]
    #[case::configured_setting(None, Some(1800.0), Some(Duration::from_secs(1800)))]
    #[case::call_timeout_wins(
        Some(Duration::from_secs(30)),
        Some(1800.0),
        Some(Duration::from_secs(30))
    )]
    #[case::call_timeout_only(Some(Duration::from_secs(30)), None, Some(Duration::from_secs(30)))]
    #[case::unusable_setting_ignored(None, Some(-1.0), None)]
    #[case::zero_setting_ignored(None, Some(0.0), None)]
    #[tokio::test]
    async fn timeout_falls_back_to_the_configured_request_timeout(
        shaping: MessagesShaping,
        #[case] call_timeout: Option<Duration>,
        #[case] request_timeout: Option<f64>,
        #[case] expected: Option<Duration>,
    ) {
        let call = provider_call_for(MessagesCall { mock_response: None,
            body: body(json!({"model": "claude-test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16})),
            api_key: Some("sk-test".into()),
            api_base: Some("https://anthropic.test".into()),
            custom_llm_provider: Some("anthropic".into()),
            litellm_params: Default::default(),
            extra_headers: None,
            provider_specific_header: None,
            timeout: call_timeout,
            shaping: MessagesShaping {
                settings: MessagesSettings { request_timeout, ..shaping.settings },
                ..shaping
            },
        })
        .await
        .unwrap();
        assert_eq!(call.timeout, expected);
    }

    #[rstest]
    #[case::top_level_and_nested_paths(
        json!({
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "context_management": {"edits": [{"type": "clear_thinking_20251015"}]},
            "metadata": {"user_id": "u1"},
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}, "input_examples": [{"q": "x"}]}]
        }),
        &["thinking", "context_management", "tools[*].input_examples"],
        json!({
            "max_tokens": 1024,
            "metadata": {"user_id": "u1"},
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}]
        }),
    )]
    #[case::no_paths(
        json!({"max_tokens": 16, "safeguards": [{"type": "dangerous_tool_use"}]}),
        &[],
        json!({"max_tokens": 16, "safeguards": [{"type": "dangerous_tool_use"}]}),
    )]
    #[case::model_and_messages_are_never_dropped(
        json!({"max_tokens": 16}),
        &["model", "messages", "messages[0].content"],
        json!({"max_tokens": 16}),
    )]
    #[tokio::test]
    async fn wire_body_drops_configured_paths(
        shaping: MessagesShaping,
        #[case] fields: Value,
        #[case] additional_drop_params: &[&str],
        #[case] expected_fields: Value,
    ) {
        let with_messages = |fields: Value| -> Value {
            let Value::Object(fields) = fields else {
                unreachable!()
            };
            Value::Object(
                [
                    ("model".to_string(), json!("claude-test")),
                    (
                        "messages".to_string(),
                        json!([{"role": "user", "content": "hi"}]),
                    ),
                ]
                .into_iter()
                .chain(fields)
                .collect(),
            )
        };
        let shaping = MessagesShaping {
            settings: MessagesSettings {
                additional_drop_params: additional_drop_params
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
                ..shaping.settings
            },
            ..shaping
        };
        assert_eq!(
            wire_body(with_messages(fields), shaping).await,
            Ok(with_messages(expected_fields))
        );
    }

    #[rstest]
    #[case::model_prefix_picks_the_provider(
        "azure_ai/claude-test",
        None,
        &[("x-priority", "extra"), ("x-scoped", "azure_ai")]
    )]
    #[case::explicit_provider(
        "claude-test",
        Some("anthropic"),
        &[("x-priority", "scoped"), ("x-scoped", "anthropic")]
    )]
    #[case::provider_prefix_on_an_anthropic_model(
        "anthropic/claude-test",
        None,
        &[("x-priority", "scoped"), ("x-scoped", "anthropic")]
    )]
    #[tokio::test]
    async fn provider_specific_headers_follow_the_resolved_provider(
        shaping: MessagesShaping,
        #[case] model: &str,
        #[case] custom_llm_provider: Option<&str>,
        #[case] expected: &[(&str, &str)],
    ) {
        let configured: ProviderSpecificHeaders = serde_json::from_value(json!([
            {"custom_llm_provider": "azure_ai", "extra_headers": {"x-scoped": "azure_ai"}},
            {"custom_llm_provider": "anthropic", "extra_headers": {"x-scoped": "anthropic", "x-priority": "scoped"}}
        ]))
        .unwrap();
        let call = provider_call_for(MessagesCall { mock_response: None,
            body: body(
                json!({"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}),
            ),
            api_key: Some("sk-test".into()),
            api_base: Some("https://resource.services.ai.azure.com".into()),
            custom_llm_provider: custom_llm_provider.map(Into::into),
            litellm_params: Default::default(),
            extra_headers: Some(Map::from_iter([("x-priority".into(), json!("extra"))])),
            provider_specific_header: Some(configured),
            timeout: None,
            shaping,
        })
        .await
        .unwrap();
        let caller_headers: Vec<(&str, &str)> = call
            .wire
            .headers
            .iter()
            .filter(|(name, _)| matches!(name.as_str(), "x-priority" | "x-scoped"))
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        assert_eq!(caller_headers, expected);
    }

    #[rstest]
    #[tokio::test]
    async fn wire_body_carries_the_provider_stripped_model(shaping: MessagesShaping) {
        assert_eq!(
            wire_body(
                json!({
                    "model": "anthropic/claude-test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16
                }),
                shaping,
            )
            .await,
            Ok(json!({
                "model": "claude-test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16
            }))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn dropped_thinking_display_is_not_restored_by_auto_summary(shaping: MessagesShaping) {
        let shaping = MessagesShaping {
            settings: MessagesSettings {
                reasoning_auto_summary: true,
                additional_drop_params: vec!["thinking.display".to_string()],
                ..shaping.settings
            },
            ..shaping
        };
        assert_eq!(
            wire_body(
                json!({
                    "model": "claude-test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 4096,
                    "thinking": {"type": "enabled", "budget_tokens": 2048}
                }),
                shaping,
            )
            .await,
            Ok(json!({
                "model": "claude-test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4096,
                "thinking": {"type": "enabled", "budget_tokens": 2048}
            }))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn dropping_an_invalid_metadata_user_id_does_not_skip_its_validation(
        shaping: MessagesShaping,
    ) {
        let shaping = MessagesShaping {
            settings: MessagesSettings {
                additional_drop_params: vec!["metadata.user_id".to_string()],
                ..shaping.settings
            },
            ..shaping
        };
        let error = wire_body(
            json!({
                "model": "claude-test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
                "metadata": {"user_id": 123}
            }),
            shaping,
        )
        .await
        .unwrap_err();
        assert!(error.is_request(), "{error:?}");
    }

    #[rstest]
    #[tokio::test]
    async fn wire_body_rejects_invalid_metadata_before_the_call(shaping: MessagesShaping) {
        assert_eq!(
            wire_body(
                json!({
                    "model": "claude-test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                    "metadata": {"user_id": 123}
                }),
                shaping,
            )
            .await,
            Err(Error::invalid_request(
                litellm_llms::ErrorDetail::InvalidValue {
                    field: "metadata.user_id",
                    expected: "a string",
                    actual: json!(123),
                }
            ))
        );
    }

    #[rstest]
    #[case::mantle(
        "bedrock_mantle/claude-test",
        "https://bedrock-mantle.us-west-2.api.aws",
        "/anthropic/v1/messages",
        "bedrock"
    )]
    #[case::platform(
        "bedrock/claude_platform/claude-test",
        "https://aws-external-anthropic.us-west-2.api.aws",
        "/v1/messages",
        "aws-external-anthropic"
    )]
    #[tokio::test]
    async fn explicit_endpoint_region_is_used_by_the_signer(
        shaping: MessagesShaping,
        #[case] model: &str,
        #[case] base: &str,
        #[case] path: &str,
        #[case] service: &str,
    ) {
        let call = provider_call_with_secrets(MessagesCall { mock_response: None,
            body: body(json!({"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]})),
            api_key: None,
            api_base: Some(base.into()),
            custom_llm_provider: None,
            litellm_params: crate::litellm_params(Map::from_iter([
                ("model".into(), json!(model)),
                ("aws_access_key_id".into(), json!("AKIDEXAMPLE")),
                ("aws_secret_access_key".into(), json!("secret")),
            ]))
            .unwrap(),
            extra_headers: None,
            provider_specific_header: None,
            timeout: None,
            shaping,
        }, &|name: &str| match name {
            "AWS_REGION_NAME" => Some("us-east-1".into()),
            "ANTHROPIC_AWS_WORKSPACE_ID" => Some("workspace".into()),
            _ => None,
        })
        .await
        .unwrap();
        assert_eq!(call.wire.url, format!("{base}{path}"));
        let signed = outbound_request(
            Authenticated {
                headers: call.wire.headers,
                signer: call.signer,
            },
            &call.wire.url,
            &call.wire.body,
            None,
        )
        .unwrap();
        let authorization = signed.headers()["authorization"].to_str().unwrap();
        assert!(
            authorization.contains(&format!("/us-west-2/{service}/aws4_request")),
            "{authorization}"
        );
    }

    #[rstest]
    #[case::event(
        "data: {\"type\":\"ping\"}\n\n",
        Some("event: ping\ndata: {\"type\":\"ping\"}\n\n")
    )]
    #[rstest::rstest]
    #[case::invalid_event("data: invalid\n\ndata: {\"type\":\"ping\"}\n\n", None)]
    #[tokio::test]
    async fn decoded_streams_encode_events_and_stop_at_the_first_error(
        #[case] body: &'static str,
        #[case] expected: Option<&str>,
    ) {
        let upstream = MockServer::start().await;
        Mock::given(any())
            .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
            .mount(&upstream)
            .await;
        let response = litellm_http::Client::for_test()
            .get(upstream.uri())
            .send()
            .await
            .unwrap();
        let MessagesCallResponse::Stream { mut chunks, .. } = streaming_response(
            response,
            Some(streaming::event_stream),
            "test",
            Duration::from_secs(5),
            false,
        ) else {
            panic!("a streaming response returns chunks");
        };

        let chunk = chunks.next().await.unwrap();
        match expected {
            Some(expected) => assert_eq!(chunk.unwrap().as_ref(), expected.as_bytes()),
            None => assert!(chunk.unwrap_err().is_response()),
        }
        assert!(chunks.next().await.is_none());
    }

    #[rstest]
    #[case::delimiter_straddles_reads(
        b"data: 1\r\n",
        b"\r\ndata: 2",
        b"data: 1\r\n\r\n",
        b"data: 2"
    )]
    #[case::boundary_in_new_bytes(b"data: 1", b"\n\ndata: 2\n\nx", b"data: 1\n\ndata: 2\n\n", b"x")]
    #[case::no_boundary_yet(b"data: 1\r", b"\ndata", b"", b"data: 1\r\ndata")]
    fn incremental_split_matches_a_full_rescan(
        #[case] earlier: &[u8],
        #[case] read: &[u8],
        #[case] complete: &[u8],
        #[case] tail: &[u8],
    ) {
        let pending = [earlier, read].concat();
        assert_eq!(
            split_complete_sse_frames_after(&pending, earlier.len()),
            (complete, tail)
        );
        assert_eq!(
            split_complete_sse_frames_after(&pending, 0),
            (complete, tail)
        );
    }

    #[rstest]
    #[case::no_boundary(b"event: a\ndata: 1", b"", b"event: a\ndata: 1")]
    #[case::one_frame(b"data: 1\n\n", b"data: 1\n\n", b"")]
    #[case::frame_and_tail(b"data: 1\n\ndata: 2", b"data: 1\n\n", b"data: 2")]
    #[case::crlf(b"data: 1\r\n\r\ndata: 2\n\nx", b"data: 1\r\n\r\ndata: 2\n\n", b"x")]
    #[case::cr_only(b"data: 1\r\rdata", b"data: 1\r\r", b"data")]
    fn sse_frames_split_at_the_last_boundary(
        #[case] pending: &[u8],
        #[case] complete: &[u8],
        #[case] tail: &[u8],
    ) {
        assert_eq!(
            split_complete_sse_frames_after(pending, 0),
            (complete, tail)
        );
    }

    #[rstest]
    #[case::regrouped(
        &["event: a\nda", "ta: 1\n\nevent: b\n", "data: 2\n\n"],
        &["event: a\ndata: 1\n\n", "event: b\ndata: 2\n\n"],
    )]
    #[case::tail_after_the_end(&["data: 1\n\nda", "ta"], &["data: 1\n\n", "data"])]
    #[case::nothing(&[], &[])]
    #[tokio::test]
    async fn whole_frames_regroup_the_reads(#[case] reads: &[&str], #[case] expected: &[&str]) {
        let reads: Vec<Result<Bytes, Error>> = reads
            .iter()
            .map(|read| Ok(Bytes::copy_from_slice(read.as_bytes())))
            .collect();
        let frames: Vec<Bytes> = in_whole_frames(futures_util::stream::iter(reads).boxed())
            .try_collect()
            .await
            .unwrap();
        let expected: Vec<Bytes> = expected
            .iter()
            .map(|frame| Bytes::copy_from_slice(frame.as_bytes()))
            .collect();
        assert_eq!(frames, expected);
    }
}
