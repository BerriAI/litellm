use std::time::Duration;

use bytes::Bytes;
use futures_util::{FutureExt, StreamExt, TryStreamExt, stream::BoxStream};
use litellm_auth::{AuthServices, SecretValue};
use litellm_core_utils::{
    dot_notation_indexing::delete_nested_value,
    get_provider_specific_headers::get_provider_specific_headers, settings::Lookup,
};
use litellm_host::interceptors::{Interceptors, ProviderIdentity, RequestContext, WireRequest};
use litellm_http::transport::Error as TransportError;
use litellm_llms::base_llm::{
    auth::{Authenticated, resolve_auth},
    messages::{
        context::MessagesTransformContext,
        streaming::{ByteStream, StreamDecoder, encode_anthropic_sse},
        transformation::BaseMessagesConfig,
    },
};
use litellm_llms_types::formats::messages::{MessagesRequest, MessagesResponse};
use litellm_secrets::source::Secrets;
use litellm_tracing::ByteChunk;
use serde_json::Value;

use litellm_inference::{
    caching::CallCache, context::CallContext, outbound::outbound_request,
    provider::ResolvedProvider,
};

use super::{
    Error, MessagesCall, MessagesCallResponse, MessagesRoute,
    common_utils::{MessagesProvider, string_headers, truncate_error_body},
    constants::MESSAGES_TIMEOUT_SECS,
    types::invalid_request,
};

pub(super) struct ProviderCall {
    pub identity: ProviderIdentity,
    pub wire: WireRequest,
    provider: super::common_utils::MessagesProvider,
    signer: Option<litellm_auth_aws::SigV4Signer>,
    timeout: Option<Duration>,
    stream: bool,
}

impl ProviderCall {
    pub fn cacheable(&self) -> bool {
        self.signer.is_none()
    }
}

pub(super) async fn execute(
    route: &MessagesRoute,
    config: MessagesProvider,
    provider: ResolvedProvider<'_>,
    call: MessagesCall,
    secrets: Secrets,
    context: &CallContext<'_, impl Interceptors<Error>>,
) -> Result<MessagesCallResponse, Error> {
    let request = provider_call(
        &route.auth,
        config,
        provider,
        call,
        secrets.as_ref(),
        context,
    )
    .boxed()
    .await?;
    let cache = CallCache::<super::route::Messages>::from_wire(
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
            litellm_host::interceptors::ResultSource::Provider,
        ),
    };
    context
        .result_ready(litellm_host::interceptors::ExecutionFacts {
            provider: identity,
            source: source.clone(),
        })
        .await?;
    Ok(cache.finish(output, &source).await)
}

pub(super) async fn provider_call(
    auth: &AuthServices,
    config: MessagesProvider,
    provider: ResolvedProvider<'_>,
    call: MessagesCall,
    secrets: &(dyn Lookup + Sync),
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
    let env_lookup = |key: &str| secrets.get(key);
    let scoped = get_provider_specific_headers(provider_specific_header.as_ref(), config.as_str());
    let forwarded = string_headers(Some(
        extra_headers.into_iter().flatten().chain(scoped).collect(),
    ))?;
    let validated = config.config().validate_environment(
        forwarded,
        api_key.as_deref(),
        provider.model,
        &litellm_params,
        &env_lookup,
    )?;
    let sanitized = config.config().shape_request(
        MessagesRequest {
            model: provider.model.to_owned(),
            ..body
        },
        shaping.settings.reasoning_auto_summary,
    )?;
    let trimmed =
        without_additional_drop_params(sanitized, &shaping.settings.additional_drop_params)?;
    let body = config.config().transform_anthropic_messages_request(
        trimmed,
        &MessagesTransformContext::new(shaping.capabilities, shaping.settings.drop_params),
    )?;
    let environment = litellm_llms::base_llm::auth::ValidatedEnvironment {
        headers: config.config().request_headers(
            litellm_http::request::with_default_headers(
                validated.headers,
                config.config().default_headers(),
            ),
            &body,
        ),
        auth: validated.auth,
    };
    let url = config.config().get_complete_url(
        api_base.as_deref(),
        &body.model,
        &litellm_params,
        body.params.stream == Some(true),
        &env_lookup,
    )?;
    let api_key = api_key.map(SecretValue::new);
    let request_context = RequestContext {
        model: body.model.clone(),
        custom_llm_provider: config.as_str().to_string(),
        optional_params: serde_json::to_value(&body.params).map_err(serialize_failure)?,
        secret_fields: Vec::new(),
        api_key,
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
                headers: authenticated.headers,
                body: config
                    .config()
                    .wire_body(serde_json::to_value(&body).map_err(serialize_failure)?),
            },
            request_context,
        )
        .await?;
    let stream = match wire.body.get("stream") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(stream)) => *stream,
        Some(value) => {
            return Err(Error::InvalidRequest(
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
        provider: config,
        signer: authenticated.signer,
        timeout,
        stream,
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
    } = request;
    let provider_name = provider.as_str();
    log_request_body(provider_name, stream, &wire.body);
    let response = send(
        http,
        Authenticated {
            headers: wire.headers,
            signer,
        },
        &wire.url,
        &wire.body,
        timeout,
    )
    .await?;
    if !response.status().is_success() {
        return Err(provider_error(response).await);
    }
    let config = provider.config();
    if stream {
        return Ok(streaming_response(
            response,
            config.stream_decoder(),
            provider_name,
        ));
    }
    let text = response.text().await.map_err(network)?;
    log_response_body(&text);
    context.response_received(&text).await?;
    decode_response(config, &identity.model, &text)
        .map(|message| MessagesCallResponse::Complete(Box::new(message)))
}

fn serialize_failure(err: serde_json::Error) -> Error {
    Error::InvalidRequest(litellm_llms::ErrorDetail::failed(
        "Anthropic messages request serialization",
        err,
    ))
}

fn network(error: reqwest::Error) -> Error {
    Error::Transport(TransportError::Network(error.to_string()))
}

async fn send(
    http: &litellm_http::Client,
    authenticated: Authenticated,
    url: &str,
    body: &Value,
    timeout: Option<Duration>,
) -> Result<reqwest::Response, Error> {
    let request = outbound_request(
        authenticated,
        url.to_string(),
        body,
        Some(timeout.unwrap_or(Duration::from_secs(MESSAGES_TIMEOUT_SECS))),
    )?;
    litellm_inference::outbound::send(request, http)
        .await
        .map_err(network)
}

async fn provider_error(response: reqwest::Response) -> Error {
    let status = response.status().as_u16();
    match response.text().await {
        Ok(text) => {
            log_error_body(status, &text);
            Error::Transport(TransportError::Http {
                status,
                body: truncate_error_body(&text),
            })
        }
        Err(error) => network(error),
    }
}

fn decode_response(
    config: &dyn BaseMessagesConfig,
    model: &str,
    text: &str,
) -> Result<MessagesResponse, Error> {
    let response = serde_json::from_str(text).map_err(|err| {
        Error::InvalidResponse(litellm_llms::ErrorDetail::invalid(
            "messages response JSON",
            err,
        ))
    })?;
    config
        .transform_anthropic_messages_response(model, response)
        .map_err(Error::from)
}

fn streaming_response(
    response: reqwest::Response,
    decoder: Option<StreamDecoder>,
    provider: &'static str,
) -> MessagesCallResponse {
    let headers = response
        .headers()
        .iter()
        .filter_map(|(name, value)| Some((name.to_string(), value.to_str().ok()?.to_string())))
        .collect();
    let chunks = match decoder {
        None => futures_util::stream::try_unfold(response, move |mut response| async move {
            let chunk = response.chunk().await.map_err(network)?;
            Ok(chunk.map(|chunk| {
                log_chunk(provider, "provider_response", &chunk);
                (chunk, response)
            }))
        })
        .boxed(),
        Some(decode) => decoded_chunks(response, decode, provider),
    };
    MessagesCallResponse::Stream {
        head: super::route::MessagesStreamHead { headers },
        chunks,
    }
}

fn decoded_chunks(
    response: reqwest::Response,
    decode: StreamDecoder,
    provider: &'static str,
) -> BoxStream<'static, Result<Bytes, Error>> {
    let bytes: ByteStream = response
        .bytes_stream()
        .inspect_ok(move |chunk| log_chunk(provider, "provider_response", chunk))
        .map_err(std::io::Error::other)
        .boxed();
    futures_util::stream::try_unfold(decode(bytes), move |mut events| async move {
        let Some(event) = events.try_next().await? else {
            return Ok(None);
        };
        let chunk = encode_anthropic_sse(&event)?;
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
    use litellm_core_utils::settings::Lookup;
    use litellm_llms::base_llm::messages::streaming::anthropic_sse_event_stream;
    use litellm_llms_types::headers::ProviderSpecificHeaders;
    use rstest::{fixture, rstest};
    use serde_json::{Map, Value, json};
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::any};

    use super::*;
    use crate::{MessagesSettings, MessagesShaping};

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
        secrets: &(dyn Lookup + Sync),
    ) -> Result<ProviderCall, Error> {
        let model = call.body.model.clone();
        let custom_llm_provider = call.custom_llm_provider.clone();
        let (provider, config) = crate::provider_config::resolve_provider_config(
            &model,
            custom_llm_provider.as_deref(),
        )?;
        let interceptors = ();
        let context = CallContext::new(&interceptors, litellm_inference::CallOptions::default());
        let auth = AuthServices::default();
        provider_call(&auth, config, provider, call, secrets, &context).await
    }

    async fn wire_body(fields: Value, shaping: MessagesShaping) -> Result<Value, Error> {
        provider_call_for(MessagesCall {
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
            MessagesCall {
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
    async fn prepared_body_drops_configured_paths(
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
        let call = provider_call_for(MessagesCall {
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
    async fn prepared_body_carries_the_provider_stripped_model(shaping: MessagesShaping) {
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
        assert!(matches!(
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
            Err(Error::InvalidRequest(_))
        ));
    }

    #[rstest]
    #[tokio::test]
    async fn prepared_body_rejects_invalid_metadata_before_the_call(shaping: MessagesShaping) {
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
            Err(Error::InvalidRequest(
                litellm_llms::ErrorDetail::InvalidValue {
                    field: "metadata.user_id",
                    expected: "a string",
                    actual: json!(123),
                }
            ))
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
        let response = litellm_http::Client::plain_for_test()
            .get(upstream.uri())
            .send()
            .await
            .unwrap();
        let MessagesCallResponse::Stream { mut chunks, .. } =
            streaming_response(response, Some(anthropic_sse_event_stream), "test")
        else {
            panic!("a streaming response returns chunks");
        };

        let chunk = chunks.next().await.unwrap();
        match expected {
            Some(expected) => assert_eq!(chunk.unwrap().as_ref(), expected.as_bytes()),
            None => assert!(matches!(chunk, Err(Error::InvalidResponse(_))), "{chunk:?}"),
        }
        assert!(chunks.next().await.is_none());
    }
}
