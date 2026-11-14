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
        transformation::{BaseMessagesConfig, FederatedEnvironment},
    },
};
use litellm_llms_types::formats::messages::{
    MessagesOptionalParams, MessagesRequest, MessagesResponse,
};
use litellm_secrets::source::Secrets;
use litellm_tracing::{ByteChunk, REDACTED};
use serde_json::{Map, Value};

use super::{
    Delivered, Error, HostAnswer, MessagesBody, MessagesCall, MessagesCallResponse, MessagesRoute,
    Transport, TransportRequest, TransportResponse, UnencodableHeader, WorkloadIdentity,
    common_utils::{MessagesProvider, merge_messages_header_sources, string_headers},
    constants::{
        MESSAGES_HTTP_ERROR_ATTEMPTS, MESSAGES_STREAM_ERROR_BODY_READ_SECS, MESSAGES_TIMEOUT_SECS,
    },
    route::Messages,
    types::invalid_request,
};
use litellm_inference::{
    caching::{CacheRequest, CallCache},
    context::CallContext,
    outbound::{raw_leaf_body, serialized_outbound_request},
};

struct ProviderCall {
    identity: ProviderIdentity,
    wire: WireRequest,
    provider: MessagesProvider,
    signer: Option<litellm_auth_aws::SigV4Signer>,
    timeout: Option<Duration>,
    stream: bool,
    whole_sse_frames: bool,
    joined_reads: bool,
    transform_context: MessagesTransformContext,
    ascii_response_headers: bool,
    unencodable_request_header: Option<Vec<u8>>,
    ascii_request_headers: bool,
}

impl ProviderCall {
    fn cacheable(&self) -> bool {
        self.signer.is_none()
    }
}

pub(super) async fn cached(
    route: &MessagesRoute,
    provider: MessagesProvider,
    model: &str,
    context: &CallContext<'_, impl Interceptors<Error>>,
) -> Result<Option<MessagesCallResponse>, Error> {
    let Some(cache) = caller_keyed(route) else {
        return Ok(None);
    };
    let identity = ProviderIdentity {
        model: model.to_owned(),
        provider: provider.as_str().to_string(),
    };
    let Some((output, source)) = CallCache::<Messages>::new(
        Some(cache),
        context.cache,
        CacheRequest::from_wire(identity.clone(), None),
    )
    .lookup()
    .await
    else {
        return Ok(None);
    };
    context
        .result_ready(execution_facts(provider, identity, &output, &source))
        .await?;
    Ok(Some(output))
}

pub(super) async fn mocked(
    route: &MessagesRoute,
    provider: MessagesProvider,
    model: String,
    context: &CallContext<'_, impl Interceptors<Error>>,
    mock: Map<String, Value>,
    stream: bool,
) -> MessagesCallResponse {
    let output = MessagesCallResponse::Complete(MessagesBody::Unchecked(mock));
    let Some(cache) = caller_keyed(route).filter(|_| !stream) else {
        return output;
    };
    let identity = ProviderIdentity {
        model,
        provider: provider.as_str().to_string(),
    };
    CallCache::<Messages>::new(
        Some(cache),
        context.cache,
        CacheRequest::from_wire(identity, None),
    )
    .finish(output, &ResultSource::Provider)
    .await
}

pub(super) async fn answered(
    route: &MessagesRoute,
    host_answer: &dyn HostAnswer,
    provider: MessagesProvider,
    model: String,
    stream: bool,
    context: &CallContext<'_, impl Interceptors<Error>>,
) -> Result<MessagesCallResponse, Error> {
    host_answer.answer(stream).await?;
    deliver_answer(route, provider, model, stream, context).await
}

pub(super) async fn deliver_answer(
    route: &MessagesRoute,
    provider: MessagesProvider,
    model: String,
    stream: bool,
    context: &CallContext<'_, impl Interceptors<Error>>,
) -> Result<MessagesCallResponse, Error> {
    let identity = ProviderIdentity {
        model,
        provider: provider.as_str().to_string(),
    };
    let cache = CallCache::<Messages>::new(
        caller_keyed(route),
        context.cache,
        CacheRequest::from_wire(identity.clone(), None),
    );
    context
        .result_ready(ExecutionFacts {
            provider: identity,
            source: ResultSource::Host,
            cost: Cost::Deferred,
        })
        .await?;
    let output = MessagesCallResponse::Complete(MessagesBody::Unchecked(Map::new()));
    if stream {
        return Ok(output);
    }
    Ok(cache.finish(output, &ResultSource::Host).await)
}

fn caller_keyed(route: &MessagesRoute) -> Option<&litellm_cache_response::ScopedCache> {
    route
        .cache
        .as_ref()
        .filter(|cache| cache.service.keyed_by_caller())
}

pub(super) async fn execute(
    route: &MessagesRoute,
    provider: MessagesProvider,
    model: String,
    call: MessagesCall,
    secrets: Secrets,
    requested_stream: bool,
    context: &CallContext<'_, impl Interceptors<Error>>,
) -> Result<MessagesCallResponse, Error> {
    let caller_keyed = caller_keyed(route).is_some();
    let request = provider_call(
        &route.auth,
        route.workload_identity.as_deref(),
        provider,
        model,
        call,
        secrets.as_ref(),
        context,
    )
    .boxed()
    .await?;
    let cache = CallCache::<Messages>::from_wire(
        route
            .cache
            .as_ref()
            .filter(|_| caller_keyed || request.cacheable()),
        context.cache,
        &request.identity,
        &request.wire,
    );
    let identity = request.identity.clone();
    let hit = match caller_keyed {
        true => None,
        false => cache.lookup().await,
    };
    let (output, source) = match hit {
        Some(hit) => hit,
        None => (
            call_provider(&route.http, route.transport.as_deref(), request, context).await?,
            ResultSource::Provider,
        ),
    };
    context
        .result_ready(execution_facts(provider, identity, &output, &source))
        .await?;
    let output = match source {
        ResultSource::Provider => {
            super::safeguards::stamped(route.safeguards.as_ref(), output).await?
        }
        ResultSource::Cache { .. } | ResultSource::Host => output,
    };
    if !requested_stream && matches!(output, MessagesCallResponse::Stream { .. }) {
        return Ok(output);
    }
    Ok(cache.finish(output, &source).await)
}

fn execution_facts(
    provider: MessagesProvider,
    identity: ProviderIdentity,
    output: &MessagesCallResponse,
    source: &ResultSource,
) -> ExecutionFacts {
    let reported = match output {
        MessagesCallResponse::Complete(response) => match response {
            MessagesBody::Message(message) => provider.config().reported_cost(message),
            MessagesBody::Unchecked(_) => None,
        },
        MessagesCallResponse::Stream { .. } => None,
    };
    ExecutionFacts {
        provider: identity,
        cost: Cost::settle(source, reported),
        source: source.clone(),
    }
}

async fn provider_call(
    auth: &AuthServices,
    workload_identity: Option<&dyn WorkloadIdentity>,
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

    let sanitized = config.shape_request(
        MessagesRequest { model, ..body },
        shaping.settings.shape_options(),
    )?;
    let transform_context = MessagesTransformContext {
        drops_unsupported_params: shaping.settings.drops_unsupported_params,
        ..MessagesTransformContext::new(shaping.capabilities, shaping.settings.drop_params)
    };
    let dropped_keys = top_level_paths(&shaping.settings.additional_drop_params);
    let without_keys = without_additional_drop_params(sanitized, &dropped_keys)?;
    let requested = requested_optional_params(without_keys, provider, &transform_context)?;
    let requested_params = serde_json::to_value(&requested.params).map_err(serialize_failure)?;
    let trimmed =
        without_additional_drop_params(requested, &shaping.settings.additional_drop_params)?;

    let scoped =
        get_provider_specific_headers(provider_specific_header.as_ref(), provider.as_str());
    let forwarded = string_headers(Some(merge_messages_header_sources([
        extra_headers.unwrap_or_default(),
        scoped,
    ])))?;
    let validated = match config.validate_federated_environment(
        forwarded,
        api_key.as_deref(),
        &trimmed.model,
        &trimmed.params,
        &connection_params,
        &env_lookup,
    )? {
        FederatedEnvironment::Validated(validated) => validated,
        FederatedEnvironment::Unauthenticated {
            headers,
            missing,
            federate,
        } => match workload_identity {
            Some(identity) => match identity
                .token(&connection_params, api_base.as_deref())
                .await?
            {
                Some(token) => federate(headers, token),
                None => return Err(missing.into()),
            },
            None => return Err(missing.into()),
        },
    };

    let tool_custom_field = config.tool_custom_field(&trimmed.model);
    let requested_output_format = trimmed.params.output_format.clone();
    let body = config.request_body(
        tool_custom_field.apply(
            config.transform_anthropic_messages_request(trimmed, &transform_context)?,
            shaping.settings.drops_unsupported_params,
        ),
        &validated.headers,
        shaping.settings.drop_params,
    )?;
    let url = config.get_complete_url(
        api_base.as_deref(),
        &body.model,
        &connection_params,
        body.params.streams(),
        &env_lookup,
    )?;
    let environment = ValidatedEnvironment {
        headers: config.headers_for_url(
            config.request_headers(
                with_python_default_headers(validated.headers, config.default_headers()),
                &with_requested_output_format(&body, requested_output_format),
            ),
            &url,
        ),
        auth: validated.auth,
    };

    let request_context = RequestContext {
        model: body.model.clone(),
        custom_llm_provider: provider.as_str().to_string(),
        optional_params: requested_params,
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
    let stream = wire
        .body
        .get("stream")
        .is_some_and(litellm_llms_types::recognized::python_truthy);
    Ok(ProviderCall {
        identity,
        wire,
        provider,
        signer: authenticated.signer,
        timeout: timeout.or_else(|| shaping.settings.request_timeout()),
        stream,
        whole_sse_frames: shaping.settings.whole_sse_frames,
        joined_reads: shaping.settings.joined_reads,
        transform_context,
        ascii_response_headers: shaping.settings.ascii_response_headers,
        unencodable_request_header: shaping.settings.unencodable_request_header,
        ascii_request_headers: shaping.settings.ascii_request_headers,
    })
}

fn with_requested_output_format(
    body: &MessagesRequest,
    requested: Option<Value>,
) -> std::borrow::Cow<'_, MessagesRequest> {
    match requested {
        Some(output_format) if body.params.output_format.is_none() => {
            std::borrow::Cow::Owned(MessagesRequest {
                params: MessagesOptionalParams {
                    output_format: Some(output_format),
                    ..body.params.clone()
                },
                ..body.clone()
            })
        }
        _ => std::borrow::Cow::Borrowed(body),
    }
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

fn top_level_paths(paths: &[String]) -> Vec<String> {
    paths
        .iter()
        .filter(|path| !path.contains(['.', '[']))
        .cloned()
        .collect()
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

enum Upstream {
    Native(reqwest::Response),
    Hosted(TransportResponse),
}

async fn call_provider(
    http: &litellm_http::Client,
    transport: Option<&dyn Transport>,
    request: ProviderCall,
    context: &CallContext<'_, impl Interceptors<Error>>,
) -> Result<MessagesCallResponse, Error> {
    let timeout = request
        .timeout
        .unwrap_or(Duration::from_secs(MESSAGES_TIMEOUT_SECS));
    let (upstream, sent_body) =
        send_recovering(http, transport, &request, timeout, context).await?;
    let response = match upstream {
        Upstream::Native(response) => response,
        Upstream::Hosted(response) => {
            return hosted_response(response, sent_body, request, context).await;
        }
    };
    let ProviderCall {
        identity,
        provider,
        stream,
        whole_sse_frames,
        joined_reads,
        ..
    } = request;
    let provider_name = provider.as_str();
    let config = provider.config();
    let head = ResponseHead {
        status: response.status().as_u16(),
        headers: super::encoding::header_pairs(response.headers()),
        sent_body,
        streamed: stream,
    };
    if stream {
        context
            .raw_response_received(RawResponse::new(String::new()).with_head(head))
            .await?;
        let streamed = streaming_response(
            response,
            config.stream_decoder(),
            provider_name,
            timeout,
            joined_reads,
        );
        return Ok(if whole_sse_frames {
            framed(streamed)
        } else {
            streamed
        });
    }
    let status = head.status;
    let text = read_body(response, timeout).await?;
    log_response_body(&text);
    context
        .raw_response_received(RawResponse::new(text.clone()).with_head(head))
        .await?;
    if serde_json::from_str::<serde::de::IgnoredAny>(&text).is_err() {
        // Python raises `AnthropicError(status_code=<the response's>, message=<its text>)` for a
        // body it cannot parse, and the public mapping classifies it by that status.
        return Err(litellm_http::Error::upstream(status, text).into());
    }
    decode_response(config, &identity.model, &text).map(MessagesCallResponse::Complete)
}

async fn hosted_response(
    response: TransportResponse,
    sent_body: Bytes,
    request: ProviderCall,
    context: &CallContext<'_, impl Interceptors<Error>>,
) -> Result<MessagesCallResponse, Error> {
    let TransportResponse {
        status,
        headers,
        delivered,
    } = response;
    let ProviderCall {
        identity,
        provider,
        whole_sse_frames,
        ..
    } = request;
    let provider_name = provider.as_str();
    let config = provider.config();
    let text = match delivered {
        Delivered::Complete(text) => text,
        Delivered::Stream(chunks) => {
            context
                .raw_response_received(RawResponse::new(String::new()).with_head(ResponseHead {
                    status,
                    headers: headers.clone(),
                    sent_body,
                    streamed: true,
                }))
                .await?;
            let streamed =
                hosted_streaming_response(headers, chunks, config.stream_decoder(), provider_name);
            return Ok(if whole_sse_frames {
                framed(streamed)
            } else {
                streamed
            });
        }
        Delivered::Rejected { error, .. } => return Err(error),
    };
    log_response_body(&text);
    context
        .raw_response_received(RawResponse::new(text.clone()).with_head(ResponseHead {
            status,
            headers,
            sent_body,
            streamed: false,
        }))
        .await?;
    if serde_json::from_str::<serde::de::IgnoredAny>(&text).is_err() {
        return Err(litellm_http::Error::upstream(status, text).into());
    }
    decode_response(config, &identity.model, &text).map(MessagesCallResponse::Complete)
}

fn hosted_request(
    authenticated: Authenticated,
    url: &str,
    body: Bytes,
    stream: bool,
) -> Result<TransportRequest, Error> {
    if authenticated.signer.is_none() {
        return Ok(TransportRequest {
            url: url.to_owned(),
            headers: authenticated.headers,
            body,
            stream,
        });
    }
    let names: Vec<String> = authenticated
        .headers
        .iter()
        .map(|(name, _)| name.clone())
        .collect();
    let signed = serialized_outbound_request(authenticated, url, body.clone(), None)?;
    let headers = signed
        .headers()
        .iter()
        .filter_map(|(name, value)| {
            let spelled = names
                .iter()
                .find(|known| known.eq_ignore_ascii_case(name.as_str()))
                .cloned()
                .unwrap_or_else(|| name.to_string());
            Some((spelled, value.to_str().ok()?.to_owned()))
        })
        .collect();
    Ok(TransportRequest {
        url: signed.url().to_string(),
        headers,
        body,
        stream,
    })
}

async fn send_recovering(
    http: &litellm_http::Client,
    transport: Option<&dyn Transport>,
    request: &ProviderCall,
    timeout: Duration,
    context: &CallContext<'_, impl Interceptors<Error>>,
) -> Result<(Upstream, Bytes), Error> {
    let config = request.provider.config();
    let mut body = std::borrow::Cow::Borrowed(&request.wire.body);
    let mut attempt = 1;
    loop {
        log_request_body(request.provider.as_str(), request.stream, &body);
        if let Some(value) = &request.unencodable_request_header {
            return Err(Error::transport(UnencodableHeader::Sent(value.clone())));
        }
        // The credential header built from the caller's api_key never passes the caller header
        // check, so it is checked on the wire headers as Python's client encodes them.
        if request.ascii_request_headers
            && let Some((_, value)) = request
                .wire
                .headers
                .iter()
                .find(|(_, value)| !value.is_ascii())
        {
            return Err(Error::transport(UnencodableHeader::Sent(
                value.as_bytes().to_vec(),
            )));
        }
        let authenticated = Authenticated {
            headers: request.wire.headers.clone(),
            signer: request.signer.clone(),
        };
        let sent_body = raw_leaf_body(&body)?;
        let response = match transport {
            None => Upstream::Native(
                send(
                    http,
                    authenticated,
                    &request.wire.url,
                    sent_body.clone(),
                    timeout,
                )
                .await?,
            ),
            Some(transport) => Upstream::Hosted(
                transport
                    .send(hosted_request(
                        authenticated,
                        &request.wire.url,
                        sent_body.clone(),
                        request.stream,
                    )?)
                    .await?,
            ),
        };
        if let Upstream::Native(response) = &response
            && request.ascii_response_headers
            && let Some(unencodable) = UnencodableHeader::received(response.headers())
        {
            return Err(Error::transport(unencodable));
        }
        let failure = match response {
            Upstream::Native(response) if response.status().is_success() => {
                return Ok((Upstream::Native(response), sent_body));
            }
            Upstream::Native(response) => {
                provider_failure(response, timeout, request.stream).await?
            }
            Upstream::Hosted(TransportResponse {
                status,
                headers,
                delivered: Delivered::Rejected { text, error },
            }) => {
                log_error_body(status, &text);
                ProviderFailure {
                    status,
                    text,
                    headers,
                    raised: Some(error),
                }
            }
            hosted @ Upstream::Hosted(_) => return Ok((hosted, sent_body)),
        };
        let retried = if attempt == MESSAGES_HTTP_ERROR_ATTEMPTS {
            None
        } else {
            config.retry_after_http_error(
                failure.status,
                &failure.text,
                &body,
                &request.transform_context,
            )
        };
        let Some(retried) = retried else {
            return Err(failure.into());
        };
        attempt += 1;
        tracing::debug!(
            "Anthropic /v1/messages: retrying after a recoverable HTTP error (attempt {attempt}/{MESSAGES_HTTP_ERROR_ATTEMPTS})."
        );
        body = std::borrow::Cow::Owned(retried.body.clone());
        context.provider_request_retried(retried).await?;
    }
}

fn serialize_failure(err: serde_json::Error) -> Error {
    Error::invalid_request(litellm_llms::ErrorDetail::failed(
        "Anthropic messages request serialization",
        err,
    ))
}

async fn send(
    http: &litellm_http::Client,
    authenticated: Authenticated,
    url: &str,
    body: Bytes,
    timeout: Duration,
) -> Result<reqwest::Response, Error> {
    let request = serialized_outbound_request(authenticated, url, body, None)?;
    tokio::time::timeout(timeout, litellm_inference::outbound::send(request, http))
        .await
        .map_err(litellm_http::Error::read_timeout)?
        .map(super::encoding::decoded)
        .map_err(Error::from)
}

async fn read_body(
    mut response: reqwest::Response,
    timeout: Duration,
) -> Result<String, litellm_http::Error> {
    let expected = response.content_length();
    let mut body = Vec::new();
    while let Some(chunk) = tokio::time::timeout(timeout, response.chunk())
        .await
        .map_err(litellm_http::Error::read_timeout)?
        .map_err(|error| litellm_http::Error::body_read(error, body.len() as u64, expected))?
    {
        body.extend_from_slice(&chunk);
    }
    Ok(String::from_utf8_lossy(&body).into_owned())
}

/// The stream's body, each read bounded by `timeout` as httpx bounds each read.
fn timed_chunks(
    response: reqwest::Response,
    timeout: Duration,
) -> impl futures_util::Stream<Item = Result<Bytes, litellm_http::Error>> + Send + 'static {
    let expected = response.content_length();
    futures_util::stream::try_unfold(
        (response, 0_u64),
        move |(mut response, received)| async move {
            let chunk = tokio::time::timeout(timeout, response.chunk())
                .await
                .map_err(litellm_http::Error::body_read_timeout)?
                .map_err(|error| litellm_http::Error::body_read(error, received, expected))?;
            Ok(chunk.map(|chunk| {
                let received = received + chunk.len() as u64;
                (chunk, (response, received))
            }))
        },
    )
}

struct ProviderFailure {
    status: u16,
    text: String,
    headers: Vec<(String, String)>,
    raised: Option<Error>,
}

impl From<ProviderFailure> for Error {
    fn from(failure: ProviderFailure) -> Self {
        match failure.raised {
            Some(raised) => raised,
            None => litellm_http::Error::upstream_with_headers(
                failure.status,
                failure.text,
                failure.headers,
            )
            .into(),
        }
    }
}

async fn provider_failure(
    response: reqwest::Response,
    timeout: Duration,
    stream: bool,
) -> Result<ProviderFailure, Error> {
    let status = response.status().as_u16();
    let headers = super::encoding::header_pairs(response.headers());
    let text = if stream {
        tokio::time::timeout(
            Duration::from_secs(MESSAGES_STREAM_ERROR_BODY_READ_SECS),
            read_body(response, timeout),
        )
        .await
        .ok()
        .and_then(Result::ok)
        .unwrap_or_default()
    } else {
        read_body(response, timeout).await?
    };
    log_error_body(status, &text);
    Ok(ProviderFailure {
        status,
        text,
        headers,
        raised: None,
    })
}

fn decode_response(
    config: &dyn BaseMessagesConfig,
    model: &str,
    text: &str,
) -> Result<MessagesBody, Error> {
    let marked = litellm_host::raw_json::mark_wide_numbers(text);
    let response = match serde_json::from_str::<MessagesResponse>(&marked) {
        Ok(response) => response,
        Err(err) => {
            let Ok(mut body) = serde_json::from_str::<Map<String, Value>>(&marked) else {
                return Err(Error::invalid_response(litellm_llms::ErrorDetail::invalid(
                    "messages response JSON",
                    err,
                )));
            };
            body.entry("stop_sequence").or_insert(Value::Null);
            return Ok(MessagesBody::Unchecked(body));
        }
    };
    let message = config
        .transform_anthropic_messages_response(model, response)
        .map_err(Error::from)
        .map_err(Error::after_send)?;
    if text_has_stop_reason(text) {
        return Ok(MessagesBody::Message(Box::new(message)));
    }
    match serde_json::to_value(message).map_err(serialize_failure)? {
        Value::Object(mut body) => {
            body.remove("stop_reason");
            Ok(MessagesBody::Unchecked(body))
        }
        _ => unreachable!("a message serializes as a JSON object"),
    }
}

#[derive(serde::Deserialize)]
struct StopReasonKey {
    #[serde(default, deserialize_with = "key_present")]
    stop_reason: bool,
}

fn key_present<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
    <serde::de::IgnoredAny as serde::Deserialize<'de>>::deserialize(deserializer).map(|_| true)
}

fn text_has_stop_reason(text: &str) -> bool {
    serde_json::from_str::<StopReasonKey>(text).is_ok_and(|key| key.stop_reason)
}

fn streaming_response(
    response: reqwest::Response,
    decoder: Option<StreamDecoder>,
    provider: &'static str,
    timeout: Duration,
    joined: bool,
) -> MessagesCallResponse {
    let headers = super::encoding::header_pairs(response.headers());
    let chunks = timed_chunks(response, timeout);
    let chunks = match (joined, &decoder) {
        (true, None) => super::reads::joined_reads(chunks),
        _ => chunks.boxed(),
    }
    .inspect_ok(move |chunk| log_chunk(provider, "provider_response", chunk));
    let chunks = match decoder {
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

fn hosted_streaming_response(
    headers: Vec<(String, String)>,
    chunks: BoxStream<'static, Result<Bytes, Error>>,
    decoder: Option<StreamDecoder>,
    provider: &'static str,
) -> MessagesCallResponse {
    let chunks = chunks.inspect_ok(move |chunk| log_chunk(provider, "provider_response", chunk));
    let chunks = match decoder {
        None => chunks.boxed(),
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

fn framed(response: MessagesCallResponse) -> MessagesCallResponse {
    let MessagesCallResponse::Stream { head, chunks } = response else {
        return response;
    };
    MessagesCallResponse::Stream {
        head,
        chunks: whole_sse_frames(chunks),
    }
}

fn whole_sse_frames(
    chunks: BoxStream<'static, Result<Bytes, Error>>,
) -> BoxStream<'static, Result<Bytes, Error>> {
    futures_util::stream::try_unfold(
        (Some(chunks), Vec::new()),
        |(chunks, mut pending)| async move {
            let Some(mut chunks) = chunks else {
                return Ok(None);
            };
            while let Some(chunk) = chunks.try_next().await? {
                pending.extend_from_slice(&chunk);
                let end = complete_frames_end(&pending);
                if end > 0 {
                    let rest = pending.split_off(end);
                    return Ok(Some((Bytes::from(pending), (Some(chunks), rest))));
                }
            }
            Ok((!pending.is_empty()).then(|| (Bytes::from(pending), (None, Vec::new()))))
        },
    )
    .boxed()
}

fn complete_frames_end(pending: &[u8]) -> usize {
    [b"\r\n\r\n".as_slice(), b"\n\n", b"\r\r"]
        .iter()
        .filter_map(|delimiter| {
            pending
                .windows(delimiter.len())
                .rposition(|window| window == *delimiter)
                .map(|at| at + delimiter.len())
        })
        .max()
        .unwrap_or(0)
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
    tracing::debug!(provider, stream, body = %without_mcp_tokens(body), "provider request");
}

fn without_mcp_tokens(body: &Value) -> std::borrow::Cow<'_, Value> {
    let Some(servers) = body.get("mcp_servers").and_then(Value::as_array) else {
        return std::borrow::Cow::Borrowed(body);
    };
    let mut redacted = body.clone();
    redacted["mcp_servers"] = servers.iter().map(without_mcp_token).collect();
    std::borrow::Cow::Owned(redacted)
}

fn without_mcp_token(server: &Value) -> Value {
    match server {
        Value::Object(fields) if fields.contains_key("authorization_token") => Value::Object(
            fields
                .iter()
                .map(|(key, value)| match key.as_str() {
                    "authorization_token" => (key.clone(), Value::from(REDACTED)),
                    _ => (key.clone(), value.clone()),
                })
                .collect(),
        ),
        other => other.clone(),
    }
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
        provider_call_federating(call, secrets, None).await
    }

    async fn provider_call_federating(
        call: MessagesCall,
        secrets: &(dyn Lookup + Send + Sync),
        workload_identity: Option<&dyn WorkloadIdentity>,
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
            workload_identity,
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

    fn vertex_call_requesting(
        shaping: MessagesShaping,
        requested: &str,
        drop_params: bool,
    ) -> MessagesCall {
        MessagesCall {
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

    struct ScriptedIdentity {
        minted: Result<Option<&'static str>, ()>,
        asked: std::sync::Mutex<Vec<(Option<Value>, Option<String>)>>,
    }

    impl ScriptedIdentity {
        fn new(minted: Result<Option<&'static str>, ()>) -> Self {
            Self {
                minted,
                asked: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn asked(&self) -> Vec<(Option<Value>, Option<String>)> {
            self.asked.lock().unwrap().clone()
        }
    }

    impl WorkloadIdentity for ScriptedIdentity {
        fn token<'a>(
            &'a self,
            litellm_params: &'a litellm_router_types::LitellmParams,
            api_base: Option<&'a str>,
        ) -> futures_util::future::BoxFuture<'a, Result<Option<SecretValue>, litellm_auth::Error>>
        {
            self.asked.lock().unwrap().push((
                litellm_params
                    .extra
                    .get("anthropic_federation_rule_id")
                    .cloned(),
                api_base.map(str::to_owned),
            ));
            let minted = self.minted;
            Box::pin(async move {
                minted
                    .map(|token| token.map(SecretValue::new))
                    .map_err(|()| litellm_auth::Error::InvalidHeader)
            })
        }
    }

    fn federated_call(
        shaping: MessagesShaping,
        api_key: Option<&str>,
        extra_headers: Option<Value>,
    ) -> MessagesCall {
        MessagesCall {
            body: body(
                json!({"model": "claude-test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}),
            ),
            api_key: api_key.map(str::to_owned),
            api_base: Some("https://anthropic.test".into()),
            custom_llm_provider: Some("anthropic".into()),
            litellm_params: crate::litellm_params(
                serde_json::from_value(
                    json!({"model": "claude-test", "anthropic_federation_rule_id": "fdrl_1"}),
                )
                .unwrap(),
            )
            .unwrap(),
            extra_headers: extra_headers.map(|headers| serde_json::from_value(headers).unwrap()),
            provider_specific_header: None,
            timeout: None,
            shaping,
        }
    }

    fn sent_headers<'a>(call: &'a ProviderCall, names: &[&str]) -> Vec<(&'a str, &'a str)> {
        call.wire
            .headers
            .iter()
            .filter(|(name, _)| names.contains(&name.as_str()))
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect()
    }

    #[rstest]
    #[case::alone(None, &[("anthropic-beta", "oauth-2025-04-20"), ("authorization", "Bearer sk-ant-oat01-minted")])]
    #[case::beside_a_caller_beta(
        Some(json!({"anthropic-beta": "files-api-2025-04-14"})),
        &[("anthropic-beta", "files-api-2025-04-14,oauth-2025-04-20"), ("authorization", "Bearer sk-ant-oat01-minted")]
    )]
    #[tokio::test]
    async fn a_minted_federation_token_is_sent_as_an_oauth_bearer(
        shaping: MessagesShaping,
        #[case] extra_headers: Option<Value>,
        #[case] expected: &[(&str, &str)],
    ) {
        let identity = ScriptedIdentity::new(Ok(Some("sk-ant-oat01-minted")));
        let call = provider_call_federating(
            federated_call(shaping, None, extra_headers),
            &|_: &str| None,
            Some(&identity),
        )
        .await
        .unwrap();
        let mut sent = sent_headers(
            &call,
            &[
                "anthropic-beta",
                "authorization",
                "x-api-key",
                "anthropic-dangerous-direct-browser-access",
            ],
        );
        sent.sort_unstable();
        assert_eq!(
            (sent.as_slice(), identity.asked()),
            (
                expected,
                vec![(Some(json!("fdrl_1")), Some("https://anthropic.test".into()))]
            )
        );
    }

    #[rstest]
    #[case::static_key(Some("sk-static"), None, &[("x-api-key", "sk-static")])]
    #[case::caller_credential(
        None,
        Some(json!({"x-api-key": "sk-caller"})),
        &[("x-api-key", "sk-caller")]
    )]
    #[tokio::test]
    async fn a_configured_credential_is_never_federated(
        shaping: MessagesShaping,
        #[case] api_key: Option<&str>,
        #[case] extra_headers: Option<Value>,
        #[case] expected: &[(&str, &str)],
    ) {
        let identity = ScriptedIdentity::new(Ok(Some("sk-ant-oat01-minted")));
        let call = provider_call_federating(
            federated_call(shaping, api_key, extra_headers),
            &|_: &str| None,
            Some(&identity),
        )
        .await
        .unwrap();
        assert_eq!(
            (
                sent_headers(&call, &["authorization", "x-api-key"]),
                identity.asked()
            ),
            (expected.to_vec(), Vec::new())
        );
    }

    #[rstest]
    #[case::nothing_minted(Ok(None), true)]
    #[case::minting_failed(Err(()), false)]
    #[tokio::test]
    async fn an_unminted_federation_token_fails_the_call(
        shaping: MessagesShaping,
        #[case] minted: Result<Option<&'static str>, ()>,
        #[case] missing_credential: bool,
    ) {
        let identity = ScriptedIdentity::new(minted);
        let Err(error) = provider_call_federating(
            federated_call(shaping, None, None),
            &|_: &str| None,
            Some(&identity),
        )
        .await
        else {
            panic!("a call without a minted token fails");
        };
        assert_eq!(
            error.is_missing_credential(),
            missing_credential,
            "{error:?}"
        );
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
        let call = provider_call_for(MessagesCall {
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
        let call = provider_call_with_secrets(MessagesCall {
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
        let signed = serialized_outbound_request(
            Authenticated {
                headers: call.wire.headers,
                signer: call.signer,
            },
            &call.wire.url,
            raw_leaf_body(&call.wire.body).unwrap(),
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
    #[case::no_delimiter(b"data: a".as_slice(), 0)]
    #[case::lf(b"data: a\n\ndata: b", 9)]
    #[case::crlf(b"data: a\r\n\r\ndata: b", 11)]
    #[case::cr(b"data: a\r\rdata: b", 9)]
    #[case::last_of_several(b"data: a\n\ndata: b\n\ndata: c", 18)]
    fn complete_frames_end_is_after_the_last_delimiter(#[case] pending: &[u8], #[case] end: usize) {
        assert_eq!(complete_frames_end(pending), end);
    }

    #[rstest]
    #[case::split_frame(
        vec![&b"data: a\n"[..], b"\ndata: ", b"b\n\n"],
        vec![&b"data: a\n\n"[..], b"data: b\n\n"]
    )]
    #[case::several_frames_in_one_chunk(
        vec![&b"data: a\n\ndata: b\n\nda"[..], b"ta: c\n\n"],
        vec![&b"data: a\n\ndata: b\n\n"[..], b"data: c\n\n"]
    )]
    #[case::unterminated_tail_is_flushed(
        vec![&b"data: a\n\nda"[..], b"ta: b"],
        vec![&b"data: a\n\n"[..], b"data: b"]
    )]
    #[tokio::test]
    async fn whole_sse_frames_regroups_chunks_at_frame_boundaries(
        #[case] chunks: Vec<&'static [u8]>,
        #[case] expected: Vec<&'static [u8]>,
    ) {
        let upstream = futures_util::stream::iter(
            chunks
                .into_iter()
                .map(|chunk| Ok::<_, Error>(Bytes::from_static(chunk))),
        )
        .boxed();
        let framed: Vec<Bytes> = whole_sse_frames(upstream).try_collect().await.unwrap();
        assert_eq!(
            framed,
            expected
                .into_iter()
                .map(Bytes::from_static)
                .collect::<Vec<_>>()
        );
    }
}
