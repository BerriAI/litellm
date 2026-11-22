use std::time::Duration;

use bytes::Bytes;
use futures_util::{FutureExt, StreamExt, TryStreamExt, stream::BoxStream};
use litellm_host::{
    interceptors::{
        Cost, ExecutionFacts, Interceptors, ProviderIdentity, RawResponse, ResponseHead,
        ResultSource,
    },
    protocol::UnobservedTail,
};
use litellm_http::Phases;
use litellm_llms::base_llm::{
    auth::Authenticated,
    messages::{
        streaming::{ByteStream, StreamDecoder},
        transformation::BaseMessagesConfig,
    },
};
use litellm_llms_types::formats::messages::MessagesResponse;
use litellm_secrets::source::Secrets;
use litellm_tracing::{ByteChunk, REDACTED};
use serde_json::{Map, Value};

use super::{
    Delivered, Error, HostAnswer, MessagesBody, MessagesCall, MessagesCallResponse, MessagesRoute,
    Transport, TransportRequest, TransportResponse, UnencodableHeader, UnsendableUrl,
    common_utils::MessagesProvider,
    constants::{
        MESSAGES_CONNECT_TIMEOUT_SECS, MESSAGES_HTTP_ERROR_ATTEMPTS,
        MESSAGES_STREAM_ERROR_BODY_READ_SECS, MESSAGES_TIMEOUT_SECS,
    },
    request::{CallerCredentials, ProviderCall, provider_call, serialize_failure},
    route::Messages,
};
use litellm_inference::{
    caching::{CacheRequest, CallCache},
    context::CallContext,
    outbound::{raw_leaf_body, serialized_outbound_request},
};

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
        CallerCredentials {
            auth: &route.auth,
            workload_identity: route.workload_identity.as_deref(),
            azure_ad_token_hosts: route.azure_ad_token_hosts.clone(),
            deferred_secrets: Some(route.secrets.as_ref()),
        },
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
        detached_drain,
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
        let streamed = draining_after_detach(streamed, config.stream_decoder(), detached_drain);
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
        detached_drain,
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
            let streamed = draining_after_detach(streamed, config.stream_decoder(), detached_drain);
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

fn unsendable_url(provider: MessagesProvider, url: &str) -> Option<UnsendableUrl> {
    provider
        .refuses_unsendable_urls()
        .then(|| UnsendableUrl::of(url))
        .flatten()
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
    let mut recovered: Vec<(u16, String)> = Vec::new();
    loop {
        log_request_body(request.provider.as_str(), request.stream, &body);
        if let Some(value) = &request.unencodable_request_header {
            return Err(Error::transport(UnencodableHeader::Sent(value.clone())));
        }
        if let Some(unsendable) = unsendable_url(request.provider, &request.wire.url) {
            return Err(Error::transport(unsendable));
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
                    Phases {
                        connect: request
                            .timeout
                            .unwrap_or(Duration::from_secs(MESSAGES_CONNECT_TIMEOUT_SECS)),
                        read: timeout,
                    },
                    request.connect_retry,
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
        if let Some(fallback) = request.native_output_fallback.as_ref().filter(|_| {
            config.retries_without_native_output(
                failure.status,
                &failure.text,
                &body,
                &request.identity.model,
            )
        }) {
            let rebuilt = fallback.body(config)?;
            let replayed = replayed_recoveries(config, rebuilt, &recovered, request);
            body = std::borrow::Cow::Owned(replayed.body.clone());
            context.provider_request_retried(replayed).await?;
            continue;
        }
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
        recovered.push((failure.status, failure.text.clone()));
        tracing::debug!(
            "Anthropic /v1/messages: retrying after a recoverable HTTP error (attempt {attempt}/{MESSAGES_HTTP_ERROR_ATTEMPTS})."
        );
        body = std::borrow::Cow::Owned(retried.body.clone());
        context.provider_request_retried(retried).await?;
    }
}

fn replayed_recoveries(
    config: &dyn BaseMessagesConfig,
    rebuilt: Value,
    recovered: &[(u16, String)],
    request: &ProviderCall,
) -> litellm_host::interceptors::RetriedRequest {
    recovered.iter().fold(
        litellm_host::interceptors::RetriedRequest {
            reasons: Vec::new(),
            body: rebuilt,
        },
        |replayed, (status, text)| match config.retry_after_http_error(
            *status,
            text,
            &replayed.body,
            &request.transform_context,
        ) {
            Some(retried) => litellm_host::interceptors::RetriedRequest {
                reasons: replayed
                    .reasons
                    .into_iter()
                    .chain(retried.reasons)
                    .collect(),
                body: retried.body,
            },
            None => replayed,
        },
    )
}

async fn send(
    http: &litellm_http::Client,
    authenticated: Authenticated,
    url: &str,
    body: Bytes,
    phases: Phases,
    connect_retry: bool,
) -> Result<reqwest::Response, Error> {
    let request = serialized_outbound_request(authenticated, url, body, None)?;
    litellm_inference::outbound::send_in_phases(request, http, phases, connect_retry)
        .await
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
            return config
                .transform_unchecked_response(body)
                .map(MessagesBody::Unchecked)
                .map_err(Error::from)
                .map_err(Error::after_send);
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
    let unobserved_tail = decoder.map(|_| UnobservedTail::default());
    let chunks = match decoder.zip(unobserved_tail.clone()) {
        None => chunks.map_err(Error::from).boxed(),
        Some((decode, tail)) => decoded_chunks(
            chunks.map_err(std::io::Error::other).boxed(),
            decode,
            headers.clone(),
            provider,
            tail,
        ),
    };
    MessagesCallResponse::Stream {
        head: super::route::MessagesStreamHead {
            headers,
            detached_drain: None,
            unobserved_tail,
        },
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
    let unobserved_tail = decoder.map(|_| UnobservedTail::default());
    let chunks = match decoder.zip(unobserved_tail.clone()) {
        None => chunks.boxed(),
        Some((decode, tail)) => decoded_chunks(
            chunks.map_err(std::io::Error::other).boxed(),
            decode,
            headers.clone(),
            provider,
            tail,
        ),
    };
    MessagesCallResponse::Stream {
        head: super::route::MessagesStreamHead {
            headers,
            detached_drain: None,
            unobserved_tail,
        },
        chunks,
    }
}

fn draining_after_detach(
    response: MessagesCallResponse,
    decoder: Option<StreamDecoder>,
    detached_drain: Option<litellm_host::protocol::DetachedDrain>,
) -> MessagesCallResponse {
    match response {
        MessagesCallResponse::Stream { head, chunks } => MessagesCallResponse::Stream {
            head: super::route::MessagesStreamHead {
                detached_drain: decoder.and(detached_drain),
                ..head
            },
            chunks,
        },
        complete => complete,
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
    headers: Vec<(String, String)>,
    provider: &'static str,
    tail: UnobservedTail,
) -> BoxStream<'static, Result<Bytes, Error>> {
    let frames = decode(bytes, headers).map_err(Error::from).boxed();
    super::relay::relayed(frames, tail)
        .inspect_ok(move |chunk| log_chunk(provider, "client_response", chunk))
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
    use litellm_llms::base_llm::messages::streaming;
    use rstest::rstest;
    use serde_json::{Value, json};
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::any};

    use super::*;
    use crate::common_utils::messages_provider;

    #[rstest]
    #[case::anthropic(
        litellm_core_utils::get_llm_provider_logic::LlmProviders::Anthropic,
        false
    )]
    #[case::azure_ai(
        litellm_core_utils::get_llm_provider_logic::LlmProviders::AzureAi,
        false
    )]
    #[case::vertex_ai(
        litellm_core_utils::get_llm_provider_logic::LlmProviders::VertexAi,
        true
    )]
    fn only_vertex_ai_refuses_a_url_with_a_control_character(
        #[case] provider: litellm_core_utils::get_llm_provider_logic::LlmProviders,
        #[case] refused: bool,
    ) {
        let provider = crate::common_utils::messages_provider(provider, "claude-sonnet-4-5")
            .expect("provider");
        assert_eq!(
            super::unsendable_url(provider, "https://api.example.com/v1\t/messages").is_some(),
            refused
        );
    }

    #[rstest]
    #[case::event(
        "data: {\"type\":\"message_stop\"}\n\n",
        Some("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
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
            Some(|bytes, _| {
                streaming::event_stream(bytes)
                    .and_then(|event| futures_util::future::ready(streaming::encode(&event)))
                    .boxed()
            }),
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

    async fn upstream_stream(body: &'static str) -> reqwest::Response {
        let upstream = MockServer::start().await;
        Mock::given(any())
            .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
            .mount(&upstream)
            .await;
        litellm_http::Client::for_test()
            .get(upstream.uri())
            .send()
            .await
            .unwrap()
    }

    fn reencoding(bytes: ByteStream, _: Vec<(String, String)>) -> streaming::FrameStream {
        streaming::event_stream(bytes)
            .and_then(|event| futures_util::future::ready(streaming::encode(&event)))
            .boxed()
    }

    #[rstest]
    #[case::decoded_and_cut_short(
        Some(reencoding as StreamDecoder),
        "data: {\"type\":\"ping\"}\n\n",
        true
    )]
    #[case::decoded_and_stopped(
        Some(reencoding as StreamDecoder),
        "data: {\"type\":\"ping\"}\n\ndata: {\"type\":\"message_stop\"}\n\n",
        false
    )]
    #[case::passed_through_and_cut_short(None, "data: {\"type\":\"ping\"}\n\n", false)]
    #[tokio::test]
    async fn a_decoded_stream_cut_short_ends_with_the_incomplete_stream_error_on_its_tail(
        #[case] decoder: Option<StreamDecoder>,
        #[case] body: &'static str,
        #[case] incomplete: bool,
    ) {
        let relays = decoder.is_some();
        let MessagesCallResponse::Stream { head, mut chunks } = streaming_response(
            upstream_stream(body).await,
            decoder,
            "test",
            Duration::from_secs(5),
            false,
        ) else {
            panic!("a streaming response returns chunks");
        };
        let tail = head.unobserved_tail;
        assert_eq!(tail.is_some(), relays);
        let mut delivered = Vec::new();
        let mut on_tail = Vec::new();
        while let Some(chunk) = chunks.try_next().await.unwrap() {
            match tail.as_ref().is_some_and(UnobservedTail::begun) {
                true => on_tail.push(chunk),
                false => delivered.push(chunk),
            }
        }
        let error_event = Bytes::from(format!(
            "event: error\ndata: {{\"type\": \"error\", \"error\": {{\"type\": \"api_error\", \"message\": \"{}\"}}}}\n\n",
            super::super::relay::INCOMPLETE_STREAM_ERROR_MESSAGE
        ));
        assert!(!delivered.contains(&error_event));
        assert_eq!(
            on_tail,
            if incomplete {
                vec![error_event]
            } else {
                Vec::new()
            }
        );
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

    #[rstest]
    #[case::bedrock_mantle(litellm_core_utils::get_llm_provider_logic::LlmProviders::BedrockMantle, "anthropic.claude-sonnet-5-5", Some(json!({"input_tokens": 0, "output_tokens": 0})))]
    #[case::bedrock_mantle_route(litellm_core_utils::get_llm_provider_logic::LlmProviders::Bedrock, "mantle/anthropic.claude-sonnet-5-5", Some(json!({"input_tokens": 0, "output_tokens": 0})))]
    #[case::anthropic(
        litellm_core_utils::get_llm_provider_logic::LlmProviders::Anthropic,
        "claude-sonnet-5-5",
        None
    )]
    fn an_error_envelope_bills_zero_usage_only_on_mantle(
        #[case] provider: litellm_core_utils::get_llm_provider_logic::LlmProviders,
        #[case] model: &str,
        #[case] expected: Option<Value>,
    ) {
        let config = messages_provider(provider, model).unwrap().config();
        let envelope =
            r#"{"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}"#;
        let Ok(MessagesBody::Unchecked(body)) = decode_response(config, model, envelope) else {
            panic!("an error envelope decodes unchecked");
        };
        assert_eq!(body.get("usage").cloned(), expected);
    }
}
