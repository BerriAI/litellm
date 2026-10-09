use litellm_host::{lifecycle::ExecutionEvent, observation::ObservationSender};
use std::time::Duration;

use litellm_auth::AuthServices;
use litellm_host::interceptors::{Interceptors, RawResponse, RequestContext, WireRequest};
use litellm_http::{
    Client,
    outbound::OutboundRequest,
    request::{truncate_error_body, with_default_headers},
};
use litellm_llms::base_llm::{
    auth::{Authenticated, ValidatedEnvironment, resolve_auth},
    chat::transformation::BaseConfig,
    chat::transformation::ProviderChatResponseData,
};
use litellm_llms_types::formats::chat_completions::ChatCompletionsResponse;
use litellm_secrets::source::Secrets;
use serde_json::Value;

use super::Error;
use crate::{
    common_utils::string_headers, constants::CHAT_COMPLETIONS_TIMEOUT_SECS,
    types::ResolvedChatCompletionsRequest,
};

#[allow(clippy::too_many_arguments)] // Required by the shared route execution signature.
pub(super) async fn execute(
    http: &Client,
    auth: &AuthServices,
    config: &'static dyn BaseConfig,
    provider: litellm_inference::provider::ResolvedProvider<'_>,
    call: ResolvedChatCompletionsRequest,
    secrets: Secrets,
    cache: Option<litellm_cache_response::ScopedCache>,
    cache_options: Option<litellm_cache_response::CachePolicy>,
    interceptors: &impl Interceptors<Error>,
    observers: Option<&ObservationSender>,
) -> Result<ChatCompletionsResponse, Error> {
    let validated = config.validate_environment(
        string_headers(call.extra_headers)?,
        call.api_key.as_deref(),
        provider.model,
        &call.optional_params,
        &|key| secrets.get(key),
    )?;
    let url = config.get_complete_url(
        call.api_base.as_deref(),
        provider.model,
        &call.optional_params,
        &|key| secrets.get(key),
    )?;
    let body = config
        .transform_request(provider.model, call.messages, call.optional_params.clone())?
        .body;
    let environment = ValidatedEnvironment {
        headers: with_default_headers(validated.headers, config.default_headers()),
        auth: validated.auth,
    };
    let provider_name = <&'static str>::from(provider.provider);
    let context = RequestContext {
        model: provider.model.to_owned(),
        custom_llm_provider: provider_name.to_owned(),
        optional_params: Value::Object(call.optional_params),
        secret_fields: Vec::new(),
        api_key: call.api_key.map(litellm_auth::SecretValue::new),
    };
    let authenticated = resolve_auth(auth, environment, &|key| secrets.get(key)).await?;
    let identity = litellm_host::interceptors::ProviderIdentity {
        model: context.model.clone(),
        provider: context.custom_llm_provider.clone(),
    };
    let wire = interceptors
        .before_provider_request(
            WireRequest {
                url,
                headers: authenticated.headers,
                body,
            },
            context,
        )
        .await?;
    let cache = cache.filter(|_| authenticated.signer.is_none());
    let cache_request = litellm_inference::caching::CacheRequest::from_wire(
        identity,
        cache.as_ref().map(|_| &wire),
    );
    litellm_inference::caching::execute_unary::<super::route::ChatCompletions, _, _>(
        cache_request,
        cache.as_ref().map(|cache| cache.service.clone()),
        cache.as_ref().map(|cache| cache.options(cache_options)),
        interceptors,
        observers,
        || async move {
            let outbound = outbound_request(
                Authenticated {
                    headers: wire.headers,
                    signer: authenticated.signer,
                },
                wire.url,
                &wire.body,
                call.timeout,
            )?;

            let response = litellm_inference::outbound::send(outbound, http)
                .await
                .map_err(|err| {
                    // Failing to establish the connection means the request never went out,
                    // so the host can still serve it. Everything else here, a timeout
                    // above all, may have reached the provider and been answered.
                    if err.is_connect() || err.is_builder() {
                        Error::Transport(litellm_http::transport::Error::Connect(err.to_string()))
                    } else {
                        Error::Transport(litellm_http::transport::Error::Network(err.to_string()))
                    }
                })?;

            let status = response.status();
            let text = response.text().await.map_err(|err| {
                Error::Transport(litellm_http::transport::Error::Network(err.to_string()))
            })?;

            if !status.is_success() {
                return Err(Error::Transport(litellm_http::transport::Error::Http {
                    status: status.as_u16(),
                    body: truncate_error_body(&text),
                }));
            }
            let raw = RawResponse { body: text.clone() };
            if let Some(observers) = observers {
                observers.emit(litellm_host::lifecycle::CallEvent::Execution(
                    ExecutionEvent::ProviderResponseReceived { raw: raw.clone() },
                ));
            }
            interceptors
                .after_provider_response(raw)
                .await
                .map_err(Error::post_call)?;

            let body: Value = serde_json::from_str(&text).map_err(|err| {
                Error::InvalidResponse(litellm_llms::ErrorDetail::invalid(
                    "chat completions response JSON",
                    err,
                ))
            })?;
            config
                .transform_response(provider.model, ProviderChatResponseData { body })
                .map_err(Error::from)
                .map_err(as_response_error)
        },
    )
    .await
}

/// Re-tag an error raised while normalizing a response the provider already
/// returned.
///
/// A config reports the same variants on either side of the call: a missing
/// field or an unsupported block can mean "this request cannot be translated"
/// before the request is sent and "this response cannot be normalized" here. Only the
/// second kind has already been billed, and a host that keeps a reference
/// implementation must not retry those, so collapse them to one variant that
/// can only mean the provider was already called.
pub(super) fn as_response_error(err: Error) -> Error {
    match err {
        already @ (Error::InvalidResponse(_)
        | Error::Transport(litellm_http::transport::Error::Http { .. })) => already,
        other => Error::InvalidResponse(other.to_string().into()),
    }
}

pub(super) fn outbound_request(
    authenticated: Authenticated,
    url: String,
    body: &Value,
    timeout: Option<Duration>,
) -> Result<OutboundRequest, Error> {
    litellm_inference::outbound::outbound_request(
        authenticated,
        url,
        body,
        Some(timeout.unwrap_or(Duration::from_secs(CHAT_COMPLETIONS_TIMEOUT_SECS))),
    )
    .map_err(|error| match error {
        // Python drops the caller's copy and prefers a forwarded Authorization
        // over the signature, so leave the request to it.
        litellm_http::Error::ComputedHeader(_) => {
            Error::Unsupported("request forwards a header AWS SigV4 computes")
        }
        other => Error::Http(other),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use rstest::rstest;
    use serde_json::json;
    use wiremock::{Mock, MockServer, Request, ResponseTemplate, matchers::any};

    use super::*;
    use crate::{
        common_utils::resolve_request,
        provider_config::resolve_provider_config,
        types::{ChatCompletionsCall, ResolvedChatCompletionsRequest},
    };

    const ANTHROPIC_MESSAGE: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":11,"output_tokens":4}}"#;

    /// Rewrites the outgoing request and records what the call reports back.
    #[derive(Default)]
    struct RecordingHooks {
        contexts: Mutex<Vec<RequestContext>>,
        raw: Mutex<Vec<String>>,
    }

    impl Interceptors<Error> for RecordingHooks {
        async fn before_provider_request(
            &self,
            wire: WireRequest,
            context: RequestContext,
        ) -> Result<WireRequest, Error> {
            self.contexts.lock().unwrap().push(context);
            let mut body = wire.body;
            body["system"] = json!("added by the host");
            Ok(WireRequest {
                headers: wire
                    .headers
                    .into_iter()
                    .chain([("x-host".to_string(), "seen".to_string())])
                    .collect(),
                body,
                ..wire
            })
        }

        async fn after_provider_response(&self, raw: RawResponse) -> Result<(), Error> {
            self.raw.lock().unwrap().push(raw.body);
            Ok(())
        }
    }

    fn prepared(
        api_base: &str,
    ) -> (
        litellm_inference::provider::ResolvedProvider<'static>,
        &'static dyn BaseConfig,
        ResolvedChatCompletionsRequest,
    ) {
        let (provider, config) =
            resolve_provider_config("anthropic/claude-sonnet-4-5", None).unwrap();
        let request = resolve_request(
            ChatCompletionsCall {
                model: "anthropic/claude-sonnet-4-5".into(),
                messages: json!([{"role": "user", "content": "hi"}]),
                optional_params: json!({"max_tokens": 16}).as_object().unwrap().clone(),
                api_key: Some("sk-test".into()),
                api_base: Some(api_base.into()),
                custom_llm_provider: None,
                extra_headers: None,
                timeout: None,
            },
            config,
        )
        .unwrap();
        (provider, config, request)
    }

    #[rstest::rstest]
    fn prepares_a_bedrock_call_without_resolving_credentials() {
        let (provider, config) =
            resolve_provider_config("bedrock/us-east-1/anthropic.claude-v2", None).unwrap();
        let request = resolve_request(
            ChatCompletionsCall {
                model: "bedrock/us-east-1/anthropic.claude-v2".into(),
                messages: json!([{"role": "user", "content": "hi"}]),
                optional_params: json!({"maxTokens": 16}).as_object().unwrap().clone(),
                api_key: None,
                api_base: None,
                custom_llm_provider: None,
                extra_headers: None,
                timeout: None,
            },
            config,
        )
        .unwrap();
        let environment = config
            .validate_environment(
                Vec::new(),
                request.api_key.as_deref(),
                provider.model,
                &request.optional_params,
                &|_| None,
            )
            .unwrap();
        let url = config
            .get_complete_url(
                request.api_base.as_deref(),
                provider.model,
                &request.optional_params,
                &|_| None,
            )
            .unwrap();
        let body = config
            .transform_request(
                provider.model,
                request.messages,
                request.optional_params.clone(),
            )
            .unwrap()
            .body;
        assert_eq!(
            url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/converse"
        );
        assert!(matches!(
            &environment.auth,
            litellm_llms::base_llm::auth::AuthScheme::AwsSigV4 {
                region,
                service: "bedrock",
                ..
            } if region == "us-east-1"
        ));
        assert!(
            !environment
                .headers
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case("authorization"))
        );
        assert_eq!(body["inferenceConfig"], json!({"maxTokens": 16}));
    }

    #[rstest]
    #[tokio::test]
    async fn the_hooks_rewrite_the_wire_request_and_see_the_raw_response() {
        let upstream = MockServer::start().await;
        Mock::given(any())
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(ANTHROPIC_MESSAGE, "application/json"),
            )
            .mount(&upstream)
            .await;
        let interceptors = RecordingHooks::default();

        let (provider, config, request) = prepared(&upstream.uri());
        execute(
            &Client::plain_for_test(),
            &AuthServices::default(),
            config,
            provider,
            request,
            std::sync::Arc::new(|_: &str| None),
            None,
            None,
            &interceptors,
            None,
        )
        .await
        .expect("chat completions call succeeds");

        let [request] = <[Request; 1]>::try_from(upstream.received_requests().await.unwrap())
            .unwrap_or_else(|requests| panic!("one request, saw {}", requests.len()));
        let sent: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(sent["system"], "added by the host");
        assert_eq!(request.headers["x-host"], "seen");
        assert_eq!(request.headers["x-api-key"], "sk-test");
        let [context] =
            <[RequestContext; 1]>::try_from(interceptors.contexts.into_inner().unwrap())
                .unwrap_or_else(|seen| {
                    panic!("before_provider_request runs once, saw {}", seen.len())
                });
        assert_eq!(
            (context.model.as_str(), context.custom_llm_provider.as_str()),
            ("claude-sonnet-4-5", "anthropic")
        );
        assert_eq!(context.optional_params, json!({"max_tokens": 16}));
        assert_eq!(interceptors.raw.into_inner().unwrap(), [ANTHROPIC_MESSAGE]);
    }

    #[rstest]
    #[tokio::test]
    async fn an_upstream_failure_is_not_reported_as_a_received_response() {
        let upstream = MockServer::start().await;
        Mock::given(any())
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .mount(&upstream)
            .await;
        let interceptors = RecordingHooks::default();

        let (provider, config, request) = prepared(&upstream.uri());
        let error = execute(
            &Client::plain_for_test(),
            &AuthServices::default(),
            config,
            provider,
            request,
            std::sync::Arc::new(|_: &str| None),
            None,
            None,
            &interceptors,
            None,
        )
        .await
        .expect_err("the upstream failure fails the call");

        assert!(matches!(
            error,
            Error::Transport(litellm_http::transport::Error::Http { status: 500, .. })
        ));
        assert!(interceptors.raw.into_inner().unwrap().is_empty());
    }

    #[rstest::rstest]
    fn response_errors_collapse_to_one_variant_that_can_only_mean_already_sent() {
        for original in [
            Error::MissingField("usage"),
            Error::Unsupported("non-text response content block"),
            Error::InvalidRequest("whatever".to_string().into()),
            Error::Auth(litellm_auth::Error::InvalidHeader),
        ] {
            let label = format!("{original:?}");
            assert!(
                matches!(as_response_error(original), Error::InvalidResponse(_)),
                "{label} must not stay retryable once the provider has answered"
            );
        }
        let upstream = Error::Transport(litellm_http::transport::Error::Http {
            status: 500,
            body: "boom".to_string(),
        });
        assert_eq!(as_response_error(upstream.clone()), upstream);
    }
}
