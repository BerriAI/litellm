use std::time::Duration;

use litellm_auth::AuthServices;
use litellm_host::{
    event::{MachineEvent, RawResponse, RequestContext, WireRequest},
    hooks::RouteHooks,
};
use litellm_http::{Client, outbound::OutboundRequest, request::truncate_error_body};
use litellm_llms::base_llm::{
    auth::{Authenticated, resolve_auth},
    chat::transformation::ProviderChatResponseData,
};
use litellm_types::utils::ChatCompletionsResponse;
use serde_json::Value;

use super::Error;
use crate::{
    chat_completions::types::ProviderChatCompletionsRequest,
    constants::CHAT_COMPLETIONS_TIMEOUT_SECS,
};

pub(super) async fn execute(
    http: &Client,
    auth: &AuthServices,
    request: ProviderChatCompletionsRequest,
    hooks: &impl RouteHooks<Error>,
) -> Result<ChatCompletionsResponse, Error> {
    let ProviderChatCompletionsRequest {
        model,
        custom_llm_provider,
        config,
        url,
        body,
        optional_params,
        environment,
        timeout,
        api_key,
    } = request;
    let context = RequestContext {
        model: model.clone(),
        custom_llm_provider,
        optional_params: Value::Object(optional_params),
        secret_fields: Vec::new(),
        api_key,
    };
    let authenticated = resolve_auth(auth, environment, &|key| std::env::var(key).ok()).await?;
    let wire = hooks
        .before_send(
            WireRequest {
                url,
                headers: authenticated.headers,
                body,
            },
            context,
        )
        .await?;
    let outbound = outbound_request(
        Authenticated {
            headers: wire.headers,
            signer: authenticated.signer,
        },
        wire.url,
        &wire.body,
        timeout,
    )?;

    let response = outbound.send(http).await.map_err(|err| {
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
    hooks
        .emit(MachineEvent::ResponseReceived {
            raw: RawResponse { body: text.clone() },
        })
        .await?;

    let body: Value = serde_json::from_str(&text).map_err(|err| {
        Error::InvalidResponse(format!("invalid chat completions response JSON: {err}"))
    })?;
    config
        .transform_response(&model, ProviderChatResponseData { body })
        .map_err(Error::from)
        .map_err(as_response_error)
}

/// Re-tag an error raised while normalizing a response the provider already
/// returned.
///
/// A config reports the same variants on either side of the call: a missing
/// field or an unsupported block can mean "this request cannot be translated"
/// during prepare and "this response cannot be normalized" here. Only the
/// second kind has already been billed, and a host that keeps a reference
/// implementation must not retry those, so collapse them to one variant that
/// can only mean the provider was already called.
pub(super) fn as_response_error(err: Error) -> Error {
    match err {
        already @ (Error::InvalidResponse(_)
        | Error::Transport(litellm_http::transport::Error::Http { .. })) => already,
        other => Error::InvalidResponse(other.to_string()),
    }
}

fn outbound_request(
    authenticated: Authenticated,
    url: String,
    body: &Value,
    timeout: Option<Duration>,
) -> Result<OutboundRequest, Error> {
    crate::outbound::outbound_request(
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
    use crate::chat_completions::{
        prepare::{prepare_provider_request, resolve_request},
        types::ChatCompletionsRequest,
    };

    const ANTHROPIC_MESSAGE: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":11,"output_tokens":4}}"#;

    /// Rewrites the outgoing request and records what the call reports back.
    #[derive(Default)]
    struct RecordingHooks {
        contexts: Mutex<Vec<RequestContext>>,
        raw: Mutex<Vec<String>>,
    }

    impl RouteHooks<Error> for RecordingHooks {
        async fn before_send(
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

        async fn emit(&self, event: MachineEvent) -> Result<(), Error> {
            let MachineEvent::ResponseReceived { raw } = event;
            self.raw.lock().unwrap().push(raw.body);
            Ok(())
        }
    }

    fn prepared(api_base: &str) -> ProviderChatCompletionsRequest {
        prepare_provider_request(
            resolve_request(ChatCompletionsRequest {
                model: "anthropic/claude-sonnet-4-5",
                messages: json!([{"role": "user", "content": "hi"}]),
                optional_params: json!({"max_tokens": 16}).as_object().unwrap().clone(),
                api_key: Some("sk-test"),
                api_base: Some(api_base),
                custom_llm_provider: None,
                extra_headers: None,
                timeout: None,
            })
            .unwrap(),
        )
        .unwrap()
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
        let hooks = RecordingHooks::default();

        execute(
            &Client::plain_for_test(),
            &AuthServices::default(),
            prepared(&upstream.uri()),
            &hooks,
        )
        .await
        .expect("chat completions call succeeds");

        let [request] = <[Request; 1]>::try_from(upstream.received_requests().await.unwrap())
            .unwrap_or_else(|requests| panic!("one request, saw {}", requests.len()));
        let sent: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(sent["system"], "added by the host");
        assert_eq!(request.headers["x-host"], "seen");
        assert_eq!(request.headers["x-api-key"], "sk-test");
        let [context] = <[RequestContext; 1]>::try_from(hooks.contexts.into_inner().unwrap())
            .unwrap_or_else(|seen| panic!("before_send runs once, saw {}", seen.len()));
        assert_eq!(
            (context.model.as_str(), context.custom_llm_provider.as_str()),
            ("claude-sonnet-4-5", "anthropic")
        );
        assert_eq!(context.optional_params, json!({"max_tokens": 16}));
        assert_eq!(hooks.raw.into_inner().unwrap(), [ANTHROPIC_MESSAGE]);
    }

    #[rstest]
    #[tokio::test]
    async fn an_upstream_failure_is_not_reported_as_a_received_response() {
        let upstream = MockServer::start().await;
        Mock::given(any())
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .mount(&upstream)
            .await;
        let hooks = RecordingHooks::default();

        let error = execute(
            &Client::plain_for_test(),
            &AuthServices::default(),
            prepared(&upstream.uri()),
            &hooks,
        )
        .await
        .err()
        .expect("the upstream failure fails the call");

        assert!(matches!(
            error,
            Error::Transport(litellm_http::transport::Error::Http { status: 500, .. })
        ));
        assert!(hooks.raw.into_inner().unwrap().is_empty());
    }

    #[test]
    fn response_errors_collapse_to_one_variant_that_can_only_mean_already_sent() {
        for original in [
            Error::MissingField("usage"),
            Error::Unsupported("non-text response content block"),
            Error::InvalidRequest("whatever".to_string()),
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
