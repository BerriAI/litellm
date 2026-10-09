mod support;

use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_gateway_auth::AuthenticatedRequest;
use litellm_gateway_inference::{Error, GatewayLayer};
use litellm_host::{
    error::HookError,
    hooks::NativeHooks,
    interceptors::{RequestContext, WireRequest},
};
use rstest::rstest;
use serde_json::{Map, Value, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_partial_json, header, method},
};

struct SetMaxTokens(u64);

impl GatewayLayer for SetMaxTokens {
    fn pre_call<'a>(
        &'a self,
        _: &'a AuthenticatedRequest,
        body: Map<String, Value>,
    ) -> BoxFuture<'a, Result<Map<String, Value>, Error>> {
        Box::pin(async move {
            Ok(body
                .into_iter()
                .chain([("max_tokens".into(), json!(self.0))])
                .collect())
        })
    }
}

struct DoubleMaxTokens;

impl GatewayLayer for DoubleMaxTokens {
    fn pre_call<'a>(
        &'a self,
        _: &'a AuthenticatedRequest,
        mut body: Map<String, Value>,
    ) -> BoxFuture<'a, Result<Map<String, Value>, Error>> {
        Box::pin(async move {
            let doubled = body
                .get("max_tokens")
                .and_then(Value::as_u64)
                .unwrap_or_default()
                * 2;
            body.insert("max_tokens".into(), json!(doubled));
            Ok(body)
        })
    }
}

struct RejectPreCall;

impl GatewayLayer for RejectPreCall {
    fn pre_call<'a>(
        &'a self,
        _: &'a AuthenticatedRequest,
        _: Map<String, Value>,
    ) -> BoxFuture<'a, Result<Map<String, Value>, Error>> {
        Box::pin(async {
            Err(HookError::Rejected {
                reason: "blocked by policy".into(),
            }
            .into())
        })
    }
}

struct WireMarker {
    reject: bool,
}

impl NativeHooks for WireMarker {
    fn before_provider_request(
        &self,
        mut wire: Box<WireRequest>,
        _: &RequestContext,
    ) -> Result<Box<WireRequest>, HookError> {
        if self.reject {
            return Err(HookError::Rejected {
                reason: "wire refused".into(),
            });
        }
        wire.headers.push(("x-hooked".into(), "1".into()));
        Ok(wire)
    }
}

struct CallLayer {
    reject: bool,
}

impl GatewayLayer for CallLayer {
    fn call_hooks(&self) -> Option<Box<dyn NativeHooks>> {
        Some(Box::new(WireMarker {
            reject: self.reject,
        }))
    }
}

fn provider_reply() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "msg_test", "model": "test-model", "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}
    }))
}

fn chat() -> Value {
    json!({"model": "public/model", "messages": [{"role": "user", "content": "hi"}]})
}

#[rstest]
#[case::set_then_double(vec![Arc::new(SetMaxTokens(3)) as Arc<dyn GatewayLayer>, Arc::new(DoubleMaxTokens)], 6)]
#[case::double_then_set(vec![Arc::new(DoubleMaxTokens) as Arc<dyn GatewayLayer>, Arc::new(SetMaxTokens(3))], 3)]
#[tokio::test]
async fn pre_call_layers_rewrite_the_body_in_stack_order(
    #[case] layers: Vec<Arc<dyn GatewayLayer>>,
    #[case] max_tokens: u64,
) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_partial_json(json!({"max_tokens": max_tokens})))
        .respond_with(provider_reply())
        .expect(1)
        .mount(&upstream)
        .await;
    let app = support::app_with_layers("anthropic/test-model", &upstream.uri(), layers);
    let response = support::post(app, "/v1/chat/completions", chat()).await;
    assert_eq!(response.status(), 200);
}

#[tokio::test]
async fn a_rejecting_pre_call_layer_stops_the_request_before_the_provider() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(provider_reply())
        .expect(0)
        .mount(&upstream)
        .await;
    let app = support::app_with_layers(
        "anthropic/test-model",
        &upstream.uri(),
        vec![Arc::new(RejectPreCall)],
    );
    let response = support::post(app, "/v1/chat/completions", chat()).await;
    assert_eq!(response.status(), 400);
    assert!(
        support::json(response).await["error"]["message"]
            .as_str()
            .unwrap()
            .contains("blocked by policy")
    );
}

#[tokio::test]
async fn call_hooks_rewrite_the_request_the_provider_receives() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(header("x-hooked", "1"))
        .respond_with(provider_reply())
        .expect(1)
        .mount(&upstream)
        .await;
    let app = support::app_with_layers(
        "anthropic/test-model",
        &upstream.uri(),
        vec![Arc::new(CallLayer { reject: false })],
    );
    let response = support::post(app, "/v1/chat/completions", chat()).await;
    assert_eq!(response.status(), 200);
}

#[tokio::test]
async fn a_rejecting_call_hook_stops_the_request_before_the_provider() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(provider_reply())
        .expect(0)
        .mount(&upstream)
        .await;
    let app = support::app_with_layers(
        "anthropic/test-model",
        &upstream.uri(),
        vec![Arc::new(CallLayer { reject: true })],
    );
    let response = support::post(app, "/v1/chat/completions", chat()).await;
    assert_eq!(response.status(), 400);
}
