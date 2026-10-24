use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use litellm_auth::{AuthServices, InputSource, SecretValue, Sourced};
use litellm_host::{
    call::HostedCompletion,
    interceptors::{
        Cost, ExecutionFacts, Interceptors, RawResponse, RequestContext, ResultSource, WireRequest,
    },
};
use litellm_host_native::in_process::{Host, run_hosted};
use litellm_http::{Client, transport::Error as TransportError};
use litellm_inference_testing::{RecordingSecrets, no_secrets};
use litellm_inference_tokenize::{Error, TokenizeRequest, TokenizeRoute};
use litellm_llms_types::formats::tokenize::TokenizeInclude;
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header, method, path},
};

#[fixture]
fn route() -> TokenizeRoute {
    TokenizeRoute::new(
        Client::plain_for_test(),
        Arc::new(AuthServices::default()),
        no_secrets(),
    )
}

fn sourced<T>(value: T) -> Sourced<T> {
    Sourced::new(value, InputSource::Request)
}

fn sourced_map(value: &Value) -> BTreeMap<String, Sourced<Value>> {
    sourced_map_at(value, InputSource::Request)
}

fn sourced_at<T>(value: T, source: InputSource) -> Sourced<T> {
    Sourced::new(value, source)
}

fn sourced_map_at(value: &Value, source: InputSource) -> BTreeMap<String, Sourced<Value>> {
    value
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, value)| (key.clone(), sourced_at(value.clone(), source)))
        .collect()
}

fn assert_credential_boundary_error(error: &Error) {
    assert!(error.is_request());
    assert!(
        error
            .to_string()
            .contains("request-controlled inputs cannot be combined with host credentials"),
        "{error:?}"
    );
}

#[rstest]
#[tokio::test]
async fn hosted_vllm_returns_a_typed_response(route: TokenizeRoute) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/tokenize"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_json(json!({
            "model":"org/model",
            "prompt":"hello",
            "add_special_tokens":true,
            "return_token_strs":true
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "count":1,
            "max_model_len":4096,
            "tokens":[42],
            "token_strs":["hello"]
        })))
        .expect(1)
        .mount(&server)
        .await;
    let result = route
        .execute(
            TokenizeRequest {
                model: "hosted_vllm/org/model".into(),
                prompt: "hello".into(),
                include: TokenizeInclude::new(true, true),
                api_base: Some(sourced(server.uri())),
                api_key: Some(sourced(SecretValue::new("test-key"))),
                ..Default::default()
            },
            &(),
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        serde_json::to_value(result).unwrap(),
        json!({
            "model":"org/model", "count":1, "token_ids":[42], "tokens":["hello"]
        })
    );
}

#[rstest]
#[tokio::test]
async fn request_hosted_vllm_endpoint_is_rejected_with_environment_credentials() {
    let server = MockServer::start().await;
    let route = TokenizeRoute::new(
        Client::plain_for_test(),
        Arc::new(AuthServices::default()),
        Arc::new(RecordingSecrets::new([(
            "HOSTED_VLLM_API_KEY",
            "host-api-key",
        )])),
    );
    let error = route
        .execute(
            TokenizeRequest {
                model: "hosted_vllm/org/model".into(),
                prompt: "hello".into(),
                api_base: Some(sourced(server.uri())),
                ..Default::default()
            },
            &(),
            None,
        )
        .await
        .unwrap_err();
    assert_credential_boundary_error(&error);
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[derive(Default)]
struct Hooks {
    context: Mutex<Option<RequestContext>>,
    raw: Mutex<Option<RawResponse>>,
    facts: Mutex<Option<ExecutionFacts>>,
}

impl Interceptors<Error> for Hooks {
    async fn before_provider_request(
        &self,
        mut wire: WireRequest,
        context: RequestContext,
    ) -> Result<WireRequest, Error> {
        *self.context.lock().unwrap() = Some(context);
        wire.body["prompt"] = json!("rewritten");
        Ok(wire)
    }
    async fn after_provider_response(&self, raw: RawResponse) -> Result<(), Error> {
        *self.raw.lock().unwrap() = Some(raw);
        Ok(())
    }
    async fn result_ready(&self, facts: ExecutionFacts) -> Result<(), Error> {
        *self.facts.lock().unwrap() = Some(facts);
        Ok(())
    }
}

#[rstest]
#[tokio::test]
async fn hosted_execution_is_lazy_and_applies_hooks_before_dispatch() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_json(json!({
            "model":"org/model", "prompt":"rewritten", "add_special_tokens":true, "return_token_strs":true
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"count":1,"tokens":[7],"token_strs":["hello"]})),
        )
        .expect(1)
        .mount(&server)
        .await;
    let secrets = Arc::new(RecordingSecrets::empty());
    let route = TokenizeRoute::new(
        Client::plain_for_test(),
        Arc::new(AuthServices::default()),
        secrets.clone(),
    );
    let machine = route.machine(
        TokenizeRequest {
            model: "org/model".into(),
            custom_llm_provider: Some("hosted_vllm".into()),
            prompt: "original".into(),
            include: TokenizeInclude::new(true, true),
            api_base: Some(sourced(server.uri())),
            ..Default::default()
        },
        None,
    );
    assert!(secrets.requested().is_empty());
    assert!(server.received_requests().await.unwrap().is_empty());
    let hooks = Hooks::default();
    let completion = run_hosted(
        machine,
        Host {
            services: &(),
            interceptors: &hooks,
            stream: &(),
            observers: None,
        },
    )
    .await
    .unwrap();
    let HostedCompletion::Complete(response) = completion else {
        panic!("tokenization must return a unary response")
    };
    assert_eq!(response.token_ids, Some(vec![7]));
    assert_eq!(
        hooks
            .context
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .custom_llm_provider,
        "hosted_vllm"
    );
    let facts = hooks.facts.lock().unwrap().clone().unwrap();
    assert_eq!(facts.provider.model, "org/model");
    assert_eq!(facts.provider.provider, "hosted_vllm");
    assert_eq!(facts.source, ResultSource::Provider);
    assert_eq!(facts.cost, Cost::Deferred);
    let raw = hooks.raw.lock().unwrap().clone().unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&raw.body).unwrap(),
        json!({"count":1,"tokens":[7],"token_strs":["hello"]})
    );
}

struct FailingPostCall;

impl Interceptors<Error> for FailingPostCall {
    async fn before_provider_request(
        &self,
        wire: WireRequest,
        _: RequestContext,
    ) -> Result<WireRequest, Error> {
        Ok(wire)
    }

    async fn after_provider_response(&self, _: RawResponse) -> Result<(), Error> {
        Err(Error::InvalidRequest("post-call callback failed".into()))
    }
}

#[rstest]
#[tokio::test]
async fn post_call_failures_are_not_request_failures_and_do_not_replay(route: TokenizeRoute) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"count":1,"tokens":[7]})))
        .expect(1)
        .mount(&server)
        .await;
    let error = route
        .execute(
            TokenizeRequest {
                model: "hosted_vllm/model".into(),
                prompt: "hello".into(),
                api_base: Some(sourced(server.uri())),
                ..Default::default()
            },
            &FailingPostCall,
            None,
        )
        .await
        .unwrap_err();
    assert!(!error.is_request());
    let Error::PostCallHook(cause) = error else {
        panic!("post-call errors must preserve their origin")
    };
    assert!(
        matches!(cause.as_ref(), Error::InvalidRequest(message) if message == &"post-call callback failed".into())
    );
}

#[rstest]
#[case::rate_limit(429)]
#[case::unavailable(503)]
#[tokio::test]
async fn upstream_status_is_preserved_without_replaying(route: TokenizeRoute, #[case] status: u16) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(status).set_body_json(json!({"error":"busy"})))
        .expect(1)
        .mount(&server)
        .await;
    let error = route
        .execute(
            TokenizeRequest {
                model: "hosted_vllm/model".into(),
                prompt: "hello".into(),
                api_base: Some(sourced(server.uri())),
                ..Default::default()
            },
            &(),
            None,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error,Error::Transport(TransportError::Http {status: actual,..}) if actual==status)
    );
}

#[rstest]
#[tokio::test]
async fn unsupported_deployment_defaults_are_filtered_before_provider_validation() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/tokenize"))
        .and(body_json(json!({
            "model": "model",
            "prompt": "hello",
            "add_special_tokens": true,
            "return_token_strs": true
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"count": 1, "tokens": [1]})))
        .expect(1)
        .mount(&server)
        .await;
    let error = route()
        .execute(
            TokenizeRequest {
                model: "hosted_vllm/model".into(),
                prompt: "hello".into(),
                api_base: Some(sourced(server.uri())),
                deployment_params: BTreeMap::from([(
                    "temperature".into(),
                    sourced_at(json!(0.2), InputSource::Deployment),
                )]),
                ..Default::default()
            },
            &(),
            None,
        )
        .await;
    assert!(error.is_ok());
}

#[rstest]
#[tokio::test]
async fn request_owned_values_cannot_be_smuggled_through_deployment_parameters() {
    let server = MockServer::start().await;
    let error = route()
        .execute(
            TokenizeRequest {
                model: "hosted_vllm/model".into(),
                api_base: Some(sourced(server.uri())),
                deployment_params: BTreeMap::from([("temperature".into(), sourced(json!(0.2)))]),
                ..Default::default()
            },
            &(),
            None,
        )
        .await
        .unwrap_err();
    assert!(error.is_request());
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[case::unsupported_provider("gemini/model", json!({}))]
#[case::bare_vllm_is_not_hosted_vllm("vllm/model", json!({}))]
#[case::unknown_option("hosted_vllm/model", json!({"messages":[]}))]
#[tokio::test]
async fn unsupported_calls_fail_before_dispatch(
    route: TokenizeRoute,
    #[case] model: &str,
    #[case] options: Value,
) {
    let server = MockServer::start().await;
    let error = route
        .execute(
            TokenizeRequest {
                model: model.into(),
                prompt: "hello".into(),
                api_base: Some(sourced(server.uri())),
                optional_params: sourced_map(&options),
                ..Default::default()
            },
            &(),
            None,
        )
        .await
        .unwrap_err();
    assert!(error.is_request());
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn directly_constructed_empty_model_is_rejected_before_dispatch(route: TokenizeRoute) {
    let error = route
        .execute(
            TokenizeRequest {
                model: String::new(),
                ..Default::default()
            },
            &(),
            None,
        )
        .await
        .unwrap_err();

    assert!(
        matches!(error, Error::InvalidRequest(message) if message.to_string() == "model must not be empty")
    );
}

#[rstest]
#[case::enabled(true)]
#[case::disabled(false)]
#[tokio::test]
async fn top_level_special_token_policy_reaches_vllm_wire(
    route: TokenizeRoute,
    #[case] add_special_tokens: bool,
) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/tokenize"))
        .and(body_json(json!({
            "model":"org/model",
            "prompt":"hello",
            "add_special_tokens":add_special_tokens,
            "return_token_strs":true
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "count": 1,
            "tokens": [42],
            "token_strs": ["hello"]
        })))
        .expect(1)
        .mount(&server)
        .await;

    route
        .execute(
            TokenizeRequest {
                model: "hosted_vllm/org/model".into(),
                prompt: "hello".into(),
                add_special_tokens: Some(add_special_tokens),
                api_base: Some(sourced(server.uri())),
                deployment_params: sourced_map_at(
                    &json!({"add_special_tokens": !add_special_tokens}),
                    InputSource::Deployment,
                ),
                ..Default::default()
            },
            &(),
            None,
        )
        .await
        .unwrap();
}

#[rstest]
#[tokio::test]
async fn duplicate_special_token_policy_is_rejected_before_provider_io(route: TokenizeRoute) {
    let server = MockServer::start().await;
    let error = route
        .execute(
            TokenizeRequest {
                model: "hosted_vllm/org/model".into(),
                add_special_tokens: Some(true),
                api_base: Some(sourced(server.uri())),
                optional_params: BTreeMap::from([(
                    "add_special_tokens".into(),
                    sourced(json!(false)),
                )]),
                ..Default::default()
            },
            &(),
            None,
        )
        .await
        .unwrap_err();

    assert!(error.to_string().contains("both"));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn count_output_is_projected_without_becoming_a_provider_parameter(route: TokenizeRoute) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/tokenize"))
        .and(body_json(json!({
            "model":"org/model",
            "prompt":"hello",
            "add_special_tokens":true,
            "return_token_strs":true
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "count":1,
            "max_model_len":4096,
            "tokens":[42],
            "token_strs":["hello"]
        })))
        .expect(1)
        .mount(&server)
        .await;
    let response = route
        .execute(
            TokenizeRequest {
                model: "hosted_vllm/org/model".into(),
                prompt: "hello".into(),
                api_base: Some(sourced(server.uri())),
                ..Default::default()
            },
            &(),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response.count, 1);
    assert_eq!(response.token_ids, None);
    assert_eq!(response.tokens, None);
}

#[rstest]
#[tokio::test]
async fn strict_provider_output_rejects_missing_but_preserves_empty_representations(
    route: TokenizeRoute,
) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/tokenize"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "count":0,
            "max_model_len":4096,
            "tokens":[],
            "token_strs":[]
        })))
        .expect(1)
        .mount(&server)
        .await;
    let response = route
        .execute(
            TokenizeRequest {
                model: "hosted_vllm/org/model".into(),
                prompt: "".into(),
                include: TokenizeInclude::new(true, true),
                api_base: Some(sourced(server.uri())),
                ..Default::default()
            },
            &(),
            None,
        )
        .await
        .unwrap();
    assert_eq!(response.token_ids, Some(Vec::new()));
    assert_eq!(response.tokens, Some(Vec::new()));

    let missing = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/tokenize"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "count":1,
            "max_model_len":4096,
            "tokens":[42],
            "token_strs":null
        })))
        .expect(1)
        .mount(&missing)
        .await;
    let error = route
        .execute(
            TokenizeRequest {
                model: "hosted_vllm/org/model".into(),
                prompt: "hello".into(),
                include: TokenizeInclude::new(false, true),
                api_base: Some(sourced(missing.uri())),
                ..Default::default()
            },
            &(),
            None,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidResponse(_)));
}
