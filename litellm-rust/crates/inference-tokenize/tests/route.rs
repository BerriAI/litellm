use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use litellm_auth::{AuthServices, InputSource, SecretValue, Sourced};
use litellm_host::{
    call::HostedCompletion,
    interceptors::{ExecutionFacts, Interceptors, RawResponse, RequestContext, WireRequest},
};
use litellm_host_native::in_process::{Host, run_hosted};
use litellm_http::{Client, transport::Error as TransportError};
use litellm_inference_testing::{RecordingSecrets, no_secrets};
use litellm_inference_tokenize::{Error, TokenizeRequest, TokenizeRoute};
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
#[case::vllm("hosted_vllm/org/model", "/tokenize", json!({}),
    json!({"model":"org/model","prompt":"hello","add_special_tokens":true,"return_token_strs":true}),
    json!({"count":1,"max_model_len":4096,"tokens":[42],"token_strs":["hello"]}),
    json!([42]))]
#[case::watsonx("watsonx/org/model", "/ml/v1/text/tokenization",
    json!({"project_id":"project","token":"test-key"}),
    json!({"model_id":"org/model","input":"hello","project_id":"project","parameters":{"return_tokens":true}}),
    json!({"model_id":"org/model","result":{"token_count":1,"tokens":["hello"]}}),
    Value::Null)]
#[tokio::test]
async fn providers_share_a_typed_response(
    route: TokenizeRoute,
    #[case] model: &str,
    #[case] endpoint: &str,
    #[case] params: Value,
    #[case] request: Value,
    #[case] response: Value,
    #[case] token_ids: Value,
) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(endpoint))
        .and(header("authorization", "Bearer test-key"))
        .and(body_json(request))
        .respond_with(ResponseTemplate::new(200).set_body_json(response))
        .expect(1)
        .mount(&server)
        .await;
    let result = route
        .execute(
            TokenizeRequest {
                model: model.into(),
                prompt: "hello".into(),
                api_base: Some(sourced(server.uri())),
                api_key: Some(sourced(SecretValue::new("test-key"))),
                optional_params: sourced_map(&params),
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
            "model":"org/model", "count":1, "token_ids":token_ids, "tokens":["hello"]
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
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"count":1,"tokens":[7]})))
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
    assert_eq!(
        hooks.facts.lock().unwrap().as_ref().unwrap().provider.model,
        "org/model"
    );
    assert!(
        hooks
            .raw
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .body
            .contains("tokens")
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
                model: "vllm/model".into(),
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
async fn rate_limit_metadata_is_preserved_without_forwarding_sensitive_headers(
    route: TokenizeRoute,
) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "7")
                .insert_header("x-request-id", "request-429")
                .insert_header("authorization", "Bearer private"),
        )
        .expect(1)
        .mount(&server)
        .await;
    let error = route
        .execute(
            TokenizeRequest {
                model: "vllm/model".into(),
                prompt: "hello".into(),
                api_base: Some(sourced(server.uri())),
                ..Default::default()
            },
            &(),
            None,
        )
        .await
        .unwrap_err();
    let Error::Transport(TransportError::Http {
        status, headers, ..
    }) = error
    else {
        panic!("expected upstream HTTP error")
    };
    assert_eq!(status, 429);
    assert_eq!(headers.get("retry-after").unwrap(), "7");
    assert_eq!(headers.get("x-request-id").unwrap(), "request-429");
    assert!(!headers.contains_key("authorization"));
}

#[rstest]
#[case::project_id("project_id", json!("caller-project"))]
#[case::project_alias("watsonx_project", json!("caller-project"))]
#[case::space_id("space_id", json!("caller-space"))]
#[case::api_version("api_version", json!("2025-01-01"))]
#[case::endpoint("api_base", json!("https://caller.example"))]
#[tokio::test]
async fn request_watsonx_resources_are_rejected_with_environment_credentials(
    #[case] field: &str,
    #[case] value: Value,
) {
    let server = MockServer::start().await;
    let api_base = server.uri();
    let route = TokenizeRoute::new(
        Client::plain_for_test(),
        Arc::new(AuthServices::default()),
        Arc::new(RecordingSecrets::new([
            ("WATSONX_API_BASE", api_base.as_str()),
            ("WATSONX_API_KEY", "host-api-key"),
            ("WATSONX_PROJECT_ID", "host-project"),
        ])),
    );
    let request = if field == "api_base" {
        TokenizeRequest {
            model: "watsonx/model".into(),
            api_base: Some(sourced(value.as_str().unwrap().to_owned())),
            ..Default::default()
        }
    } else {
        TokenizeRequest {
            model: "watsonx/model".into(),
            optional_params: BTreeMap::from([(field.to_owned(), sourced(value))]),
            ..Default::default()
        }
    };
    let error = route.execute(request, &(), None).await.unwrap_err();
    assert_credential_boundary_error(&error);
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn shadowed_request_alias_is_validated_before_precedence() {
    let server = MockServer::start().await;
    let api_base = server.uri();
    let route = TokenizeRoute::new(
        Client::plain_for_test(),
        Arc::new(AuthServices::default()),
        Arc::new(RecordingSecrets::new([
            ("WATSONX_API_BASE", api_base.as_str()),
            ("WATSONX_API_KEY", "host-api-key"),
            ("WATSONX_PROJECT_ID", "host-project"),
        ])),
    );
    let error = route
        .execute(
            TokenizeRequest {
                model: "watsonx/model".into(),
                deployment_params: BTreeMap::from([(
                    "project_id".into(),
                    sourced_at(json!("trusted-project"), InputSource::Deployment),
                )]),
                optional_params: BTreeMap::from([(
                    "watsonx_project".into(),
                    sourced(json!("caller-project")),
                )]),
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

#[rstest]
#[case::token_and_zen("token", "zen_api_key")]
#[case::token_and_authorization("token", "Authorization")]
#[case::zen_and_authorization("zen_api_key", "Authorization")]
#[tokio::test]
async fn mixed_credential_origins_fail_before_provider_io(
    #[case] deployment_credential: &str,
    #[case] request_credential: &str,
) {
    let server = MockServer::start().await;
    let api_base = server.uri();
    let route = TokenizeRoute::new(
        Client::plain_for_test(),
        Arc::new(AuthServices::default()),
        Arc::new(RecordingSecrets::new([(
            "WATSONX_API_BASE",
            api_base.as_str(),
        )])),
    );
    let params = BTreeMap::from([
        (
            deployment_credential.to_owned(),
            sourced_at(json!("host-secret"), InputSource::Deployment),
        ),
        (
            "project_id".into(),
            sourced_at(json!("host-project"), InputSource::Deployment),
        ),
    ]);
    let (extra_headers, optional_params) = if request_credential == "Authorization" {
        (
            BTreeMap::from([(
                request_credential.to_owned(),
                sourced(json!("Bearer caller-secret")),
            )]),
            BTreeMap::new(),
        )
    } else {
        (
            BTreeMap::new(),
            BTreeMap::from([(
                request_credential.to_owned(),
                sourced(json!("caller-secret")),
            )]),
        )
    };
    let error = route
        .execute(
            TokenizeRequest {
                model: "watsonx/model".into(),
                deployment_params: params,
                extra_headers,
                optional_params,
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

#[rstest]
#[case::api_base("api_base")]
#[case::api_key("api_key")]
#[case::api_key_alias("watsonx_api_key")]
#[case::iam_url("WATSONX_IAM_URL")]
#[case::reserved_deployment_params("_tokenize_deployment_params")]
#[tokio::test]
async fn unknown_watsonx_optional_parameters_fail_before_io(#[case] name: &str) {
    let server = MockServer::start().await;
    let route = TokenizeRoute::new(
        Client::plain_for_test(),
        Arc::new(AuthServices::default()),
        no_secrets(),
    );
    let error = route
        .execute(
            TokenizeRequest {
                model: "watsonx/model".into(),
                optional_params: BTreeMap::from([(name.to_owned(), sourced(json!("unexpected")))]),
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
async fn request_owned_credentials_work_with_unselected_environment_defaults() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/ml/v1/text/tokenization"))
        .and(header("authorization", "Bearer request-token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": {"token_count": 1, "tokens": ["hello"]}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let api_base = server.uri();
    let route = TokenizeRoute::new(
        Client::plain_for_test(),
        Arc::new(AuthServices::default()),
        Arc::new(RecordingSecrets::new([
            ("WATSONX_API_BASE", "https://unused.example"),
            ("WATSONX_API_KEY", "unused-host-key"),
            ("WATSONX_TOKEN", "unused-host-token"),
        ])),
    );
    let response = route
        .execute(
            TokenizeRequest {
                model: "watsonx/model".into(),
                prompt: "hello".into(),
                api_base: Some(sourced(api_base)),
                optional_params: BTreeMap::from([
                    ("project_id".into(), sourced(json!("request-project"))),
                    ("token".into(), sourced(json!("request-token"))),
                ]),
                ..Default::default()
            },
            &(),
            None,
        )
        .await
        .unwrap();
    assert_eq!(response.count, 1);
}

#[rstest]
#[tokio::test]
async fn environment_watsonx_token_precedes_request_zen_and_api_keys() {
    let server = MockServer::start().await;
    let api_base = server.uri();
    let route = TokenizeRoute::new(
        Client::plain_for_test(),
        Arc::new(AuthServices::default()),
        Arc::new(RecordingSecrets::new([
            ("WATSONX_API_BASE", api_base.as_str()),
            ("WATSONX_TOKEN", "host-token"),
        ])),
    );
    let error = route
        .execute(
            TokenizeRequest {
                model: "watsonx/model".into(),
                prompt: "hello".into(),
                api_key: Some(sourced(SecretValue::new("request-api-key"))),
                deployment_params: BTreeMap::from([(
                    "project_id".into(),
                    sourced_at(json!("host-project"), InputSource::Deployment),
                )]),
                optional_params: BTreeMap::from([(
                    "zen_api_key".into(),
                    sourced(json!("request-zen-key")),
                )]),
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

#[rstest]
#[tokio::test]
async fn deployment_owned_sdk_inputs_are_accepted() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(header("authorization", "Bearer sdk-token"))
        .and(body_json(json!({
            "model_id": "model",
            "input": "hello",
            "project_id": "sdk-project",
            "parameters": {"return_tokens": true}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": {"token_count": 1, "tokens": ["hello"]}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let route = TokenizeRoute::new(
        Client::plain_for_test(),
        Arc::new(AuthServices::default()),
        no_secrets(),
    );
    let response = route
        .execute(
            TokenizeRequest {
                model: "watsonx/model".into(),
                prompt: "hello".into(),
                api_base: Some(sourced_at(server.uri(), InputSource::Deployment)),
                extra_headers: BTreeMap::from([(
                    "Authorization".into(),
                    sourced_at(json!("Bearer sdk-token"), InputSource::Deployment),
                )]),
                optional_params: BTreeMap::from([(
                    "project_id".into(),
                    sourced_at(json!("sdk-project"), InputSource::Deployment),
                )]),
                ..Default::default()
            },
            &(),
            None,
        )
        .await
        .unwrap();
    assert_eq!(response.count, 1);
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
                model: "vllm/model".into(),
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
                model: "vllm/model".into(),
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
