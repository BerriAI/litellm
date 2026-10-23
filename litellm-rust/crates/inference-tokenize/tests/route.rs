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
use litellm_inference_tokenize::{
    Error, LocalTokenizerRegistry, TokenizeBackend, TokenizeRequest, TokenizeRoute,
};
use litellm_llms_types::formats::tokenize::TokenizeInclude;
use litellm_token_counter::raw_text::{LocalTokenizerConfig, RawTextArtifact, RawTextEngine};
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

fn local_config(engine: RawTextEngine, artifact: RawTextArtifact) -> Arc<LocalTokenizerConfig> {
    Arc::new(LocalTokenizerConfig {
        id: "local-test".into(),
        revision: "revision".into(),
        engine,
        artifact,
        special_text: None,
        add_special_tokens: None,
    })
}

fn local_hugging_face_postprocessor_artifact() -> String {
    let mut artifact: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../litellm/litellm_core_utils/tokenizers/anthropic_tokenizer.json"
    )))
    .unwrap();
    artifact["post_processor"] = json!({
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "<EOT>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}}
        ],
        "pair": [],
        "special_tokens": {
            "<EOT>": {"id": "<EOT>", "ids": [0], "tokens": ["<EOT>"]}
        }
    });
    serde_json::to_string(&artifact).unwrap()
}

fn local_call(config: Arc<LocalTokenizerConfig>, source: InputSource) -> TokenizeRequest {
    TokenizeRequest {
        model: "local-test".into(),
        prompt: "hello".into(),
        backend: TokenizeBackend::Local {
            config: sourced_at(config, source),
        },
        ..Default::default()
    }
}

#[rstest]
#[case::vllm(false)]
#[case::watsonx(true)]
#[tokio::test]
async fn providers_share_a_typed_response(route: TokenizeRoute, #[case] watsonx: bool) {
    let (model, endpoint, params, request, response, token_ids, include_ids, include_tokens) =
        if watsonx {
            (
                "watsonx/org/model",
                "/ml/v1/text/tokenization",
                json!({"project_id":"project","token":"test-key"}),
                json!({"model_id":"org/model","input":"hello","project_id":"project","parameters":{"return_tokens":true}}),
                json!({"model_id":"org/model","result":{"token_count":1,"tokens":["hello"]}}),
                Value::Null,
                false,
                true,
            )
        } else {
            (
                "hosted_vllm/org/model",
                "/tokenize",
                json!({}),
                json!({"model":"org/model","prompt":"hello","add_special_tokens":true,"return_token_strs":true}),
                json!({"count":1,"max_model_len":4096,"tokens":[42],"token_strs":["hello"]}),
                json!([42]),
                true,
                true,
            )
        };
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
                include: TokenizeInclude::new(include_ids, include_tokens),
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
#[case::count(false, false)]
#[case::ids(true, false)]
#[case::tokens(false, true)]
#[case::both(true, true)]
#[tokio::test]
async fn local_output_modes_return_only_requested_representations(
    route: TokenizeRoute,
    #[case] ids: bool,
    #[case] tokens: bool,
) {
    let route = route.with_local_registry(Arc::new(LocalTokenizerRegistry::with_capacity(4)));
    let mut call = local_call(
        local_config(
            RawTextEngine::HuggingFace,
            RawTextArtifact::Inline {
                content: local_hugging_face_postprocessor_artifact(),
            },
        ),
        InputSource::Deployment,
    );
    call.include = TokenizeInclude::new(ids, tokens);
    let response = route.execute(call, &(), None).await.unwrap();

    assert!(response.count > 0);
    assert_eq!(
        response.token_ids.as_ref().map(Vec::len),
        ids.then_some(response.count as usize)
    );
    assert_eq!(response.token_ids.is_some(), ids);
    assert_eq!(response.tokens.is_some(), tokens);
}

#[rstest]
#[tokio::test]
async fn request_sourced_local_file_config_is_rejected_before_file_access(route: TokenizeRoute) {
    let request = local_call(
        local_config(
            RawTextEngine::HuggingFace,
            RawTextArtifact::File {
                path: "/missing/request-tokenizer.json".into(),
                sha256: "0".repeat(64),
            },
        ),
        InputSource::Request,
    );
    let error = route.execute(request, &(), None).await.unwrap_err();

    assert!(error.to_string().contains("deployment or environment"));
    assert!(
        !error
            .to_string()
            .contains("failed to read tokenizer artifact")
    );
}

#[rstest]
#[tokio::test]
async fn unsupported_local_pieces_are_rejected_before_file_access(route: TokenizeRoute) {
    let mut request = local_call(
        local_config(
            RawTextEngine::Tiktoken {
                encoding: "cl100k_base".into(),
            },
            RawTextArtifact::File {
                path: "/missing/request-ranks.tiktoken".into(),
                sha256: "0".repeat(64),
            },
        ),
        InputSource::Deployment,
    );
    request.include = TokenizeInclude::new(false, true);
    let error = route.execute(request, &(), None).await.unwrap_err();

    assert!(matches!(
        error,
        Error::Unsupported("requested tokenize output")
    ));
}

#[rstest]
#[tokio::test]
async fn local_calls_reject_provider_credentials_and_parameters(route: TokenizeRoute) {
    let mut request = local_call(
        local_config(
            RawTextEngine::HuggingFace,
            RawTextArtifact::Inline {
                content: include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../../litellm/litellm_core_utils/tokenizers/anthropic_tokenizer.json"
                ))
                .into(),
            },
        ),
        InputSource::Deployment,
    );
    request.api_key = Some(sourced(SecretValue::new("not-used")));
    request.api_base = Some(sourced("https://example.invalid".into()));
    request.optional_params = sourced_map(&json!({"add_special_tokens":false}));

    let error = route.execute(request, &(), None).await.unwrap_err();

    assert!(
        error
            .to_string()
            .contains("provider credentials and parameters")
    );
}

#[rstest]
#[tokio::test]
async fn local_success_reports_identity_and_zero_cost_without_provider_hooks(route: TokenizeRoute) {
    let route = route.with_local_registry(Arc::new(LocalTokenizerRegistry::with_capacity(2)));
    let hooks = Hooks::default();
    let mut request = local_call(
        local_config(
            RawTextEngine::HuggingFace,
            RawTextArtifact::Inline {
                content: include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../../litellm/litellm_core_utils/tokenizers/anthropic_tokenizer.json"
                ))
                .into(),
            },
        ),
        InputSource::Environment,
    );
    request.model = "public-alias".into();
    let response = route.execute(request.clone(), &hooks, None).await.unwrap();

    assert_eq!(response.model, "local-test");
    assert!(hooks.context.lock().unwrap().is_none());
    assert!(hooks.raw.lock().unwrap().is_none());
    let facts = hooks.facts.lock().unwrap().clone().unwrap();
    assert_eq!(
        facts.source,
        litellm_host::interceptors::ResultSource::Provider
    );
    assert_eq!(facts.provider.model, "local-test");
    assert_eq!(facts.provider.provider, "local");
    assert_eq!(
        facts.cost,
        litellm_host::interceptors::Cost::Reported {
            amount: serde_json::Number::from(0)
        }
    );

    let warm_hooks = Hooks::default();
    let warm_response = route.execute(request, &warm_hooks, None).await.unwrap();
    assert_eq!(warm_response.model, "local-test");
    let warm_facts = warm_hooks.facts.lock().unwrap().clone().unwrap();
    assert_eq!(
        warm_facts.source,
        litellm_host::interceptors::ResultSource::Provider
    );
    assert_eq!(
        warm_facts.cost,
        litellm_host::interceptors::Cost::Reported {
            amount: serde_json::Number::from(0)
        }
    );
    assert!(warm_hooks.context.lock().unwrap().is_none());
    assert!(warm_hooks.raw.lock().unwrap().is_none());
}

#[rstest]
#[tokio::test]
async fn unsupported_provider_representation_fails_before_secret_resolution() {
    let secrets = Arc::new(RecordingSecrets::new([(
        "WATSONX_APIKEY",
        "must-not-be-requested",
    )]));
    let route = TokenizeRoute::new(
        Client::plain_for_test(),
        Arc::new(AuthServices::default()),
        secrets.clone(),
    );
    let error = route
        .execute(
            TokenizeRequest {
                model: "watsonx/org/model".into(),
                prompt: "hello".into(),
                include: TokenizeInclude::new(true, false),
                api_base: Some(sourced("https://example.invalid".to_string())),
                ..Default::default()
            },
            &(),
            None,
        )
        .await
        .unwrap_err();

    assert!(matches!(
        error,
        Error::Unsupported("requested tokenize output")
    ));
    assert!(secrets.requested().is_empty());
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
#[case::enabled(true)]
#[case::disabled(false)]
#[tokio::test]
async fn watsonx_special_token_policy_is_rejected_before_secret_resolution(
    #[case] add_special_tokens: bool,
) {
    let secrets = Arc::new(RecordingSecrets::new([(
        "WATSONX_APIKEY",
        "must-not-be-requested",
    )]));
    let route = TokenizeRoute::new(
        Client::plain_for_test(),
        Arc::new(AuthServices::default()),
        secrets.clone(),
    );
    let error = route
        .execute(
            TokenizeRequest {
                model: "watsonx/org/model".into(),
                add_special_tokens: Some(add_special_tokens),
                ..Default::default()
            },
            &(),
            None,
        )
        .await
        .unwrap_err();

    assert!(matches!(error, Error::Unsupported("add_special_tokens")));
    assert!(secrets.requested().is_empty());
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
async fn local_hugging_face_special_token_override_is_policy_scoped(route: TokenizeRoute) {
    let route = route.with_local_registry(Arc::new(LocalTokenizerRegistry::with_capacity(4)));
    let configured = {
        let mut config = (*local_config(
            RawTextEngine::HuggingFace,
            RawTextArtifact::Inline {
                content: local_hugging_face_postprocessor_artifact(),
            },
        ))
        .clone();
        config.add_special_tokens = Some(false);
        Arc::new(config)
    };

    let omitted = route
        .execute(
            local_call(Arc::clone(&configured), InputSource::Deployment),
            &(),
            None,
        )
        .await
        .unwrap()
        .count;
    let mut explicit_true = local_call(Arc::clone(&configured), InputSource::Deployment);
    explicit_true.add_special_tokens = Some(true);
    let enabled = route.execute(explicit_true, &(), None).await.unwrap().count;
    let mut explicit_false = local_call(configured, InputSource::Deployment);
    explicit_false.add_special_tokens = Some(false);
    let disabled = route
        .execute(explicit_false, &(), None)
        .await
        .unwrap()
        .count;

    assert!(enabled > omitted);
    assert_eq!(disabled, omitted);
}

#[rstest]
#[tokio::test]
async fn local_tiktoken_special_token_insertion_is_rejected(route: TokenizeRoute) {
    let mut request = local_call(
        local_config(
            RawTextEngine::Tiktoken {
                encoding: "cl100k_base".into(),
            },
            RawTextArtifact::Inline {
                content: String::new(),
            },
        ),
        InputSource::Deployment,
    );
    request.add_special_tokens = Some(true);

    let error = route.execute(request, &(), None).await.unwrap_err();

    assert!(error.to_string().contains("add_special_tokens"));
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
