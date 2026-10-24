mod support;

use std::sync::{Arc, Mutex};

use litellm_gateway_inference::Deployment;
use litellm_host::{
    interceptors::{
        Cost, ExecutionFacts, Interceptors, RawResponse, RequestContext, ResultSource, WireRequest,
    },
    lifecycle::{CallEvent, ExecutionEvent},
    observation::observation_channel,
};
use rstest::rstest;
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header, method, path},
};

fn deployment(model: &str, api_base: Option<String>) -> Deployment {
    let mut deployment = Deployment {
        model: model.into(),
        api_base,
        api_key: Some("deployment-key".into()),
        ..Default::default()
    };
    deployment.litellm_params = serde_json::from_value(json!({
        "model": model
    }))
    .unwrap();
    deployment
}

async fn response_json(response: axum::response::Response) -> Value {
    serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap(),
    )
    .unwrap()
}

#[rstest]
#[case::count_omitted(json!({}), json!({"model":"org/model","prompt":"  café 世界  ","add_special_tokens":true,"return_token_strs":true}), json!({"token_ids":null,"tokens":null}))]
#[case::count_empty(json!({"include":[]}), json!({"model":"org/model","prompt":"  café 世界  ","add_special_tokens":true,"return_token_strs":true}), json!({"token_ids":null,"tokens":null}))]
#[case::ids(json!({"include":["token_ids"]}), json!({"model":"org/model","prompt":"  café 世界  ","add_special_tokens":true,"return_token_strs":true}), json!({"token_ids":[11,12],"tokens":null}))]
#[case::pieces(json!({"include":["tokens"]}), json!({"model":"org/model","prompt":"  café 世界  ","add_special_tokens":true,"return_token_strs":true}), json!({"token_ids":null,"tokens":["  café"," 世界  "]}))]
#[case::both(json!({"include":["token_ids","tokens"]}), json!({"model":"org/model","prompt":"  café 世界  ","add_special_tokens":true,"return_token_strs":true}), json!({"token_ids":[11,12],"tokens":["  café"," 世界  "]}))]
#[tokio::test]
async fn aliases_project_strict_payload_and_keep_all_response_keys(
    #[case] options: Value,
    #[case] upstream_request: Value,
    #[case] expected: Value,
    #[values("/tokenize", "/v1/tokenize")] endpoint: &str,
) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/tokenize"))
        .and(header("authorization", "Bearer deployment-key"))
        .and(body_json(upstream_request))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "count": 2, "tokens": [11, 12], "token_strs": ["  café", " 世界  "]
        })))
        .expect(1)
        .mount(&upstream)
        .await;
    let body = json!({"model":"public/model","prompt":"  café 世界  "})
        .as_object()
        .unwrap()
        .iter()
        .chain(options.as_object().unwrap().iter())
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<serde_json::Map<_, _>>();
    let response = support::post(
        support::app_with_deployment(deployment(
            "hosted_vllm/org/model",
            Some(format!("{}/v1", upstream.uri())),
        )),
        endpoint,
        Value::Object(body),
    )
    .await;
    assert_eq!(response.status(), 200);
    let result = response_json(response).await;
    assert_eq!(result["model"], "org/model");
    assert_eq!(result["count"], 2);
    assert_eq!(result["token_ids"], expected["token_ids"]);
    assert_eq!(result["tokens"], expected["tokens"]);
    assert_eq!(result.as_object().unwrap().len(), 4);
}

#[rstest]
#[case::unknown("messages", json!([]))]
#[case::private("api_base", json!("https://caller.invalid"))]
#[case::resource("project", json!("caller-project"))]
#[case::generation("temperature", json!(0.2))]
#[case::credentials("api_key", json!("caller-secret"))]
#[case::metadata("metadata", json!({"user":"caller"}))]
#[case::images("images", json!([]))]
#[case::tools("tools", json!([]))]
#[case::template("chat_template", json!("template"))]
#[case::output("output", json!("count"))]
#[case::available("available", json!(true))]
#[case::custom_backend("backend", json!({"type":"local"}))]
#[case::trusted_sources("trusted_sources", json!([]))]
#[case::space("space", json!("caller-space"))]
#[case::api_version("api_version", json!("caller-version"))]
#[case::credentials_map("credentials", json!({}))]
#[tokio::test]
async fn unknown_semantic_private_and_resource_fields_are_rejected(
    #[case] field: &str,
    #[case] value: Value,
) {
    let upstream = MockServer::start().await;
    let mut body = json!({"model":"public/model","prompt":"hello"});
    body[field] = value;
    let response = support::post(
        support::app("hosted_vllm/model", &upstream.uri()),
        "/tokenize",
        body,
    )
    .await;
    assert_eq!(response.status(), 400);
    assert!(response_json(response).await["error"]["message"].is_string());
    assert!(upstream.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[case::null_include(json!({"include":null}))]
#[case::scalar_include(json!({"include":"tokens"}))]
#[case::duplicate_include(json!({"include":["tokens","tokens"]}))]
#[case::unknown_include(json!({"include":["pieces"]}))]
#[case::null_special(json!({"add_special_tokens":null}))]
#[case::string_special(json!({"add_special_tokens":"false"}))]
#[case::number_special(json!({"add_special_tokens":0}))]
#[tokio::test]
async fn malformed_include_and_special_token_values_fail_before_io(#[case] extra: Value) {
    let upstream = MockServer::start().await;
    let mut body = json!({"model":"public/model","prompt":"hello"});
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    let response = support::post(
        support::app("hosted_vllm/model", &upstream.uri()),
        "/v1/tokenize",
        body,
    )
    .await;
    assert_eq!(response.status(), 400);
    assert!(upstream.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[case::tokens(json!(["tokens"]))]
#[case::ids(json!(["token_ids"]))]
#[case::both(json!(["token_ids", "tokens"]))]
#[tokio::test]
async fn requested_representations_are_enforced(#[case] include: Value) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "count": 1, "tokens": [7], "token_strs": ["hello"]
        })))
        .expect(1)
        .mount(&upstream)
        .await;
    let response = support::post(
        support::app("hosted_vllm/model", &upstream.uri()),
        "/tokenize",
        json!({"model":"public/model","prompt":"hello","include":include}),
    )
    .await;
    assert_eq!(response.status(), 200);
}

#[rstest]
#[tokio::test]
async fn configured_provider_headers_and_special_token_override_reach_vllm_without_defaults() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/tokenize"))
        .and(header("authorization", "Bearer deployment-key"))
        .and(header("x-deployment", "configured"))
        .and(body_json(json!({
            "model":"org/model",
            "prompt":"",
            "add_special_tokens":false,
            "return_token_strs":true
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "count":0,"tokens":[],"token_strs":[]
        })))
        .expect(1)
        .mount(&upstream)
        .await;
    let mut deployment = deployment(
        "hosted_vllm/org/model",
        Some(format!("{}/v1", upstream.uri())),
    );
    deployment.litellm_params = serde_json::from_value(json!({
        "model":"hosted_vllm/org/model",
        "backend":{"type":"provider"},
        "extra_headers":{"x-deployment":"configured"},
        "temperature":0.7,
        "max_tokens":100
    }))
    .unwrap();
    let response = support::post(
        support::app_with_deployment(deployment),
        "/tokenize",
        json!({
            "model":"public/model",
            "prompt":"",
            "add_special_tokens":false,
            "include":["token_ids","tokens"]
        }),
    )
    .await;
    assert_eq!(response.status(), 200);
    let body = response_json(response).await;
    assert_eq!(body["count"], 0);
    assert_eq!(body["token_ids"], json!([]));
    assert_eq!(body["tokens"], json!([]));
}

#[rstest]
#[tokio::test]
async fn watsonx_unsupported_representation_is_rejected_before_io() {
    let upstream = MockServer::start().await;
    let response = support::post(
        support::app("watsonx/model", &upstream.uri()),
        "/tokenize",
        json!({
            "model":"public/model",
            "prompt":"hello",
            "include":["token_ids"]
        }),
    )
    .await;
    assert_eq!(response.status(), 501);
    assert!(upstream.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[case::enabled(true)]
#[case::disabled(false)]
#[tokio::test]
async fn watsonx_rejects_explicit_special_token_policy_before_io(#[case] add_special_tokens: bool) {
    let upstream = MockServer::start().await;
    let response = support::post(
        support::app("watsonx/model", &upstream.uri()),
        "/tokenize",
        json!({
            "model":"public/model",
            "prompt":"hello",
            "add_special_tokens":add_special_tokens
        }),
    )
    .await;
    assert_eq!(response.status(), 501);
    assert!(upstream.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[case::pieces(json!({"count":1,"tokens":[7],"token_strs":null}), json!(["tokens"]))]
#[case::ids(json!({"count":1,"tokens":null,"token_strs":["hello"]}), json!(["token_ids"]))]
#[tokio::test]
async fn provider_missing_requested_representation_is_not_fabricated(
    #[case] response_body: Value,
    #[case] include: Value,
) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_body))
        .expect(1)
        .mount(&upstream)
        .await;
    let response = support::post(
        support::app("hosted_vllm/model", &upstream.uri()),
        "/tokenize",
        json!({"model":"public/model","prompt":"hello","include":include}),
    )
    .await;
    assert_eq!(response.status(), 500);
}

#[rstest]
#[case::rate_limit(429)]
#[case::unauthorized(401)]
#[case::unavailable(503)]
#[tokio::test]
async fn upstream_errors_keep_their_status(#[case] status: u16) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(status))
        .expect(1)
        .mount(&upstream)
        .await;
    let response = support::post(
        support::app("hosted_vllm/model", &upstream.uri()),
        "/tokenize",
        json!({"model":"public/model","prompt":"hello"}),
    )
    .await;
    assert_eq!(response.status().as_u16(), status);
}

#[rstest]
#[tokio::test]
async fn authorization_and_unknown_aliases_precede_provider_io() {
    let upstream = MockServer::start().await;
    let denied = support::post(
        support::app_with_permissions(
            "hosted_vllm/model",
            &upstream.uri(),
            litellm_gateway_auth::Permissions::None,
        ),
        "/tokenize",
        json!({"model":"public/model","prompt":"hello"}),
    )
    .await;
    assert_eq!(denied.status(), 403);
    let unknown = support::post(
        support::app("hosted_vllm/model", &upstream.uri()),
        "/tokenize",
        json!({"model":"missing","prompt":"hello"}),
    )
    .await;
    assert_eq!(unknown.status(), 400);
    assert!(upstream.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[case::alias_grant("public/model", 200, 1)]
#[case::provider_model_grant("hosted_vllm/org/model", 403, 0)]
#[tokio::test]
async fn restricted_model_grants_match_public_alias_before_provider_io(
    #[case] granted_name: &str,
    #[case] expected_status: u16,
    #[case] expected_requests: usize,
    #[values("/tokenize", "/v1/tokenize")] endpoint: &str,
) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/tokenize"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "count":1,"tokens":[7],"token_strs":["hello"]
        })))
        .mount(&upstream)
        .await;
    let permissions = litellm_gateway_auth::Permissions::Only(Arc::from([
        litellm_gateway_auth::AccessRequest::Model {
            name: granted_name.into(),
            deployment: "hosted_vllm/org/model".into(),
        },
    ]));
    let response = support::post(
        support::app_with_permissions("hosted_vllm/org/model", &upstream.uri(), permissions),
        endpoint,
        json!({"model":"public/model","prompt":"hello"}),
    )
    .await;

    assert_eq!(response.status().as_u16(), expected_status);
    assert_eq!(
        upstream.received_requests().await.unwrap().len(),
        expected_requests
    );
}

#[derive(Default, Clone)]
struct Hooks {
    rewrite: bool,
    fail_result: bool,
    calls: Arc<Mutex<Vec<String>>>,
}

impl Interceptors<litellm_inference::Error> for Hooks {
    async fn before_provider_request(
        &self,
        mut wire: WireRequest,
        _: RequestContext,
    ) -> Result<WireRequest, litellm_inference::Error> {
        self.calls.lock().unwrap().push("before".into());
        if self.rewrite {
            wire.body["prompt"] = json!("rewritten");
        }
        Ok(wire)
    }

    async fn after_provider_response(
        &self,
        _: RawResponse,
    ) -> Result<(), litellm_inference::Error> {
        self.calls.lock().unwrap().push("after".into());
        Ok(())
    }

    async fn result_ready(&self, _: ExecutionFacts) -> Result<(), litellm_inference::Error> {
        self.calls.lock().unwrap().push("result".into());
        if self.fail_result {
            return Err(litellm_inference::Error::InvalidRequest(
                "result hook failed".into(),
            ));
        }
        Ok(())
    }
}

#[rstest]
#[tokio::test]
async fn hooks_rewrite_wire_and_result_failure_does_not_replay() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_json(json!({
            "model":"org/model","prompt":"rewritten",
            "add_special_tokens":true,"return_token_strs":true
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "count":1,"tokens":[7],"token_strs":["rewritten"]
        })))
        .expect(1)
        .mount(&upstream)
        .await;
    let hooks = Hooks {
        rewrite: true,
        ..Default::default()
    };
    let response = support::app_with_tokenize_hooks(
        deployment("hosted_vllm/org/model", Some(upstream.uri())),
        litellm_gateway_auth::Permissions::All,
        hooks.clone(),
        None,
    );
    let ok = support::post(
        response,
        "/tokenize",
        json!({"model":"public/model","prompt":"original"}),
    )
    .await;
    assert_eq!(ok.status(), 200);
    assert_eq!(
        &*hooks.calls.lock().unwrap(),
        &["before", "after", "result"]
    );

    let failing_upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "count":1,"tokens":[7],"token_strs":["original"]
        })))
        .expect(1)
        .mount(&failing_upstream)
        .await;
    let failing = Hooks {
        fail_result: true,
        ..Default::default()
    };
    let response = support::post(
        support::app_with_tokenize_hooks(
            deployment("hosted_vllm/org/model", Some(failing_upstream.uri())),
            litellm_gateway_auth::Permissions::All,
            failing.clone(),
            None,
        ),
        "/tokenize",
        json!({"model":"public/model","prompt":"original"}),
    )
    .await;
    assert_eq!(response.status(), 400);
    assert_eq!(
        &*failing.calls.lock().unwrap(),
        &["before", "after", "result"]
    );
}

#[rstest]
#[tokio::test]
async fn provider_facts_defer_cost_and_observers_see_terminals() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "count":1,"tokens":[7],"token_strs":["hello"]
        })))
        .expect(1)
        .mount(&upstream)
        .await;
    let (sender, mut events) = observation_channel(std::num::NonZeroUsize::new(16).unwrap());
    let hooks = Hooks::default();
    let response = support::post(
        support::app_with_tokenize_hooks(
            deployment("hosted_vllm/org/model", Some(upstream.uri())),
            litellm_gateway_auth::Permissions::All,
            hooks.clone(),
            Some(sender),
        ),
        "/tokenize",
        json!({"model":"public/model","prompt":"hello"}),
    )
    .await;
    assert_eq!(response.status(), 200);
    assert_eq!(
        &*hooks.calls.lock().unwrap(),
        &["before", "after", "result"]
    );
    assert!(matches!(
        events.recv().await,
        Some(CallEvent::Started { .. })
    ));
    let facts = loop {
        match events.recv().await {
            Some(CallEvent::Execution(ExecutionEvent::ResultReady { facts })) => break facts,
            Some(_) => {}
            None => panic!("observer closed before facts"),
        }
    };
    assert_eq!(facts.provider.provider, "hosted_vllm");
    assert_eq!(facts.source, ResultSource::Provider);
    assert_eq!(facts.cost, Cost::Deferred);
    assert!(matches!(
        events.recv().await,
        Some(CallEvent::Succeeded { .. })
    ));
}
