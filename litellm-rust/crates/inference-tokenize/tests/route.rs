use std::sync::{Arc, Mutex};

use litellm_auth::{AuthServices, SecretValue};
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
                api_base: Some(server.uri()),
                api_key: Some(SecretValue::new("test-key")),
                optional_params: params.as_object().unwrap().clone(),
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
            api_base: Some(server.uri()),
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
                api_base: Some(server.uri()),
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
                api_base: Some(server.uri()),
                optional_params: options.as_object().unwrap().clone(),
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
