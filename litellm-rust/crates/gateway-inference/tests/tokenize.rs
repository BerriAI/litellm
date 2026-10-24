mod support;

use std::sync::{Arc, Mutex};

use axum::{body::Body, http::Request};
use litellm_gateway_inference::Deployment;
use litellm_host::{
    interceptors::{
        Cost, ExecutionFacts, Interceptors, RawResponse, RequestContext, ResultSource, WireRequest,
    },
    lifecycle::{CallEvent, ExecutionEvent},
    observation::observation_channel,
};
use litellm_token_counter::raw_text::{
    LocalTokenizerConfig, RawTextArtifact, RawTextEngine, SpecialTextPolicy,
};
use rstest::rstest;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tower::ServiceExt;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header, method, path},
};

const HF_ARTIFACT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../litellm/litellm_core_utils/tokenizers/anthropic_tokenizer.json"
));

fn local_config(engine: RawTextEngine, artifact: RawTextArtifact) -> LocalTokenizerConfig {
    LocalTokenizerConfig {
        id: "gateway-local".into(),
        revision: "test-revision".into(),
        engine,
        artifact,
        special_text: None,
        add_special_tokens: None,
    }
}

fn small_hf_artifact() -> String {
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [{
            "id": 0,
            "special": true,
            "content": "[CLS]",
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false
        }],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}}
            ],
            "pair": [],
            "special_tokens": {
                "[CLS]": {"id": "[CLS]", "ids": [0], "tokens": ["[CLS]"]}
            }
        },
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {"[CLS]": 0, "hello": 1, "world": 2, "[UNK]": 3},
            "unk_token": "[UNK]"
        }
    })
    .to_string()
}

fn deployment(model: &str, api_base: Option<String>) -> Deployment {
    let mut deployment = Deployment {
        model: model.into(),
        api_base,
        api_key: Some("deployment-key".into()),
        ..Default::default()
    };
    deployment.litellm_params = serde_json::from_value(json!({
        "model": model,
        "backend": {"type": "provider"}
    }))
    .unwrap();
    deployment
}

fn local_deployment(config: LocalTokenizerConfig) -> Deployment {
    let mut deployment = Deployment {
        model: "local-configured".into(),
        ..Default::default()
    };
    deployment.litellm_params = serde_json::from_value(json!({
        "model": "local-configured",
        "backend": {"type": "local", "config": config_value(&config)}
    }))
    .unwrap();
    deployment
}

fn config_value(config: &LocalTokenizerConfig) -> Value {
    let engine = match &config.engine {
        RawTextEngine::HuggingFace => json!({"type":"hugging_face"}),
        RawTextEngine::Tiktoken { encoding } => json!({"type":"tiktoken","encoding":encoding}),
    };
    let artifact = match &config.artifact {
        RawTextArtifact::Inline { content } => json!({"type":"inline","content":content}),
        RawTextArtifact::File { path, sha256 } => {
            json!({"type":"file","path":path,"sha256":sha256})
        }
    };
    let special_text = match config.special_text {
        Some(SpecialTextPolicy::Literal) => json!("literal"),
        Some(SpecialTextPolicy::Recognize) => json!("recognize"),
        None => Value::Null,
    };
    json!({
        "id": config.id,
        "revision": config.revision,
        "engine": engine,
        "artifact": artifact,
        "special_text": special_text,
        "add_special_tokens": config.add_special_tokens
    })
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
async fn upstream_errors_forward_safe_headers_once(#[case] status: u16) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(status)
                .insert_header("retry-after", "7")
                .insert_header("x-request-id", "request-1")
                .insert_header("x-correlation-id", "correlation-1")
                .insert_header("authorization", "Bearer secret")
                .insert_header("set-cookie", "secret=private"),
        )
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
    assert_eq!(response.headers()["retry-after"], "7");
    assert_eq!(response.headers()["x-request-id"], "request-1");
    assert_eq!(response.headers()["x-correlation-id"], "correlation-1");
    assert!(!response.headers().contains_key("authorization"));
    assert!(!response.headers().contains_key("set-cookie"));
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

#[rstest]
#[case::count_omitted(json!({}), "hello world", 3, Value::Null, Value::Null)]
#[case::count_empty_include(json!({"include": []}), "hello world", 3, Value::Null, Value::Null)]
#[case::ids(json!({"include": ["token_ids"]}), "hello world", 3, json!([0, 1, 2]), Value::Null)]
#[case::tokens(json!({"include": ["tokens"]}), "hello world", 3, Value::Null, json!(["[CLS]", "hello", "world"]))]
#[case::both(json!({"include": ["token_ids", "tokens"]}), "hello world", 3, json!([0, 1, 2]), json!(["[CLS]", "hello", "world"]))]
#[case::empty_prompt_both(json!({"include": ["token_ids", "tokens"]}), "", 1, json!([0]), json!(["[CLS]"]))]
#[tokio::test]
async fn configured_local_backends_preserve_prompt_and_response_contract(
    #[case] options: Value,
    #[case] prompt: &str,
    #[case] count: u64,
    #[case] token_ids: Value,
    #[case] tokens: Value,
) {
    let response = support::post(
        support::app_with_deployment(local_deployment(local_config(
            RawTextEngine::HuggingFace,
            RawTextArtifact::Inline {
                content: small_hf_artifact(),
            },
        ))),
        "/v1/tokenize",
        {
            let mut body = json!({"model":"public/model","prompt":prompt});
            body.as_object_mut()
                .unwrap()
                .extend(options.as_object().unwrap().clone());
            body
        },
    )
    .await;
    assert_eq!(response.status(), 200);
    let body = response_json(response).await;
    assert_eq!(body["model"], "gateway-local");
    assert_eq!(body["count"], count);
    assert_eq!(body["token_ids"], token_ids);
    assert_eq!(body["tokens"], tokens);
    assert_eq!(body.as_object().unwrap().len(), 4);
}

#[rstest]
#[tokio::test]
async fn explicit_special_token_override_preserves_configured_policy() {
    let config = LocalTokenizerConfig {
        add_special_tokens: Some(true),
        ..local_config(
            RawTextEngine::HuggingFace,
            RawTextArtifact::Inline {
                content: small_hf_artifact(),
            },
        )
    };
    let app = support::app_with_deployment(local_deployment(config));
    let initial = response_json(
        support::post(
            app.clone(),
            "/tokenize",
            json!({
                "model":"public/model",
                "prompt":"hello world",
                "include":["token_ids","tokens"]
            }),
        )
        .await,
    )
    .await;
    let explicit_false = response_json(
        support::post(
            app.clone(),
            "/tokenize",
            json!({
                "model":"public/model",
                "prompt":"hello world",
                "include":["token_ids","tokens"],
                "add_special_tokens":false
            }),
        )
        .await,
    )
    .await;
    let omitted_again = response_json(
        support::post(
            app,
            "/tokenize",
            json!({
                "model":"public/model",
                "prompt":"hello world",
                "include":["token_ids","tokens"]
            }),
        )
        .await,
    )
    .await;

    assert_eq!(initial["model"], "gateway-local");
    assert_eq!(explicit_false["model"], "gateway-local");
    assert_eq!(omitted_again["model"], "gateway-local");
    assert_eq!(initial["count"], 3);
    assert_eq!(initial["token_ids"], json!([0, 1, 2]));
    assert_eq!(initial["tokens"], json!(["[CLS]", "hello", "world"]));
    assert_eq!(explicit_false["count"], 2);
    assert_eq!(explicit_false["token_ids"], json!([1, 2]));
    assert_eq!(explicit_false["tokens"], json!(["hello", "world"]));
    assert_eq!(omitted_again["count"], 3);
    assert_eq!(omitted_again["token_ids"], json!([0, 1, 2]));
    assert_eq!(omitted_again["tokens"], json!(["[CLS]", "hello", "world"]));
}

#[rstest]
#[tokio::test]
async fn configured_tiktoken_supports_ids_but_rejects_pieces() {
    let response = support::post(
        support::app_with_deployment(local_deployment(local_config(
            RawTextEngine::Tiktoken {
                encoding: "cl100k_base".into(),
            },
            RawTextArtifact::Inline {
                content: byte_ranks(),
            },
        ))),
        "/tokenize",
        json!({"model":"public/model","prompt":"hello","include":["token_ids"]}),
    )
    .await;
    assert_eq!(response.status(), 200);
    let unsupported = support::post(
        support::app_with_deployment(local_deployment(local_config(
            RawTextEngine::Tiktoken {
                encoding: "cl100k_base".into(),
            },
            RawTextArtifact::Inline {
                content: byte_ranks(),
            },
        ))),
        "/tokenize",
        json!({"model":"public/model","prompt":"hello","include":["tokens"]}),
    )
    .await;
    assert_eq!(unsupported.status(), 501);
}

fn byte_ranks() -> String {
    use base64::{Engine, engine::general_purpose::STANDARD};
    (0u16..=255)
        .map(|byte| format!("{} {}", STANDARD.encode([byte as u8]), byte))
        .collect::<Vec<_>>()
        .join("\n")
}

#[rstest]
#[tokio::test]
async fn local_tiktoken_file_artifact_is_hash_pinned_and_pieces_are_explicitly_unsupported() {
    let content = byte_ranks();
    let path =
        std::env::temp_dir().join(format!("gateway-tokenize-{}.tiktoken", std::process::id()));
    std::fs::write(&path, &content).unwrap();
    let sha256 = format!("{:x}", Sha256::digest(content.as_bytes()));
    let mut config = local_config(
        RawTextEngine::Tiktoken {
            encoding: "cl100k_base".into(),
        },
        RawTextArtifact::File {
            path: path.clone(),
            sha256,
        },
    );
    let response = support::post(
        support::app_with_deployment(local_deployment(config.clone())),
        "/tokenize",
        json!({"model":"public/model","prompt":"hello","include":["token_ids"]}),
    )
    .await;
    assert_eq!(response.status(), 200);
    config.artifact = RawTextArtifact::File {
        path: path.clone(),
        sha256: "0".repeat(64),
    };
    let invalid = support::post(
        support::app_with_deployment(local_deployment(config)),
        "/tokenize",
        json!({"model":"public/model","prompt":"hello"}),
    )
    .await;
    assert_eq!(invalid.status(), 400);
    std::fs::remove_file(path).unwrap();
}

#[rstest]
#[tokio::test]
async fn local_backend_rejects_provider_fields_without_network_fallback() {
    let response = support::post(
        support::app_with_deployment(Deployment {
            model: "local-invalid".into(),
            api_base: Some("http://127.0.0.1:1".into()),
            ..local_deployment(local_config(
                RawTextEngine::HuggingFace,
                RawTextArtifact::Inline {
                    content: HF_ARTIFACT.into(),
                },
            ))
        }),
        "/tokenize",
        json!({"model":"public/model","prompt":"hello"}),
    )
    .await;
    assert_eq!(response.status(), 400);
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
async fn local_facts_are_provider_zero_cost_and_observers_see_terminals() {
    let (sender, mut events) = observation_channel(std::num::NonZeroUsize::new(16).unwrap());
    let hooks = Hooks::default();
    let response = support::post(
        support::app_with_tokenize_hooks(
            local_deployment(local_config(
                RawTextEngine::HuggingFace,
                RawTextArtifact::Inline {
                    content: HF_ARTIFACT.into(),
                },
            )),
            litellm_gateway_auth::Permissions::All,
            hooks.clone(),
            Some(sender),
        ),
        "/tokenize",
        json!({"model":"public/model","prompt":"hello"}),
    )
    .await;
    assert_eq!(response.status(), 200);
    assert_eq!(&*hooks.calls.lock().unwrap(), &["result"]);
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
    assert_eq!(facts.provider.provider, "local");
    assert_eq!(facts.source, ResultSource::Provider);
    assert_eq!(facts.cost, Cost::Reported { amount: 0.into() });
    assert!(matches!(
        events.recv().await,
        Some(CallEvent::Succeeded { .. })
    ));
}

#[rstest]
#[test]
fn current_thread_executor_remains_responsive_during_cold_local_load_and_encoding() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let app = support::app_with_deployment(local_deployment(local_config(
                RawTextEngine::HuggingFace,
                RawTextArtifact::Inline {
                    content: HF_ARTIFACT.into(),
                },
            )));
            let ticks = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let prompt = "hello ".repeat(100_000);
            let request = tokio::spawn(async move {
                app.oneshot(
                    Request::post("/tokenize")
                        .header("content-type", "application/json")
                        .body(Body::from(
                            json!({"model":"public/model","prompt":prompt}).to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap()
            });
            tokio::task::yield_now().await;
            let observed_ticks = Arc::clone(&ticks);
            let heartbeat = tokio::spawn(async move {
                loop {
                    observed_ticks.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    tokio::task::yield_now().await;
                }
            });
            let response = tokio::time::timeout(std::time::Duration::from_secs(10), request)
                .await
                .unwrap()
                .unwrap();
            heartbeat.abort();
            assert_eq!(response.status(), 200);
            assert!(ticks.load(std::sync::atomic::Ordering::Relaxed) > 0);
        });
}

#[rstest]
#[case::count_only(None, 250_000)]
#[case::both_representations(Some(json!(["token_ids","tokens"])), 80_000)]
#[test]
fn current_thread_executor_remains_responsive_during_warm_local_work(
    #[case] include: Option<Value>,
    #[case] repetitions: usize,
) {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            let app = support::app_with_deployment(local_deployment(local_config(
                RawTextEngine::HuggingFace,
                RawTextArtifact::Inline {
                    content: HF_ARTIFACT.into(),
                },
            )));
            let warm = support::post(
                app.clone(),
                "/tokenize",
                json!({"model":"public/model","prompt":"hello"}),
            )
            .await;
            assert_eq!(warm.status(), 200);

            let maximum_gap = Arc::new(std::sync::atomic::AtomicU64::new(0));
            let observed_gap = Arc::clone(&maximum_gap);
            let (started, ready) = tokio::sync::oneshot::channel();
            let heartbeat = tokio::spawn(async move {
                let mut timer = tokio::time::interval(std::time::Duration::from_millis(2));
                timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                timer.tick().await;
                let mut previous = tokio::time::Instant::now();
                let _ = started.send(());
                loop {
                    timer.tick().await;
                    let gap = previous.elapsed().as_micros() as u64;
                    observed_gap.fetch_max(gap, std::sync::atomic::Ordering::Relaxed);
                    previous = tokio::time::Instant::now();
                }
            });
            ready.await.unwrap();

            let prompt = "hello ".repeat(repetitions);
            let body = match include {
                Some(include) => json!({
                    "model":"public/model",
                    "prompt":prompt,
                    "include":include
                }),
                None => json!({"model":"public/model","prompt":prompt}),
            };
            let request = tokio::spawn(async move {
                app.oneshot(
                    Request::post("/tokenize")
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap()
            });
            let response = tokio::time::timeout(std::time::Duration::from_secs(15), request)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(response.status(), 200);
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            heartbeat.abort();

            let maximum_gap = std::time::Duration::from_micros(
                maximum_gap.load(std::sync::atomic::Ordering::Relaxed),
            );
            assert!(
                maximum_gap < std::time::Duration::from_millis(150),
                "heartbeat paused for {maximum_gap:?}"
            );
        });
}
