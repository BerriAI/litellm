use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use litellm_config::Config;
use rstest::rstest;
use serde_json::{Value, json};
use tower::ServiceExt;

const HF_ARTIFACT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../litellm/litellm_core_utils/tokenizers/anthropic_tokenizer.json"
));

fn config() -> Config {
    Config::from_yaml(
        &json!({
            "model_list": [{
                "model_name": "public/model",
                "litellm_params": {
                    "model": "local-configured",
                    "backend": {
                        "type": "local",
                        "config": {
                            "id": "mounted-local",
                            "revision": "test-revision",
                            "engine": {"type": "hugging_face"},
                            "artifact": {"type": "inline", "content": HF_ARTIFACT}
                        }
                    }
                }
            }],
            "general_settings": {"master_key": "gateway-key"}
        })
        .to_string(),
    )
    .unwrap()
}

#[rstest]
#[case::unversioned("/tokenize")]
#[case::versioned("/v1/tokenize")]
#[tokio::test]
async fn mounted_gateway_authenticates_alias_and_loads_configured_local_backend(
    #[case] endpoint: &str,
) {
    let config = config();
    let inference = litellm_gateway::build_inference(&config).unwrap();
    let app = litellm_gateway::router(inference, &config, None, None);
    let unauthenticated = app
        .clone()
        .oneshot(
            Request::post(endpoint)
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model":"public/model","prompt":"hello"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unauthenticated.status(), 401);
    let response = app
        .oneshot(
            Request::post(endpoint)
                .header("authorization", "Bearer gateway-key")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model":"public/model","prompt":"hello"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap())
            .unwrap();
    assert_eq!(body["model"], "mounted-local");
    assert!(body["count"].as_u64().unwrap() > 0);
    assert!(body.get("token_ids").is_some());
    assert!(body.get("tokens").is_some());
}
