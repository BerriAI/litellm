use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use litellm_config::Config;
use rstest::rstest;
use serde_json::{Value, json};
use tower::ServiceExt;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, method, path},
};

fn config(api_base: &str) -> Config {
    Config::from_yaml(
        &json!({
            "model_list": [{
                "model_name": "public/model",
                "litellm_params": {
                    "model": "hosted_vllm/org/model",
                    "api_base": api_base,
                    "api_key": "deployment-key"
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
async fn mounted_gateway_authenticates_alias_before_provider_io(#[case] endpoint: &str) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/tokenize"))
        .and(header("authorization", "Bearer deployment-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "count": 2, "tokens": [11, 12], "token_strs": ["hel", "lo"]
        })))
        .expect(1)
        .mount(&upstream)
        .await;
    let config = config(&upstream.uri());
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
    assert_eq!(body["count"], 2);
}
