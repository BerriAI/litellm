use std::sync::Arc;

use litellm_auth::AuthServices;
use litellm_http::Client;
use litellm_inference_testing::RecordingSecrets;
use litellm_inference_tokenize::{TokenizeRequest, TokenizeRoute};
use rstest::{fixture, rstest};
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, body_string_contains, header, method, path},
};

#[fixture]
async fn server() -> MockServer {
    MockServer::start().await
}

#[rstest]
#[tokio::test]
async fn watsonx_iam_credentials_are_resolved_once_across_routes(
    #[future(awt)] server: MockServer,
) {
    Mock::given(method("POST"))
        .and(path("/identity/token"))
        .and(header("content-type", "application/x-www-form-urlencoded"))
        .and(body_string_contains("apikey=configured-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": "iam-access-token",
            "expires_in": 3600
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/ml/v1/text/tokenization"))
        .and(header("authorization", "Bearer iam-access-token"))
        .and(body_json(json!({
            "model_id": "ibm/model",
            "input": "hello world",
            "project_id": "configured-project",
            "parameters": {"return_tokens": true}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": {"token_count": 2, "tokens": ["hello", " world"]}
        })))
        .expect(2)
        .mount(&server)
        .await;
    let iam_endpoint = format!("{}/identity/token", server.uri());
    let api_base = server.uri();
    let secrets = Arc::new(RecordingSecrets::new([
        ("WATSONX_IAM_URL", iam_endpoint.as_str()),
        ("WATSONX_API_BASE", api_base.as_str()),
        ("WATSONX_API_KEY", "configured-api-key"),
        ("WATSONX_PROJECT_ID", "configured-project"),
    ]));
    let auth = Arc::new(AuthServices::default());
    let first = TokenizeRoute::new(
        Client::no_redirect_for_test(),
        auth.clone(),
        secrets.clone(),
    );
    let second = TokenizeRoute::new(Client::no_redirect_for_test(), auth, secrets);
    let request = TokenizeRequest {
        model: "watsonx/ibm/model".into(),
        prompt: "hello world".into(),
        ..Default::default()
    };
    let first_response = first.execute(request.clone(), &(), None).await.unwrap();
    let second_response = second.execute(request, &(), None).await.unwrap();
    assert_eq!(first_response, second_response);
    assert_eq!(first_response.model, "ibm/model");
    assert_eq!(first_response.count, 2);
    assert_eq!(
        first_response.tokens,
        Some(vec!["hello".into(), " world".into()])
    );
    assert_eq!(first_response.token_ids, None);
}
