use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use litellm_auth::AuthServices;
use litellm_host::interceptors::{Interceptors, RawResponse, RequestContext, WireRequest};
use litellm_http::{Client, CredentialClient};
use litellm_inference_testing::RecordingSecrets;
use litellm_inference_tokenize::{Error, TokenizeRequest, TokenizeRoute};
use litellm_llms_types::formats::tokenize::TokenizeInclude;
use rstest::{fixture, rstest};
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, body_string_contains, header, method, path},
};

fn route_with_credentials(
    http: Client,
    auth: Arc<AuthServices>,
    secrets: Arc<dyn litellm_secrets::source::SecretSource>,
) -> TokenizeRoute {
    TokenizeRoute::new(
        http,
        CredentialClient::no_redirect_for_test(),
        auth,
        secrets,
    )
}

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
    let first = route_with_credentials(
        Client::no_redirect_for_test(),
        auth.clone(),
        secrets.clone(),
    );
    let second = route_with_credentials(Client::no_redirect_for_test(), auth, secrets);
    let request = TokenizeRequest {
        model: "watsonx/ibm/model".into(),
        prompt: "hello world".into(),
        include: Some(TokenizeInclude::new(false, true)),
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

fn iam_route(server: &MockServer, endpoint: &str) -> TokenizeRoute {
    let api_base = server.uri();
    route_with_credentials(
        Client::no_redirect_for_test(),
        Arc::new(AuthServices::default()),
        Arc::new(RecordingSecrets::new([
            ("WATSONX_IAM_URL", endpoint),
            ("WATSONX_API_BASE", api_base.as_str()),
            ("WATSONX_API_KEY", "private-api-key"),
            ("WATSONX_PROJECT_ID", "configured-project"),
        ])),
    )
}

#[fixture]
fn request() -> TokenizeRequest {
    TokenizeRequest {
        model: "watsonx/ibm/model".into(),
        prompt: "hello".into(),
        include: Some(TokenizeInclude::new(false, true)),
        ..Default::default()
    }
}

#[rstest]
#[case::invalid_credentials(401)]
#[case::unavailable(503)]
#[tokio::test]
async fn iam_http_failures_keep_their_status_without_dispatching_inference(
    #[future(awt)] server: MockServer,
    request: TokenizeRequest,
    #[case] status: u16,
) {
    Mock::given(method("POST"))
        .and(path("/identity/token"))
        .respond_with(ResponseTemplate::new(status).set_body_string("private-response"))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(path("/ml/v1/text/tokenization"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;
    let endpoint = format!("{}/identity/token?secret=private-url", server.uri());
    let error = iam_route(&server, &endpoint)
        .execute(request, &(), None)
        .await
        .unwrap_err();
    assert!(!error.is_request());
    assert!(!error.to_string().contains("private-"));
    assert_eq!(error.http_status(), Some(status));
    assert!(matches!(
        error,
        Error::Auth(litellm_auth::Error::CredentialRequest {
            status: Some(actual),
            ..
        }) if actual == status
    ));
}

#[rstest]
#[tokio::test]
async fn iam_connection_failures_are_operational_errors_without_dispatching_inference(
    #[future(awt)] server: MockServer,
    request: TokenizeRequest,
) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let error = iam_route(&server, &format!("http://{address}/identity/token"))
        .execute(request, &(), None)
        .await
        .unwrap_err();
    assert!(!error.is_request());
    assert!(matches!(
        error,
        Error::Auth(litellm_auth::Error::CredentialRequest { status: None, .. })
    ));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[case::invalid_json("not json")]
#[case::missing_token(r#"{"expires_in":3600}"#)]
#[case::invalid_expiration(r#"{"access_token":"private-token","expires_in":"private-value"}"#)]
#[tokio::test]
async fn malformed_iam_responses_are_operational_failures(
    #[future(awt)] server: MockServer,
    request: TokenizeRequest,
    #[case] response: &str,
) {
    Mock::given(path("/identity/token"))
        .respond_with(ResponseTemplate::new(200).set_body_string(response))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(path("/ml/v1/text/tokenization"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;
    let error = iam_route(&server, &format!("{}/identity/token", server.uri()))
        .execute(request, &(), None)
        .await
        .unwrap_err();
    assert!(!error.is_request());
    assert!(!error.to_string().contains("private-"));
    assert!(matches!(
        error,
        Error::Auth(litellm_auth::Error::InvalidCredentialResponse(_))
    ));
}

#[rstest]
#[tokio::test]
async fn rejected_cached_iam_token_is_invalidated_without_replaying_provider_call(
    #[future(awt)] server: MockServer,
    request: TokenizeRequest,
) {
    let token_issuances = Arc::new(AtomicUsize::new(0));
    let token_issuances_for_response = token_issuances.clone();
    Mock::given(method("POST"))
        .and(path("/identity/token"))
        .respond_with(move |_: &wiremock::Request| {
            let access_token = if token_issuances_for_response.fetch_add(1, Ordering::SeqCst) == 0 {
                "first-token"
            } else {
                "second-token"
            };
            ResponseTemplate::new(200).set_body_json(json!({
                "access_token": access_token,
                "expires_in": 3600
            }))
        })
        .expect(2)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/ml/v1/text/tokenization"))
        .and(header("authorization", "Bearer first-token"))
        .respond_with(ResponseTemplate::new(401).set_body_string("rejected"))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/ml/v1/text/tokenization"))
        .and(header("authorization", "Bearer second-token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": {"token_count": 1, "tokens": ["hello"]}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let route = iam_route(&server, &format!("{}/identity/token", server.uri()));
    let first = route.execute(request.clone(), &(), None).await.unwrap_err();
    assert_eq!(first.http_status(), Some(401));
    let second = route.execute(request, &(), None).await.unwrap();
    assert_eq!(second.tokens, Some(vec!["hello".into()]));
    assert_eq!(token_issuances.load(Ordering::SeqCst), 2);
    assert_eq!(server.received_requests().await.unwrap().len(), 4);
}

struct SubstituteBearer;

impl Interceptors<Error> for SubstituteBearer {
    async fn before_provider_request(
        &self,
        mut wire: WireRequest,
        _: RequestContext,
    ) -> Result<WireRequest, Error> {
        wire.headers
            .retain(|(name, _)| !name.eq_ignore_ascii_case("authorization"));
        wire.headers
            .push(("authorization".into(), "Bearer substituted-token".into()));
        Ok(wire)
    }

    async fn after_provider_response(&self, _: RawResponse) -> Result<(), Error> {
        Ok(())
    }
}

#[rstest]
#[tokio::test]
async fn an_interceptor_rejection_does_not_invalidate_the_original_cached_iam_token(
    #[future(awt)] server: MockServer,
    request: TokenizeRequest,
) {
    Mock::given(method("POST"))
        .and(path("/identity/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": "original-token",
            "expires_in": 3600
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/ml/v1/text/tokenization"))
        .and(header("authorization", "Bearer substituted-token"))
        .respond_with(ResponseTemplate::new(401).set_body_string("rejected"))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/ml/v1/text/tokenization"))
        .and(header("authorization", "Bearer original-token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": {"token_count": 1, "tokens": ["hello"]}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let route = iam_route(&server, &format!("{}/identity/token", server.uri()));
    let first = route
        .execute(request.clone(), &SubstituteBearer, None)
        .await
        .unwrap_err();
    assert_eq!(first.http_status(), Some(401));
    let second = route.execute(request, &(), None).await.unwrap();
    assert_eq!(second.tokens, Some(vec!["hello".into()]));
    assert_eq!(server.received_requests().await.unwrap().len(), 3);
}
