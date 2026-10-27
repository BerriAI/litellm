use std::sync::Arc;

use axum::{
    Router,
    body::{Body, to_bytes},
    extract::FromRef,
    http::{Request, StatusCode},
    middleware::from_fn_with_state,
    routing::{get, post},
};
use futures_util::future::BoxFuture;
use litellm_auth_types::SecretValue;
use litellm_config::Config;
use litellm_gateway_auth::{
    ApiKey, ApiKeySources, Auth, Bearer, Caller, Error, authenticate, hash_token,
};
use litellm_secrets::source::SecretSource;
use rstest::{fixture, rstest};
use tower::ServiceExt;

struct Secrets;

impl SecretSource for Secrets {
    fn get_secret_str<'a>(
        &'a self,
        name: &'a str,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, litellm_secrets::Error>> {
        Box::pin(async move {
            match name {
                "MASTER_KEY" => Ok(Some(SecretValue::new("resolved-key"))),
                "EMPTY" => Ok(Some(SecretValue::new(""))),
                "ERROR" => Err(litellm_secrets::Error::ExternalRead(Box::new(
                    std::io::Error::other("private-backend-detail"),
                ))),
                _ => Ok(None),
            }
        })
    }
}

#[fixture]
fn secrets() -> Arc<dyn SecretSource> {
    Arc::new(Secrets)
}

#[rstest]
#[case::literal("literal-key", Some("Bearer literal-key"), 204)]
#[case::reference("os.environ/MASTER_KEY", Some("Bearer resolved-key"), 204)]
#[case::reference_is_not_a_token(
    "os.environ/MASTER_KEY",
    Some("Bearer os.environ/MASTER_KEY"),
    401
)]
#[case::wrong("literal-key", Some("Bearer other-key"), 401)]
#[case::missing("literal-key", None, 401)]
#[case::wrong_scheme("literal-key", Some("Basic literal-key"), 401)]
#[case::empty_token("literal-key", Some("Bearer "), 401)]
#[case::missing_reference("os.environ/MISSING", Some("Bearer os.environ/MISSING"), 500)]
#[case::empty_reference("os.environ/EMPTY", Some("Bearer token"), 500)]
#[case::empty_key("", Some("Bearer token"), 500)]
#[case::whitespace_key("   ", Some("Bearer token"), 500)]
#[case::empty_reference_name("os.environ/", Some("Bearer os.environ/"), 500)]
#[case::secret_failure("os.environ/ERROR", Some("Bearer private-backend-detail"), 500)]
#[tokio::test]
async fn enforces_configured_keys_without_exposing_secrets(
    secrets: Arc<dyn SecretSource>,
    #[case] key: &str,
    #[case] authorization: Option<&str>,
    #[case] status: u16,
) {
    let config = Config::from_yaml(&format!(
        "model_list: []\ngeneral_settings:\n  master_key: '{key}'\n"
    ))
    .unwrap();
    let app = Router::new()
        .route("/protected", get(|| async { StatusCode::NO_CONTENT }))
        .layer(from_fn_with_state(
            Auth::from_config(&config, secrets),
            authenticate::<Bearer>,
        ));
    let request = Request::get("/protected");
    let request = match authorization {
        Some(value) => request.header("authorization", value),
        None => request,
    };
    let response = app
        .oneshot(request.body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), status);
    let body = to_bytes(response.into_body(), 4096).await.unwrap();
    let text = std::str::from_utf8(&body).unwrap();
    assert!(!text.contains("private-backend-detail"));
    assert!(!text.contains("literal-key"));
    assert!(!text.contains("resolved-key"));
}

#[rstest]
fn hash_token_matches_python_sha256_hexdigest() {
    assert_eq!(
        hash_token("sk-9876"),
        "595b23af2e99cee580245f388d3244a39247a25a3846e898e9c78841f8471a3e"
    );
}

fn master_auth(secrets: Arc<dyn SecretSource>, key: &str) -> Auth {
    let config = Config::from_yaml(&format!(
        "model_list: []\ngeneral_settings:\n  master_key: {key}\n"
    ))
    .unwrap();
    Auth::from_config(&config, secrets)
}

#[rstest]
#[case::valid(Some("Bearer master-key"), None, 204)]
#[case::duplicate(Some("Bearer master-key"), Some("Bearer master-key"), 401)]
#[case::forged_extension_without_credentials(None, None, 401)]
#[tokio::test]
async fn middleware_only_trusts_the_presented_credential(
    secrets: Arc<dyn SecretSource>,
    #[case] first: Option<&str>,
    #[case] second: Option<&str>,
    #[case] status: u16,
) {
    let app = Router::new()
        .route(
            "/protected",
            get(|caller: Caller| async move {
                assert_eq!(caller, Caller::Master);
                StatusCode::NO_CONTENT
            }),
        )
        .layer(from_fn_with_state(
            master_auth(secrets, "master-key"),
            authenticate::<Bearer>,
        ));
    let request = [first, second]
        .into_iter()
        .flatten()
        .fold(Request::get("/protected"), |request, value| {
            request.header("authorization", value)
        })
        .extension(Caller::Master);
    let response = app
        .oneshot(request.body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), status);
}

#[rstest]
#[tokio::test]
async fn handlers_fail_closed_without_the_middleware() {
    let app = Router::new().route(
        "/protected",
        get(|_: Caller| async { StatusCode::NO_CONTENT }),
    );
    let response = app
        .oneshot(
            Request::get("/protected")
                .header("authorization", "Bearer master-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

struct CustomHeader(SecretValue);

impl axum::extract::FromRequestParts<Auth> for CustomHeader {
    type Rejection = Error;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        _: &Auth,
    ) -> Result<Self, Error> {
        let value = parts
            .headers
            .get("x-test-key")
            .and_then(|value| value.to_str().ok())
            .ok_or(Error::InvalidToken)?;
        Ok(Self(SecretValue::new(value)))
    }
}

impl From<CustomHeader> for SecretValue {
    fn from(value: CustomHeader) -> Self {
        value.0
    }
}

#[rstest]
#[case::selected_header("master-key", "wrong-key", 204)]
#[case::no_fallback("wrong-key", "master-key", 401)]
#[tokio::test]
async fn route_profile_uses_only_its_selected_credentials(
    secrets: Arc<dyn SecretSource>,
    #[case] custom: &str,
    #[case] bearer: &str,
    #[case] status: u16,
) {
    let app = Router::new()
        .route("/protected", get(|| async { StatusCode::NO_CONTENT }))
        .layer(from_fn_with_state(
            master_auth(secrets, "master-key"),
            authenticate::<CustomHeader>,
        ));
    let response = app
        .oneshot(
            Request::get("/protected")
                .header("x-test-key", custom)
                .header("authorization", format!("Bearer {bearer}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), status);
}

#[rstest]
#[tokio::test]
async fn master_rotation_rejects_the_old_key(secrets: Arc<dyn SecretSource>) {
    let rotated = master_auth(secrets, "second-key");
    assert_eq!(
        rotated
            .authenticate(&SecretValue::new("second-key"))
            .await
            .unwrap(),
        Caller::Master
    );
    assert!(matches!(
        rotated.authenticate(&SecretValue::new("first-key")).await,
        Err(Error::InvalidToken)
    ));
}

#[derive(Clone)]
struct ApiKeyProfile {
    auth: Auth,
    sources: ApiKeySources,
}

impl FromRef<ApiKeyProfile> for Auth {
    fn from_ref(profile: &ApiKeyProfile) -> Self {
        profile.auth.clone()
    }
}

impl FromRef<ApiKeyProfile> for ApiKeySources {
    fn from_ref(profile: &ApiKeyProfile) -> Self {
        profile.sources.clone()
    }
}

fn api_key_app(secrets: Arc<dyn SecretSource>, sources: ApiKeySources) -> Router {
    Router::new()
        .route("/protected", post(|body: String| async move { body }))
        .layer(from_fn_with_state(
            ApiKeyProfile {
                auth: master_auth(secrets, "master-key"),
                sources,
            },
            authenticate::<ApiKey>,
        ))
}

#[rstest]
#[tokio::test]
async fn api_key_extractor_selects_configured_header_and_preserves_body(
    secrets: Arc<dyn SecretSource>,
) {
    let app = api_key_app(
        secrets,
        ApiKeySources::new(Some(axum::http::HeaderName::from_static("x-selected-key"))),
    );
    let response = app
        .oneshot(
            Request::post("/protected")
                .header("x-selected-key", "Bearer master-key")
                .header("authorization", "Bearer wrong-key")
                .body(Body::from("body remains available"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        to_bytes(response.into_body(), 4096).await.unwrap(),
        "body remains available"
    );
}

#[rstest]
#[case::standard_bearer(&["Bearer master-key"], 200)]
#[case::duplicate_authorization(&["Bearer master-key", "Bearer other-key"], 401)]
#[tokio::test]
async fn api_key_extractor_reads_authorization(
    secrets: Arc<dyn SecretSource>,
    #[case] values: &[&str],
    #[case] status: u16,
) {
    let app = api_key_app(secrets, ApiKeySources::default());
    let request = values
        .iter()
        .fold(Request::post("/protected"), |request, value| {
            request.header("authorization", *value)
        })
        .body(Body::from("body"))
        .unwrap();
    assert_eq!(
        app.oneshot(request).await.unwrap().status().as_u16(),
        status
    );
}
