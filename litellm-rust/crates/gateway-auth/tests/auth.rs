use std::sync::Arc;

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
    middleware::from_extractor_with_state,
    routing::get,
};
use futures_util::future::BoxFuture;
use litellm_auth_types::SecretValue;
use litellm_config::Config;
use litellm_gateway_auth::{Auth, RequireMasterKey, hash_token};
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
#[case::empty_reference("os.environ/EMPTY", Some("Bearer "), 500)]
#[case::empty_key("", Some("Bearer "), 500)]
#[case::whitespace_key("   ", Some("Bearer "), 500)]
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
        .layer(from_extractor_with_state::<RequireMasterKey, _>(
            Auth::from_config(&config, secrets),
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
        hash_token("sk-1234"),
        "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"
    );
}
