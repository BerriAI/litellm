use std::sync::Arc;

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode, request::Parts},
    middleware::from_extractor_with_state,
    routing::get,
};
use litellm_gateway_auth::{
    Auth, AuthFuture, Authenticated, Authenticator, Credential, CredentialExtractor, Error,
    RequireMasterKey, authenticate,
};
use litellm_gateway_auth_core::{
    self as auth_core, Belongs, Caller, KeyHash, Rows, authn::key::KeyRecord,
};
use litellm_secret_value::SecretValue;
use rstest::rstest;
use tower::ServiceExt;

const MASTER_TOKEN: &str = "master-token";
const KEY_TOKEN: &str = "key-token";

#[derive(Clone, Debug, PartialEq)]
struct Held;

impl Rows for Held {
    type Key = KeyRecord;
    type User = ();
    type Team = ();
    type EndUser = ();
    type ObjectPermission = ();
    type Project = ();
    type Membership = ();
    type Org = ();
    type Agent = ();
}

fn stored_key() -> Caller<Held> {
    Caller {
        credential: auth_core::Credential::Key {
            key: KeyRecord {
                token: Some(KeyHash::new("stored")),
                ..KeyRecord::default()
            },
        },
        belongs: Belongs::nobody(),
        end_user: (),
    }
}

fn master_key() -> Caller<Held> {
    Caller {
        credential: auth_core::Credential::MasterKey,
        belongs: Belongs::nobody(),
        end_user: (),
    }
}

/// Knows [`MASTER_TOKEN`] and [`KEY_TOKEN`], and answers a request that presents no token the way
/// a server with no master key does.
struct Verifier {
    failure: Option<u16>,
}

impl Authenticator<Held> for Verifier {
    fn authenticate<'a>(
        &'a self,
        credential: &'a Credential,
        _: &'a Parts,
    ) -> AuthFuture<'a, Caller<Held>> {
        Box::pin(async move {
            match (self.failure, credential) {
                (Some(401), _) => Err(Error::InvalidToken),
                (Some(_), _) => Err(Error::Unavailable(litellm_error::Source::new(
                    std::io::Error::other("verifier offline"),
                ))),
                (None, Credential::Transport) => Err(Error::Unconfigured),
                (None, Credential::Token(token)) => match token.expose() {
                    MASTER_TOKEN => Ok(master_key()),
                    KEY_TOKEN => Ok(stored_key()),
                    _ => Err(Error::InvalidToken),
                },
            }
        })
    }
}

fn auth(failure: Option<u16>) -> Auth<Held> {
    Auth::new(Arc::new(Verifier { failure }))
}

/// Answers with the stored hash of the caller the request was authenticated as.
fn protected(auth: Auth<Held>) -> Router {
    Router::new()
        .route(
            "/protected",
            get(|caller: Authenticated<Held>| async move {
                caller
                    .hash()
                    .map_or("master".to_owned(), |hash| hash.as_str().to_owned())
            }),
        )
        .layer(axum::middleware::from_fn_with_state(
            auth,
            authenticate::<Held>,
        ))
}

fn bearer(token: &str) -> Request<Body> {
    Request::get("/protected")
        .header("authorization", format!("Bearer {token}"))
        .body(Body::empty())
        .unwrap()
}

#[rstest]
#[case::invalid_token(401)]
#[case::unavailable_authenticator(503)]
#[tokio::test]
async fn a_failed_authentication_is_answered_in_its_own_status(#[case] status: u16) {
    let response = protected(auth(Some(status)))
        .oneshot(bearer(MASTER_TOKEN))
        .await
        .unwrap();

    assert_eq!(response.status().as_u16(), status);
}

#[rstest]
#[case::the_master_key(MASTER_TOKEN, "master")]
#[case::a_key(KEY_TOKEN, "stored")]
#[tokio::test]
async fn a_handler_reads_the_caller_the_authenticator_answered(
    #[case] token: &str,
    #[case] caller: &str,
) {
    let response = protected(auth(None)).oneshot(bearer(token)).await.unwrap();

    assert_eq!(response.status(), 200);
    let body = to_bytes(response.into_body(), 4096).await.unwrap();
    assert_eq!(body, caller);
}

#[rstest]
#[tokio::test]
async fn bearer_extraction_rejects_duplicates_and_does_not_trust_a_preexisting_caller() {
    let app = protected(auth(None));
    let duplicate = Request::get("/protected")
        .header("authorization", format!("Bearer {MASTER_TOKEN}"))
        .header("authorization", format!("Bearer {MASTER_TOKEN}"))
        .body(Body::empty())
        .unwrap();
    assert_eq!(app.clone().oneshot(duplicate).await.unwrap().status(), 401);
    let forged = Request::get("/protected")
        .extension(Authenticated::<Held>::from(master_key()))
        .body(Body::empty())
        .unwrap();
    assert_eq!(app.clone().oneshot(forged).await.unwrap().status(), 401);
    assert_eq!(
        app.oneshot(bearer(MASTER_TOKEN)).await.unwrap().status(),
        200
    );
}

struct CustomHeader;

impl CredentialExtractor for CustomHeader {
    fn extract(&self, parts: &Parts) -> Result<Credential, Error> {
        let value = parts
            .headers
            .get("x-test-key")
            .and_then(|value| value.to_str().ok())
            .ok_or(Error::InvalidToken)?;
        Ok(Credential::Token(SecretValue::new(value)))
    }
}

#[rstest]
#[case::selected_header(MASTER_TOKEN, "wrong-token", 200)]
#[case::no_fallback("wrong-token", MASTER_TOKEN, 401)]
#[tokio::test]
async fn route_profile_uses_only_its_selected_credentials(
    #[case] custom: &str,
    #[case] bearer: &str,
    #[case] status: u16,
) {
    let app = protected(auth(None).with_extractor(Arc::new(CustomHeader)));
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
#[case::the_master_key(Some("Bearer master-token"), 204)]
#[case::a_key(Some("Bearer key-token"), 401)]
#[case::an_unknown_token(Some("Bearer other-token"), 401)]
#[case::no_token_is_still_put_to_the_authenticator(None, 500)]
#[case::a_malformed_header_is_no_token(Some("Basic master-token"), 500)]
#[tokio::test]
async fn require_master_key_admits_the_master_key_alone(
    #[case] authorization: Option<&str>,
    #[case] status: u16,
) {
    let app = Router::new()
        .route("/protected", get(|| async { StatusCode::NO_CONTENT }))
        .layer(from_extractor_with_state::<RequireMasterKey, _>(auth(None)));
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
}
