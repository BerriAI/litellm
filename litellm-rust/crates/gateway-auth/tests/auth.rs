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
        hash_token("sk-9876"),
        "595b23af2e99cee580245f388d3244a39247a25a3846e898e9c78841f8471a3e"
    );
}

use litellm_gateway_auth::{
    AccessRequest, AuthFuture, AuthenticatedCaller, AuthenticatedRequest, Authentication,
    AuthenticationMethod, Authenticator, Authorizer, Clock, Error, IdentityResolver, Permissions,
    Principal, PrincipalKind, ResolvedIdentity, VerifiedIdentity, authenticate,
};
use std::{
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
    time::{Duration, SystemTime},
};

struct TestClock(AtomicU64);

impl Clock for TestClock {
    fn now(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(self.0.load(Ordering::SeqCst))
    }
}

struct Verifier {
    authority: &'static str,
    restrictions: Permissions,
    failure: Option<u16>,
}

impl Authenticator for Verifier {
    fn verify<'a>(
        &'a self,
        credential: &'a litellm_gateway_auth::Credential,
        _: &'a axum::http::request::Parts,
    ) -> AuthFuture<'a, VerifiedIdentity> {
        Box::pin(async move {
            match self.failure {
                Some(401) => return Err(Error::InvalidToken),
                Some(_) => return Err(Error::Unavailable),
                None => (),
            }
            let litellm_gateway_auth::Credential::Token(credential) = credential else {
                return Err(Error::InvalidToken);
            };
            if credential.expose() != "test-token" {
                return Err(Error::InvalidToken);
            }
            Ok(VerifiedIdentity {
                principal: Principal::new(
                    self.authority.into(),
                    "subject".into(),
                    PrincipalKind::Service,
                ),
                authentication: Authentication {
                    method: AuthenticationMethod::External("test".into()),
                    verifier: "test-verifier".into(),
                    credential_id: "credential".into(),
                    expires_at: Some(SystemTime::UNIX_EPOCH + Duration::from_secs(100)),
                },
                restrictions: self.restrictions.clone(),
            })
        })
    }
}

struct Identities {
    mapped_principal: Option<Principal>,
    calls: AtomicUsize,
    permissions: Permissions,
}

impl IdentityResolver for Identities {
    fn resolve<'a>(&'a self, identity: &'a VerifiedIdentity) -> AuthFuture<'a, ResolvedIdentity> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ResolvedIdentity {
                principal: self
                    .mapped_principal
                    .clone()
                    .unwrap_or_else(|| identity.principal.clone()),
                permissions: self.permissions.clone(),
            })
        })
    }
}

struct Policy(AtomicUsize);

impl Authorizer for Policy {
    fn authorize<'a>(
        &'a self,
        _: &'a AuthenticatedCaller,
        _: &'a AccessRequest,
    ) -> AuthFuture<'a, ()> {
        Box::pin(async move {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    }
}

struct Pipeline {
    auth: Auth,
    identities: Arc<Identities>,
    policy: Arc<Policy>,
    clock: Arc<TestClock>,
}

fn pipeline(
    authority: &'static str,
    permissions: Permissions,
    restrictions: Permissions,
    failure: Option<u16>,
) -> Pipeline {
    let identities = Arc::new(Identities {
        mapped_principal: None,
        calls: AtomicUsize::new(0),
        permissions,
    });
    let policy = Arc::new(Policy(AtomicUsize::new(0)));
    let clock = Arc::new(TestClock(AtomicU64::new(10)));
    let auth = Auth::new(
        Arc::new(Verifier {
            authority,
            restrictions,
            failure,
        }),
        identities.clone(),
        policy.clone(),
        clock.clone(),
    );
    Pipeline {
        auth,
        identities,
        policy,
        clock,
    }
}

fn model(name: &str) -> AccessRequest {
    AccessRequest::Model {
        name: name.into(),
        deployment: name.into(),
    }
}

#[rstest]
#[case::invalid_token(401)]
#[case::unavailable_verifier(503)]
#[tokio::test]
async fn failed_verification_never_resolves_identity_or_runs_the_handler(#[case] status: u16) {
    let pipeline = pipeline("issuer-a", Permissions::All, Permissions::All, Some(status));
    let app = Router::new()
        .route("/protected", get(|| async { StatusCode::NO_CONTENT }))
        .layer(axum::middleware::from_fn_with_state(
            pipeline.auth,
            authenticate,
        ));
    let response = app
        .oneshot(
            Request::get("/protected")
                .header("authorization", "Bearer test-token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), status);
    assert_eq!(pipeline.identities.calls.load(Ordering::SeqCst), 0);
    assert_eq!(pipeline.policy.0.load(Ordering::SeqCst), 0);
}

#[rstest]
#[case::membership_restricts(Permissions::Only(Arc::from([model("allowed")])), Permissions::All)]
#[case::credential_restricts(Permissions::All, Permissions::Only(Arc::from([model("allowed")])))]
#[tokio::test]
async fn policy_cannot_expand_membership_or_credential_permissions(
    #[case] permissions: Permissions,
    #[case] restrictions: Permissions,
) {
    let pipeline = pipeline("issuer-a", permissions, restrictions, None);
    let identity = pipeline
        .auth
        .authenticate(&SecretValue::new("test-token"))
        .await
        .unwrap();
    assert!(matches!(
        identity.authorize(model("denied")).await,
        Err(Error::Forbidden)
    ));
    assert_eq!(pipeline.policy.0.load(Ordering::SeqCst), 0);
    let allowed = model("allowed");
    identity
        .authorize(allowed.clone())
        .await
        .unwrap()
        .consume(identity.caller(), &allowed)
        .unwrap();
    assert_eq!(pipeline.policy.0.load(Ordering::SeqCst), 1);
}

#[rstest]
#[tokio::test]
async fn authorization_grants_bind_the_exact_caller_and_target() {
    let first = pipeline("issuer-a", Permissions::All, Permissions::All, None);
    let second = pipeline("issuer-b", Permissions::All, Permissions::All, None);
    let caller = first
        .auth
        .authenticate(&SecretValue::new("test-token"))
        .await
        .unwrap();
    let other = second
        .auth
        .authenticate(&SecretValue::new("test-token"))
        .await
        .unwrap();
    assert_ne!(caller.caller().principal(), other.caller().principal());
    assert_ne!(
        caller.caller().session_owner(),
        other.caller().session_owner()
    );
    let access = model("allowed");
    assert!(matches!(
        caller
            .authorize(access.clone())
            .await
            .unwrap()
            .consume(other.caller(), &access),
        Err(Error::Forbidden)
    ));
    assert!(matches!(
        caller
            .authorize(access)
            .await
            .unwrap()
            .consume(caller.caller(), &model("different")),
        Err(Error::Forbidden)
    ));
}

#[rstest]
#[tokio::test]
async fn expiry_is_checked_again_for_operations_on_existing_connections() {
    let pipeline = pipeline("issuer-a", Permissions::All, Permissions::All, None);
    let identity = pipeline
        .auth
        .authenticate(&SecretValue::new("test-token"))
        .await
        .unwrap();
    assert!(identity.authorize(model("allowed")).await.is_ok());
    pipeline.clock.0.store(100, Ordering::SeqCst);
    assert!(matches!(
        identity.authorize(model("allowed")).await,
        Err(Error::Expired)
    ));
    assert!(matches!(
        pipeline
            .auth
            .authenticate(&SecretValue::new("test-token"))
            .await,
        Err(Error::Expired)
    ));
    assert_eq!(pipeline.identities.calls.load(Ordering::SeqCst), 1);
    assert_eq!(pipeline.policy.0.load(Ordering::SeqCst), 1);
}

#[rstest]
#[tokio::test]
async fn bearer_extraction_rejects_duplicates_and_does_not_trust_preexisting_identity() {
    let pipeline = pipeline("issuer-a", Permissions::All, Permissions::All, None);
    let identity = pipeline
        .auth
        .authenticate(&SecretValue::new("test-token"))
        .await
        .unwrap();
    let app = Router::new()
        .route(
            "/protected",
            get(|_: AuthenticatedRequest| async { StatusCode::NO_CONTENT }),
        )
        .layer(axum::middleware::from_fn_with_state(
            pipeline.auth,
            authenticate,
        ));
    let duplicate = Request::get("/protected")
        .header("authorization", "Bearer test-token")
        .header("authorization", "Bearer test-token")
        .body(Body::empty())
        .unwrap();
    assert_eq!(app.clone().oneshot(duplicate).await.unwrap().status(), 401);
    let forged = Request::get("/protected")
        .extension(identity)
        .body(Body::empty())
        .unwrap();
    assert_eq!(app.clone().oneshot(forged).await.unwrap().status(), 401);
    let valid = Request::get("/protected")
        .header("authorization", "Bearer test-token")
        .body(Body::empty())
        .unwrap();
    assert_eq!(app.oneshot(valid).await.unwrap().status(), 204);
}

#[rstest]
#[tokio::test]
async fn route_permissions_are_checked_before_parsing_or_dispatch() {
    let allowed = AccessRequest::Route {
        method: "GET".into(),
        path: "/allowed/{id}".into(),
    };
    let pipeline = pipeline(
        "issuer-a",
        Permissions::Only(Arc::from([allowed])),
        Permissions::All,
        None,
    );
    let app = Router::new()
        .route("/allowed/{id}", get(|| async { StatusCode::NO_CONTENT }))
        .route("/denied", get(|| async { StatusCode::NO_CONTENT }))
        .layer(axum::middleware::from_fn_with_state(
            pipeline.auth,
            authenticate,
        ));
    let allowed = Request::get("/allowed/123")
        .header("authorization", "Bearer test-token")
        .body(Body::empty())
        .unwrap();
    let denied = Request::get("/denied")
        .header("authorization", "Bearer test-token")
        .body(Body::empty())
        .unwrap();
    assert_eq!(app.clone().oneshot(allowed).await.unwrap().status(), 204);
    assert_eq!(app.oneshot(denied).await.unwrap().status(), 403);
}

struct CustomHeader;

impl litellm_gateway_auth::CredentialExtractor for CustomHeader {
    fn extract(
        &self,
        parts: &axum::http::request::Parts,
    ) -> Result<litellm_gateway_auth::Credential, Error> {
        let value = parts
            .headers
            .get("x-test-key")
            .and_then(|value| value.to_str().ok())
            .ok_or(Error::InvalidToken)?;
        Ok(litellm_gateway_auth::Credential::Token(SecretValue::new(
            value,
        )))
    }
}

#[rstest]
#[case::selected_header("test-token", "wrong-token", 204)]
#[case::no_fallback("wrong-token", "test-token", 401)]
#[tokio::test]
async fn route_profile_uses_only_its_selected_credentials(
    #[case] custom: &str,
    #[case] bearer: &str,
    #[case] status: u16,
) {
    let pipeline = pipeline("issuer-a", Permissions::All, Permissions::All, None);
    let app = Router::new()
        .route("/protected", get(|| async { StatusCode::NO_CONTENT }))
        .layer(axum::middleware::from_fn_with_state(
            pipeline.auth.with_extractor(Arc::new(CustomHeader)),
            authenticate,
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
async fn master_rotation_preserves_session_owner_but_rejects_the_old_key(
    secrets: Arc<dyn SecretSource>,
) {
    let first =
        Config::from_yaml("model_list: []\ngeneral_settings:\n  master_key: first-key\n").unwrap();
    let second =
        Config::from_yaml("model_list: []\ngeneral_settings:\n  master_key: second-key\n").unwrap();
    let original = Auth::from_config(&first, secrets.clone())
        .authenticate(&SecretValue::new("first-key"))
        .await
        .unwrap();
    let rotated = Auth::from_config(&second, secrets);
    let replacement = rotated
        .authenticate(&SecretValue::new("second-key"))
        .await
        .unwrap();
    assert_eq!(
        original.caller().session_owner(),
        replacement.caller().session_owner()
    );
    assert!(matches!(
        rotated.authenticate(&SecretValue::new("first-key")).await,
        Err(Error::InvalidToken)
    ));
}

#[rstest]
#[tokio::test]
async fn mapping_two_issuers_to_one_account_does_not_merge_session_ownership() {
    let resolver = Arc::new(Identities {
        mapped_principal: Some(Principal::new(
            "internal".into(),
            "shared-account".into(),
            PrincipalKind::Human,
        )),
        calls: AtomicUsize::new(0),
        permissions: Permissions::All,
    });
    let first = Auth::new(
        Arc::new(Verifier {
            authority: "issuer-a",
            restrictions: Permissions::All,
            failure: None,
        }),
        resolver.clone(),
        Arc::new(Policy(AtomicUsize::new(0))),
        Arc::new(TestClock(AtomicU64::new(10))),
    );
    let second = Auth::new(
        Arc::new(Verifier {
            authority: "issuer-b",
            restrictions: Permissions::All,
            failure: None,
        }),
        resolver,
        Arc::new(Policy(AtomicUsize::new(0))),
        Arc::new(TestClock(AtomicU64::new(10))),
    );
    let original = first
        .authenticate(&SecretValue::new("test-token"))
        .await
        .unwrap();
    let mapped = second
        .authenticate(&SecretValue::new("test-token"))
        .await
        .unwrap();
    assert_eq!(original.caller().principal(), mapped.caller().principal());
    assert_ne!(
        original.caller().verified_principal(),
        mapped.caller().verified_principal()
    );
    assert_ne!(
        original.caller().session_owner(),
        mapped.caller().session_owner()
    );
}
