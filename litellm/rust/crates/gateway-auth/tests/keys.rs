use std::{
    sync::Arc,
    time::{Duration, SystemTime},
};

use axum::http::request::Parts;
use litellm_gateway_auth::{
    AccessRequest, Auth, AuthFuture, AuthenticationMethod, Authenticator, Clock, Credential, Error,
    KeyHolder, NoAdditionalPolicy, VerifiedIdentity,
    keys::{HeldKey, KeyHash},
};
use litellm_secret_value::SecretValue;
use rstest::rstest;

const VIRTUAL_KEY: &str = "sk-test-credential";
const NOW: u64 = 100;

struct At(u64);

impl Clock for At {
    fn now(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(self.0)
    }
}

/// The row of the one key a host authenticated.
struct Held {
    expires_at: Option<SystemTime>,
    models: Arc<[String]>,
}

impl HeldKey for Held {
    fn expires_at(&self) -> Option<SystemTime> {
        self.expires_at
    }

    fn models(&self) -> Arc<[String]> {
        self.models.clone()
    }
}

/// A host that authenticated every credential as the virtual key it holds.
impl Authenticator for Held {
    fn verify<'a>(
        &'a self,
        credential: &'a Credential,
        _: &'a Parts,
    ) -> AuthFuture<'a, VerifiedIdentity> {
        Box::pin(async move {
            let Credential::Token(token) = credential else {
                return Err(Error::InvalidToken);
            };
            Ok(VerifiedIdentity::virtual_key(
                &KeyHash::from_token(token),
                self,
            ))
        })
    }
}

fn auth(held: Held) -> Auth {
    Auth::new(
        Arc::new(held),
        Arc::new(KeyHolder),
        Arc::new(NoAdditionalPolicy),
        Arc::new(At(NOW)),
    )
}

fn expiring(at: Option<u64>) -> Held {
    Held {
        expires_at: at.map(|seconds| SystemTime::UNIX_EPOCH + Duration::from_secs(seconds)),
        models: Arc::from([]),
    }
}

/// A request for `name`, which the router resolves to the model group `group` whose deployments
/// carry `access_groups`.
fn model(name: &str, group: &str, access_groups: &[&str]) -> AccessRequest {
    AccessRequest::Model {
        name: name.into(),
        group: group.into(),
        access_groups: access_groups
            .iter()
            .map(|group| (*group).to_owned())
            .collect(),
        deployment: "anthropic/claude".into(),
    }
}

#[rstest]
#[case::no_expiration(None)]
#[case::future(Some(NOW + 1))]
#[tokio::test]
async fn an_active_virtual_key_reaches_every_route_and_model(#[case] expires_at: Option<u64>) {
    let identity = auth(expiring(expires_at))
        .authenticate(&SecretValue::new(VIRTUAL_KEY))
        .await
        .unwrap();

    let authentication = identity.caller().authentication();
    assert_eq!(authentication.method, AuthenticationMethod::VirtualKey);
    assert_eq!(
        authentication.credential_id,
        KeyHash::from_token(&SecretValue::new(VIRTUAL_KEY)).as_str()
    );
    for request in [
        AccessRequest::Route {
            method: "POST".into(),
            path: "/v1/messages".into(),
        },
        model("claude", "claude", &[]),
    ] {
        assert!(identity.authorize(request).await.is_ok());
    }
}

#[rstest]
#[case::boundary(Some(NOW))]
#[case::past(Some(NOW - 1))]
#[tokio::test]
async fn an_expired_virtual_key_is_refused(#[case] expires_at: Option<u64>) {
    let result = auth(expiring(expires_at))
        .authenticate(&SecretValue::new(VIRTUAL_KEY))
        .await;

    assert!(matches!(result, Err(Error::Expired)));
}

#[rstest]
#[case::the_name_asked_for(&["claude"], model("claude", "claude", &[]), true)]
#[case::another_name(&["claude"], model("gpt", "gpt", &[]), false)]
#[case::the_group_an_alias_names(&["claude"], model("sonnet", "claude", &[]), true)]
#[case::only_the_alias(&["sonnet"], model("claude", "claude", &[]), false)]
#[case::an_access_group_of_the_group(&["haiku"], model("claude", "claude", &["haiku", "fast"]), true)]
#[case::an_access_group_of_another_group(&["haiku"], model("gpt", "gpt", &["fast"]), false)]
#[case::every_model(&["*"], model("gpt", "gpt", &[]), true)]
#[case::every_proxy_model(&["all-proxy-models"], model("gpt", "gpt", &[]), true)]
#[case::a_name_that_only_starts_the_same(&["claude"], model("claude-down", "claude-down", &[]), false)]
#[tokio::test]
async fn a_key_with_a_model_list_may_call_only_what_the_proxy_lets_it(
    #[case] models: &[&str],
    #[case] request: AccessRequest,
    #[case] allowed: bool,
) {
    let held = Held {
        expires_at: None,
        models: models.iter().map(|model| (*model).to_owned()).collect(),
    };
    let identity = auth(held)
        .authenticate(&SecretValue::new(VIRTUAL_KEY))
        .await
        .unwrap();

    let route = AccessRequest::Route {
        method: "POST".into(),
        path: "/v1/messages".into(),
    };
    assert!(identity.authorize(route).await.is_ok());
    let authorized = identity.authorize(request).await;
    assert_eq!(authorized.is_ok(), allowed);
    assert!(authorized.is_ok() || matches!(authorized, Err(Error::Forbidden)));
}
