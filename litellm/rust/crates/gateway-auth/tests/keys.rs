use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, SystemTime},
};

use litellm_gateway_auth::{
    AccessRequest, Auth, AuthenticationMethod, Clock, Error, KeyAuthenticator, KeyHolder,
    NoAdditionalPolicy,
    keys::{KeyHash, KeyLookup, KeyStatus, LookupFuture},
};
use litellm_secret_value::SecretValue;
use rstest::rstest;

const MASTER_KEY: &str = "sk-master-credential";
const VIRTUAL_KEY: &str = "sk-test-credential";
const NOW: u64 = 100;

struct At(u64);

impl Clock for At {
    fn now(&self) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(self.0)
    }
}

struct Lookup {
    status: Option<KeyStatus>,
    asked: AtomicUsize,
}

impl KeyLookup for Lookup {
    fn lookup<'a>(&'a self, hash: &'a KeyHash) -> LookupFuture<'a> {
        Box::pin(async move {
            self.asked.fetch_add(1, Ordering::SeqCst);
            Ok(self
                .status
                .clone()
                .filter(|_| hash == &KeyHash::from_token(&SecretValue::new(VIRTUAL_KEY))))
        })
    }
}

struct Unavailable;

impl KeyLookup for Unavailable {
    fn lookup<'a>(&'a self, _: &'a KeyHash) -> LookupFuture<'a> {
        Box::pin(async {
            Err(Error::Unavailable(litellm_error::Source::new(
                std::io::Error::other("backend detail"),
            )))
        })
    }
}

fn auth(keys: Arc<dyn KeyLookup>) -> Auth {
    Auth::new(
        Arc::new(KeyAuthenticator::new(
            Some(SecretValue::new(MASTER_KEY)),
            keys,
        )),
        Arc::new(KeyHolder),
        Arc::new(NoAdditionalPolicy),
        Arc::new(At(NOW)),
    )
}

fn holding(status: Option<KeyStatus>) -> Arc<Lookup> {
    Arc::new(Lookup {
        status,
        asked: AtomicUsize::new(0),
    })
}

fn expiring(at: Option<u64>) -> Option<KeyStatus> {
    Some(KeyStatus::Active {
        expires_at: at.map(|seconds| SystemTime::UNIX_EPOCH + Duration::from_secs(seconds)),
    })
}

#[rstest]
#[case::no_expiration(None)]
#[case::future(Some(NOW + 1))]
#[tokio::test]
async fn an_active_virtual_key_reaches_every_route_and_model(#[case] expires_at: Option<u64>) {
    let identity = auth(holding(expiring(expires_at)))
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
        AccessRequest::Model {
            name: "claude".into(),
            deployment: "anthropic/claude".into(),
        },
    ] {
        assert!(identity.authorize(request).await.is_ok());
    }
}

#[rstest]
#[case::boundary(Some(NOW))]
#[case::past(Some(NOW - 1))]
#[tokio::test]
async fn an_expired_virtual_key_is_refused(#[case] expires_at: Option<u64>) {
    let result = auth(holding(expiring(expires_at)))
        .authenticate(&SecretValue::new(VIRTUAL_KEY))
        .await;

    assert!(matches!(result, Err(Error::Expired)));
}

#[rstest]
#[case::missing(None, VIRTUAL_KEY)]
#[case::revoked(Some(KeyStatus::Revoked), VIRTUAL_KEY)]
#[case::a_different_token(expiring(None), "sk-other-credential")]
#[tokio::test]
async fn a_key_the_lookup_does_not_report_active_is_refused(
    #[case] status: Option<KeyStatus>,
    #[case] token: &str,
) {
    let keys = holding(status);

    let result = auth(keys.clone())
        .authenticate(&SecretValue::new(token))
        .await;

    assert!(matches!(result, Err(Error::InvalidToken)));
    assert_eq!(keys.asked.load(Ordering::SeqCst), 1);
}

#[rstest]
#[case::the_master_key(MASTER_KEY, Some(AuthenticationMethod::MasterKey))]
#[case::a_token_that_is_not_a_key("eyJhbGciOiJSUzI1NiJ9.e30.c2ln", None)]
#[tokio::test]
async fn only_a_virtual_key_is_looked_up(
    #[case] token: &str,
    #[case] method: Option<AuthenticationMethod>,
) {
    let keys = holding(expiring(None));

    let identity = auth(keys.clone())
        .authenticate(&SecretValue::new(token))
        .await
        .ok();

    assert_eq!(
        identity.map(|identity| identity.caller().authentication().method.clone()),
        method
    );
    assert_eq!(keys.asked.load(Ordering::SeqCst), 0);
}

#[rstest]
#[tokio::test]
async fn a_lookup_that_fails_is_unavailable_and_never_a_denial() {
    let result = auth(Arc::new(Unavailable))
        .authenticate(&SecretValue::new(VIRTUAL_KEY))
        .await;

    assert!(matches!(&result, Err(error @ Error::Unavailable(_)) if !error.is_denied()));
}
