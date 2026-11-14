use std::time::{Duration, SystemTime};

use litellm_gateway_auth::{
    Error,
    keys::{KeyHash, KeyLookup, KeyStatus, LookupFuture, verify_key},
};
use litellm_secret_value::SecretValue;
use rstest::{fixture, rstest};

struct Lookup {
    hash: KeyHash,
    status: Option<KeyStatus>,
}

impl KeyLookup for Lookup {
    fn lookup<'a>(&'a self, hash: &'a KeyHash) -> LookupFuture<'a> {
        Box::pin(async move {
            Ok(if hash == &self.hash {
                self.status.clone()
            } else {
                None
            })
        })
    }
}

#[fixture]
fn token() -> SecretValue {
    SecretValue::new("sk-test-credential")
}

#[fixture]
fn now() -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(100)
}

#[rstest]
#[case::no_expiration(None, true)]
#[case::future(Some(101), true)]
#[case::boundary(Some(100), false)]
#[case::past(Some(99), false)]
#[tokio::test]
async fn checks_expiration(
    token: SecretValue,
    now: SystemTime,
    #[case] expiration: Option<u64>,
    #[case] valid: bool,
) {
    let hash = KeyHash::from_token(&token);
    let lookup = Lookup {
        hash: hash.clone(),
        status: Some(KeyStatus::Active {
            expires_at: expiration
                .map(|seconds| SystemTime::UNIX_EPOCH + Duration::from_secs(seconds)),
        }),
    };

    let result = verify_key(&lookup, &token, || now).await;

    match valid {
        true => assert_eq!(result.unwrap(), hash),
        false => assert!(matches!(result, Err(Error::Expired))),
    }
}

#[rstest]
#[case::missing(None)]
#[case::revoked(Some(KeyStatus::Revoked))]
#[tokio::test]
async fn rejects_unusable_keys(
    token: SecretValue,
    now: SystemTime,
    #[case] status: Option<KeyStatus>,
) {
    let lookup = Lookup {
        hash: KeyHash::from_token(&token),
        status,
    };

    assert!(matches!(
        verify_key(&lookup, &token, || now).await,
        Err(Error::InvalidToken)
    ));
}

#[rstest]
#[tokio::test]
async fn rejects_a_different_token(token: SecretValue, now: SystemTime) {
    let lookup = Lookup {
        hash: KeyHash::from_token(&token),
        status: Some(KeyStatus::Active { expires_at: None }),
    };

    assert!(matches!(
        verify_key(&lookup, &SecretValue::new("sk-other-credential"), || now).await,
        Err(Error::InvalidToken)
    ));
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

#[rstest]
#[tokio::test]
async fn lookup_failure_preserves_source_without_exposing_it(token: SecretValue, now: SystemTime) {
    let error = verify_key(&Unavailable, &token, || now).await.unwrap_err();

    assert_eq!(error.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
    let cause = litellm_error::find::<std::io::Error>(&error).expect("lookup cause");
    assert_eq!(cause.to_string(), "backend detail");
    assert!(!error.to_string().contains("backend detail"));
}
