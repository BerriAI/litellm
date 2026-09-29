#[path = "keys/support.rs"]
mod support;

use std::{
    sync::Arc,
    time::{Duration, SystemTime},
};

use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use litellm_auth_types::SecretValue;
use litellm_gateway_auth::{
    KeyError, hash_token,
    keys::{KeyHash, KeyLookup, KeyStatus, verify_key},
};
use litellm_gateway_management::{
    Error,
    keys::{Keys, Revocation},
};
use rstest::{fixture, rstest};
use support::{MemoryStore, Unavailable};

#[fixture]
fn store() -> Arc<MemoryStore> {
    Arc::new(MemoryStore::default())
}

#[fixture]
fn now() -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(100)
}

#[rstest]
#[case::permanent(false)]
#[case::expiring(true)]
#[tokio::test]
async fn generated_key_can_be_retrieved_verified_and_revoked(
    store: Arc<MemoryStore>,
    now: SystemTime,
    #[case] expiring: bool,
) {
    let keys = Keys::new(store.clone());
    let lookup: Arc<dyn KeyLookup> = store;
    let expires_at = expiring.then_some(now + Duration::from_secs(60));

    let generated = keys.generate(expires_at, now).await.unwrap();
    let hash = &generated.record.hash;

    assert_eq!(hash.as_str(), hash_token(generated.token.expose()));
    assert_eq!(generated.record.status, KeyStatus::Active { expires_at });
    assert_eq!(
        keys.get(hash).await.unwrap(),
        Some(generated.record.clone())
    );
    assert_eq!(
        verify_key(lookup.as_ref(), &generated.token, || now)
            .await
            .unwrap(),
        *hash
    );
    assert!(!format!("{generated:?}").contains(generated.token.expose()));
    assert_eq!(
        URL_SAFE_NO_PAD
            .decode(generated.token.expose().strip_prefix("sk-").unwrap())
            .unwrap()
            .len(),
        32
    );

    assert_eq!(keys.revoke(hash).await.unwrap(), Revocation::Revoked);
    assert_eq!(
        keys.get(hash).await.unwrap().unwrap().status,
        KeyStatus::Revoked
    );
    assert!(matches!(
        verify_key(lookup.as_ref(), &generated.token, || now).await,
        Err(KeyError::Invalid)
    ));
    assert_eq!(keys.revoke(hash).await.unwrap(), Revocation::Revoked);
}

#[rstest]
#[tokio::test]
async fn generates_independent_credentials(store: Arc<MemoryStore>, now: SystemTime) {
    let keys = Keys::new(store.clone());
    let first = keys.generate(None, now).await.unwrap();
    let second = keys.generate(None, now).await.unwrap();

    assert_ne!(first.token, second.token);
    assert_ne!(first.record.hash, second.record.hash);
    keys.revoke(&first.record.hash).await.unwrap();
    assert_eq!(
        verify_key(store.as_ref(), &second.token, || now)
            .await
            .unwrap(),
        second.record.hash
    );
}

#[rstest]
#[case::past(99)]
#[case::boundary(100)]
#[tokio::test]
async fn invalid_expiration_is_rejected_before_storage(now: SystemTime, #[case] seconds: u64) {
    let keys = Keys::new(Arc::new(Unavailable));

    let result = keys
        .generate(
            Some(SystemTime::UNIX_EPOCH + Duration::from_secs(seconds)),
            now,
        )
        .await;

    assert!(matches!(result, Err(Error::InvalidExpiration)));
}

#[rstest]
#[tokio::test]
async fn missing_key_is_not_created_by_get_or_revoke(store: Arc<MemoryStore>) {
    let keys = Keys::new(store);
    let hash = KeyHash::from_token(&SecretValue::new("sk-missing"));

    assert_eq!(keys.get(&hash).await.unwrap(), None);
    assert_eq!(keys.revoke(&hash).await.unwrap(), Revocation::NotFound);
    assert_eq!(keys.get(&hash).await.unwrap(), None);
}

#[rstest]
#[tokio::test]
async fn storage_failures_are_returned_for_all_operations(now: SystemTime) {
    let keys = Keys::new(Arc::new(Unavailable));
    let hash = KeyHash::from_token(&SecretValue::new("sk-unavailable"));

    assert!(matches!(
        keys.generate(None, now).await,
        Err(Error::Storage(_))
    ));
    assert!(matches!(keys.get(&hash).await, Err(Error::Storage(_))));
    let error = keys.revoke(&hash).await.unwrap_err();
    assert!(matches!(&error, Error::Storage(_)));
    assert_eq!(
        std::error::Error::source(&error).unwrap().to_string(),
        "backend detail"
    );
    assert!(!error.to_string().contains("backend detail"));
}
