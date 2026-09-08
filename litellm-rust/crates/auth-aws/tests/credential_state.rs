use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use litellm_auth_aws::{Clock, CredentialScope, CredentialState, static_credentials};

struct TestClock(Arc<AtomicU64>);

impl Clock for TestClock {
    fn now(&self) -> Duration {
        Duration::from_secs(self.0.load(Ordering::SeqCst))
    }
}

#[test]
fn scopes_isolate_every_input_and_preserve_value_boundaries() {
    let scopes = BTreeSet::from([
        CredentialScope::from_optional_values("bedrock", [Some("ab"), Some("c")]),
        CredentialScope::from_optional_values("bedrock", [Some("a"), Some("bc")]),
        CredentialScope::from_optional_values("bedrock", [Some("ab"), None]),
        CredentialScope::from_optional_values("bedrock", [Some("ab"), Some("")]),
        CredentialScope::from_optional_values("federation", [Some("ab"), Some("c")]),
    ]);

    assert_eq!(scopes.len(), 5);
}

#[tokio::test]
async fn failed_acquisitions_are_not_cached_or_shared_as_success() {
    let state = CredentialState::new((), 1);
    let acquisitions = AtomicUsize::new(0);
    let scope = CredentialScope::from_optional_values("test", [Some("identity")]);
    let acquire = || async {
        let attempt = acquisitions.fetch_add(1, Ordering::SeqCst);
        tokio::task::yield_now().await;
        if attempt == 0 {
            Err("synthetic acquisition failure")
        } else {
            Ok(static_credentials("second-attempt", "secret"))
        }
    };

    let (first, second) = tokio::join!(
        state.get_or_acquire(scope.clone(), Duration::from_secs(60), acquire),
        state.get_or_acquire(scope, Duration::from_secs(60), acquire),
    );

    assert!(first.is_err() ^ second.is_err());
    assert_eq!(
        first
            .ok()
            .or_else(|| second.ok())
            .expect("one waiter reacquires")
            .access_key_id(),
        "second-attempt"
    );
    assert_eq!(acquisitions.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn concurrent_waiters_singleflight_an_expired_refresh() {
    let now = Arc::new(AtomicU64::new(0));
    let state = CredentialState::with_clock((), 1, TestClock(now.clone()));
    let acquisitions = AtomicUsize::new(0);
    let scope = CredentialScope::from_optional_values("test", [Some("identity")]);
    let acquire = || async {
        let attempt = acquisitions.fetch_add(1, Ordering::SeqCst) + 1;
        tokio::task::yield_now().await;
        Ok::<_, ()>(static_credentials(format!("attempt-{attempt}"), "secret"))
    };

    let initial = state
        .get_or_acquire(scope.clone(), Duration::from_secs(10), acquire)
        .await
        .expect("initial acquisition");
    assert_eq!(initial.access_key_id(), "attempt-1");

    now.store(10, Ordering::SeqCst);
    let (first, second) = tokio::join!(
        state.get_or_acquire(scope.clone(), Duration::from_secs(10), acquire),
        state.get_or_acquire(scope, Duration::from_secs(10), acquire),
    );

    assert_eq!(first.expect("first waiter").access_key_id(), "attempt-2");
    assert_eq!(second.expect("second waiter").access_key_id(), "attempt-2");
    assert_eq!(acquisitions.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn cache_entries_are_isolated_by_scope() {
    let state = CredentialState::new((), 2);
    let acquisitions = AtomicUsize::new(0);
    let first_scope = CredentialScope::from_optional_values("test", [Some("first")]);
    let second_scope = CredentialScope::from_optional_values("test", [Some("second")]);

    for scope in [first_scope.clone(), second_scope.clone(), first_scope] {
        state
            .get_or_acquire(scope, Duration::from_secs(60), || async {
                let attempt = acquisitions.fetch_add(1, Ordering::SeqCst) + 1;
                Ok::<_, ()>(static_credentials(format!("attempt-{attempt}"), "secret"))
            })
            .await
            .expect("credential acquisition");
    }

    assert_eq!(acquisitions.load(Ordering::SeqCst), 2);
}
