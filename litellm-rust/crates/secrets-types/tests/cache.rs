use std::{convert::Infallible, time::Duration};

use litellm_secrets_types::SecretCache;
use rstest::rstest;
use tokio::sync::oneshot;

#[rstest]
#[case::delete(false)]
#[case::write(true)]
#[tokio::test]
async fn an_old_load_cannot_restore_a_mutated_entry(#[case] write: bool) {
    let cache = SecretCache::new(10, Duration::from_secs(60));
    let (started, loading) = oneshot::channel();
    let (release, finish) = oneshot::channel();
    let old = cache.read("key", async {
        started.send(()).unwrap();
        finish.await.unwrap();
        Ok::<_, Infallible>(Some("old"))
    });
    let mutation = async {
        loading.await.unwrap();
        if write {
            cache.insert("key", "new").await;
        } else {
            cache.invalidate(&"key").await;
        }
        release.send(()).unwrap();
    };
    let (old_result, ()) = tokio::join!(old, mutation);
    assert_eq!(old_result.unwrap(), Some("old"));
    let current = cache
        .read("key", async { Ok::<_, Infallible>(None) })
        .await
        .unwrap();
    assert_eq!(current, write.then_some("new"));
}

#[tokio::test]
async fn location_invalidation_detaches_all_projections_and_keeps_other_secrets() {
    let cache = SecretCache::new(10, Duration::from_secs(60));
    cache.insert(("target", "first"), "old").await;
    cache.insert(("unrelated", "first"), "retained").await;
    let (started, loading) = oneshot::channel();
    let (release, finish) = oneshot::channel();
    let old = cache.read(("target", "second"), async {
        started.send(()).unwrap();
        finish.await.unwrap();
        Ok::<_, Infallible>(Some("old"))
    });
    let mutation = async {
        loading.await.unwrap();
        cache.invalidate_where(|(location, _)| *location == "target");
        release.send(()).unwrap();
    };
    let (result, ()) = tokio::join!(old, mutation);
    assert_eq!(result.unwrap(), Some("old"));
    for projection in ["first", "second"] {
        assert_eq!(
            cache
                .read(("target", projection), async { Ok::<_, Infallible>(None) })
                .await
                .unwrap(),
            None
        );
    }
    assert_eq!(
        cache
            .read(("unrelated", "first"), async { Ok::<_, Infallible>(None) })
            .await
            .unwrap(),
        Some("retained")
    );
}

#[tokio::test]
async fn concurrent_misses_share_a_load_and_cancellation_allows_a_retry() {
    let cache = SecretCache::new(10, Duration::from_secs(60));
    let (started, loading) = oneshot::channel();
    let (release, finish) = oneshot::channel();
    let first = cache.read("key", async {
        started.send(()).unwrap();
        finish.await.unwrap();
        Ok::<_, Infallible>(Some("value"))
    });
    let second = async {
        loading.await.unwrap();
        release.send(()).unwrap();
        cache.read("key", async { panic!("duplicate load") }).await
    };
    let (first, second): (_, Result<_, Infallible>) = tokio::join!(first, second);
    assert_eq!(first.unwrap(), Some("value"));
    assert_eq!(second.unwrap(), Some("value"));

    let (started, loading) = oneshot::channel();
    let cancelled = cache.read("cancelled", async {
        started.send(()).unwrap();
        std::future::pending::<Result<Option<&str>, Infallible>>().await
    });
    tokio::select! {
        _ = loading => {},
        _ = cancelled => panic!("load must remain pending"),
    }
    assert_eq!(
        cache
            .read("cancelled", async { Ok::<_, Infallible>(Some("retry")) })
            .await
            .unwrap(),
        Some("retry")
    );
}

#[tokio::test]
async fn refresh_bypasses_cached_values_and_errors_and_absence_are_retried() {
    let cache = SecretCache::new(10, Duration::from_secs(60));
    cache.insert("key", "old").await;
    assert_eq!(
        cache
            .refresh("key", async { Ok::<_, Infallible>(Some("fresh")) })
            .await
            .unwrap(),
        Some("fresh")
    );
    assert_eq!(
        cache
            .read("key", async { Ok::<_, Infallible>(None) })
            .await
            .unwrap(),
        Some("fresh")
    );
    for result in [Err("failure"), Ok(None), Ok(Some("recovered"))] {
        assert_eq!(cache.read("retry", async { result }).await, result);
    }
}

#[tokio::test]
async fn expired_entries_reload_and_empty_values_are_cached() {
    let cache = SecretCache::new(10, Duration::from_secs(60));
    assert_eq!(
        cache
            .read("empty", async { Ok::<_, Infallible>(Some("")) })
            .await
            .unwrap(),
        Some("")
    );
    assert_eq!(
        cache
            .read("empty", async { Ok::<_, Infallible>(Some("changed")) })
            .await
            .unwrap(),
        Some("")
    );
    let expiring = SecretCache::new(10, Duration::from_nanos(1));
    expiring.insert("key", "old").await;
    tokio::time::sleep(Duration::from_millis(1)).await;
    assert_eq!(
        expiring
            .read("key", async { Ok::<_, Infallible>(Some("new")) })
            .await
            .unwrap(),
        Some("new")
    );
}
