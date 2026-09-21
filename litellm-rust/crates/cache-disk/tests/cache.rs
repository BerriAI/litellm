use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    thread,
    time::Duration,
};

use litellm_cache::{
    BaseCache, BatchCache, BatchEntry, CounterCache, DeleteCache, ExactCacheContext, FlushCache,
    JsonCodec,
};
use litellm_cache_disk::{DiskCache, DiskStore, DiskcacheSqliteStore, StoredValue};
use rusqlite::Connection;
use serde_json::json;
use tempfile::TempDir;

fn store() -> (TempDir, DiskcacheSqliteStore) {
    let directory = tempfile::tempdir().unwrap();
    let store = DiskcacheSqliteStore::open(directory.path()).unwrap();
    (directory, store)
}

fn cache(directory: &Path) -> DiskCache<JsonCodec<serde_json::Value>> {
    DiskCache::open(directory, JsonCodec::new()).unwrap()
}

fn value_files(directory: &Path) -> Vec<PathBuf> {
    fn visit(directory: &Path, files: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, files);
            } else if path.extension().is_some_and(|extension| extension == "val") {
                files.push(path);
            }
        }
    }

    let mut files = Vec::new();
    visit(directory, &mut files);
    files
}

#[test]
fn roundtrip_persists_and_reopens() {
    let directory = tempfile::tempdir().unwrap();
    let context = ExactCacheContext::default();
    let opened = cache(directory.path());
    opened
        .set_cache("key", json!({"answer": 42}), &context)
        .unwrap();
    assert_eq!(
        opened.get_cache("key", &context).unwrap(),
        Some(json!({"answer": 42}))
    );
    drop(opened);
    let reopened = cache(directory.path());
    assert_eq!(
        reopened.get_cache("key", &context).unwrap(),
        Some(json!({"answer": 42}))
    );
}

#[test]
fn ttl_and_expired_culling_match_cache_contract() {
    let (directory, store) = store();
    store
        .set(
            "expired",
            StoredValue::Bytes(b"old".to_vec()),
            Some(10.0),
            0.0,
        )
        .unwrap();
    assert_eq!(store.get("expired", 10.0).unwrap(), None);
    store
        .set("new", StoredValue::Bytes(b"new".to_vec()), None, 11.0)
        .unwrap();
    let connection = Connection::open(directory.path().join("cache.db")).unwrap();
    assert_eq!(
        connection
            .query_row("SELECT COUNT(*) FROM Cache", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT value FROM Settings WHERE key = 'count'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        1
    );
}

#[test]
fn batch_preserves_order_and_classifies_misses_and_invalid_values() {
    let (directory, store) = store();
    store
        .set(
            "hit",
            StoredValue::Bytes(br#"{"ok":true}"#.to_vec()),
            None,
            0.0,
        )
        .unwrap();
    store
        .set(
            "invalid",
            StoredValue::Pickle(vec![0x80, 0x05, 0x2e]),
            None,
            0.0,
        )
        .unwrap();
    let cache = cache(directory.path());
    let entries = cache
        .batch_get_cache(
            &["hit".into(), "missing".into(), "invalid".into()],
            &ExactCacheContext::default(),
        )
        .unwrap();
    assert_eq!(
        entries,
        vec![
            BatchEntry::Hit(json!({"ok": true})),
            BatchEntry::Miss,
            BatchEntry::Invalid
        ]
    );
}

#[test]
fn falsy_values_are_misses_and_protocol_five_pickle_decodes() {
    let (directory, store) = store();
    for (key, value) in [
        ("empty-bytes", StoredValue::Bytes(Vec::new())),
        ("empty-text", StoredValue::Text(String::new())),
        ("zero-int", StoredValue::Integer(0)),
        ("zero-float", StoredValue::Float(0.0)),
        (
            "empty-pickle",
            StoredValue::Pickle(vec![0x80, 0x05, 0x7d, 0x94, 0x2e]),
        ),
    ] {
        store.set(key, value, None, 0.0).unwrap();
    }
    store
        .set(
            "pickle",
            StoredValue::Pickle(
                b"\x80\x05\x95\x30\x00\x00\x00\x00\x00\x00\x00\x7d\x94\x28\x8c\x09timestamp\x94G\x3f\xf8\x00\x00\x00\x00\x00\x00\x8c\x08response\x94\x8c\x08{\"a\": 1}\x94u."
                    .to_vec(),
            ),
            None,
            0.0,
        )
        .unwrap();
    let cache = cache(directory.path());
    for key in [
        "empty-bytes",
        "empty-text",
        "zero-int",
        "zero-float",
        "empty-pickle",
    ] {
        assert_eq!(
            cache.get_cache(key, &ExactCacheContext::default()).unwrap(),
            None
        );
    }
    assert_eq!(
        cache
            .get_cache("pickle", &ExactCacheContext::default())
            .unwrap(),
        Some(json!({"timestamp": 1.5, "response": "{\"a\": 1}"}))
    );
}

#[test]
fn counters_use_atomic_native_values_and_ignore_invalid_initial_values() {
    let (directory, store) = store();
    store
        .set("counter", StoredValue::Integer(2), None, 0.0)
        .unwrap();
    store
        .set(
            "invalid",
            StoredValue::Text("not a number".into()),
            None,
            0.0,
        )
        .unwrap();
    store
        .set(
            "pickle-counter",
            StoredValue::Pickle(vec![0x80, 0x05, 0x4b, 0x02, 0x2e]),
            None,
            0.0,
        )
        .unwrap();
    let cache = DiskCache::open(directory.path(), JsonCodec::<f64>::new()).unwrap();
    assert_eq!(
        cache
            .increment_cache("counter", 1.5, ExactCacheContext::default())
            .unwrap(),
        3.5
    );
    assert_eq!(
        cache
            .increment_cache("invalid", 2.0, ExactCacheContext::default())
            .unwrap(),
        2.0
    );
    assert_eq!(
        cache
            .increment_cache("pickle-counter", 1.0, ExactCacheContext::default())
            .unwrap(),
        3.0
    );
    let connection = Connection::open(directory.path().join("cache.db")).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT typeof(value) FROM Cache WHERE key = 'counter'",
                [],
                |row| row.get::<_, String>(0)
            )
            .unwrap(),
        "real"
    );
}

#[test]
fn counters_are_atomic_across_concurrent_callers() {
    let directory = tempfile::tempdir().unwrap();
    let cache = Arc::new(DiskCache::open(directory.path(), JsonCodec::<f64>::new()).unwrap());
    let workers = (0..8)
        .map(|_| {
            let cache = Arc::clone(&cache);
            thread::spawn(move || {
                for _ in 0..25 {
                    cache
                        .increment_cache("counter", 1.0, ExactCacheContext::default())
                        .unwrap();
                }
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(
        cache
            .increment_cache("counter", 0.0, ExactCacheContext::default())
            .unwrap(),
        200.0
    );
}

#[test]
fn delete_flush_and_spilled_file_replacement_clean_up_storage() {
    let (directory, store) = store();
    let large = vec![b'x'; 32 * 1024];
    store
        .set("large", StoredValue::Bytes(large.clone()), None, 0.0)
        .unwrap();
    assert_eq!(value_files(directory.path()).len(), 1);
    store
        .set(
            "large",
            StoredValue::Bytes(vec![b'y'; 32 * 1024]),
            None,
            0.0,
        )
        .unwrap();
    assert_eq!(value_files(directory.path()).len(), 1);
    store.pop("large", 0.0).unwrap();
    assert!(value_files(directory.path()).is_empty());
    store
        .set("a", StoredValue::Bytes(large.clone()), None, 0.0)
        .unwrap();
    store
        .set("b", StoredValue::Bytes(large), None, 0.0)
        .unwrap();
    store.clear().unwrap();
    assert!(value_files(directory.path()).is_empty());
}

#[tokio::test]
async fn async_operations_connection_and_delete_match_sync_operations() {
    let directory = tempfile::tempdir().unwrap();
    let cache = cache(directory.path());
    let context = ExactCacheContext {
        ttl: Some(Duration::from_secs(60)),
    };
    cache
        .async_set_cache("a", json!(1), context.clone())
        .await
        .unwrap();
    cache
        .async_set_cache_pipeline(
            vec![("b".into(), json!(2)), ("c".into(), json!(3))],
            context.clone(),
        )
        .await
        .unwrap();
    assert_eq!(
        cache.async_get_cache("a", &context).await.unwrap(),
        Some(json!(1))
    );
    assert_eq!(
        cache
            .async_batch_get_cache(vec!["c".into(), "missing".into()], context.clone())
            .await
            .unwrap(),
        vec![BatchEntry::Hit(json!(3)), BatchEntry::Miss]
    );
    cache.async_delete_cache("a").await.unwrap();
    cache.async_flush_cache().await.unwrap();
    assert_eq!(
        cache.test_connection().await.unwrap().status,
        litellm_cache::CacheConnectionStatus::Success
    );
}
