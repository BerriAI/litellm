use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    thread,
    time::Duration,
};

use litellm_cache::{
    BaseCache, BatchCache, BatchEntry, CacheCodec, CounterCache, DeleteCache, DisconnectCache,
    ExactCacheContext, FlushCache, JsonCodec,
};
use litellm_cache_disk::{DiskCache, DiskStore, DiskcacheSqliteStore, StoredValue, ValueAdapter};
use rstest::{fixture, rstest};
use rusqlite::Connection;
use serde_json::{Value, json};
use tempfile::TempDir;

struct Sandbox {
    directory: TempDir,
}

#[fixture]
fn sandbox() -> Sandbox {
    Sandbox {
        directory: tempfile::tempdir().unwrap(),
    }
}

impl Sandbox {
    fn store(&self) -> DiskcacheSqliteStore {
        DiskcacheSqliteStore::open(self.directory.path()).unwrap()
    }

    fn cache<V>(&self) -> DiskCache<JsonCodec<V>>
    where
        JsonCodec<V>: CacheCodec,
    {
        DiskCache::open(self.directory.path(), JsonCodec::new()).unwrap()
    }

    fn db(&self) -> Connection {
        Connection::open(self.directory.path().join("cache.db")).unwrap()
    }

    fn value_files(&self) -> Vec<PathBuf> {
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
        visit(self.directory.path(), &mut files);
        files
    }
}

#[rstest]
fn relative_store_directory_is_absolutized(sandbox: Sandbox) {
    let relative = PathBuf::from(format!(
        ".litellm-cache-disk-{}",
        sandbox
            .directory
            .path()
            .file_name()
            .unwrap()
            .to_string_lossy()
    ));
    let store = DiskcacheSqliteStore::open(&relative).unwrap();
    assert!(store.directory().is_absolute());
    assert!(store.directory().ends_with(&relative));
    let directory = store.directory().to_path_buf();
    drop(store);
    fs::remove_dir_all(directory).unwrap();
}

#[derive(Clone, Copy, Debug, Default)]
struct TextAdapter;

impl ValueAdapter for TextAdapter {
    fn read(&self, value: StoredValue) -> Result<Option<Vec<u8>>, litellm_cache::Error> {
        match value {
            StoredValue::Text(value) => Ok(Some(value.into_bytes())),
            _ => Ok(None),
        }
    }

    fn write(&self, payload: Vec<u8>) -> StoredValue {
        StoredValue::Text(String::from_utf8(payload).unwrap())
    }

    fn counter_seed(&self, _: Option<StoredValue>) -> Result<f64, litellm_cache::Error> {
        Ok(0.0)
    }

    fn counter_value(&self, value: f64) -> StoredValue {
        if value.fract() == 0.0 {
            StoredValue::Integer(value as i64)
        } else {
            StoredValue::Float(value)
        }
    }
}

#[rstest]
fn roundtrip_persists_and_reopens(sandbox: Sandbox) {
    let context = ExactCacheContext::default();
    let opened = sandbox.cache::<Value>();
    opened
        .set_cache("key", json!({"answer": 42}), &context)
        .unwrap();
    assert_eq!(
        opened.get_cache("key", &context).unwrap(),
        Some(json!({"answer": 42}))
    );
    drop(opened);
    let reopened = sandbox.cache::<Value>();
    assert_eq!(
        reopened.get_cache("key", &context).unwrap(),
        Some(json!({"answer": 42}))
    );
}

#[rstest]
fn ttl_and_expired_culling_match_cache_contract(sandbox: Sandbox) {
    let store = sandbox.store();
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
    assert_eq!(
        sandbox
            .db()
            .query_row("SELECT COUNT(*) FROM Cache", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        sandbox
            .db()
            .query_row(
                "SELECT value FROM Settings WHERE key = 'count'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        1
    );
}

#[rstest]
fn batch_preserves_order_and_classifies_misses_and_invalid_values(sandbox: Sandbox) {
    let store = sandbox.store();
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
    let entries = sandbox
        .cache::<Value>()
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

#[rstest]
#[case(StoredValue::Bytes(Vec::new()))]
#[case(StoredValue::Text(String::new()))]
#[case(StoredValue::Integer(0))]
#[case(StoredValue::Float(0.0))]
#[case(StoredValue::Pickle(vec![0x80, 0x05, 0x4e, 0x2e]))]
#[case(StoredValue::Pickle(vec![0x80, 0x05, 0x89, 0x2e]))]
#[case(StoredValue::Pickle(vec![0x80, 0x05, 0x4b, 0x00, 0x2e]))]
#[case(StoredValue::Pickle(vec![0x80, 0x05, 0x95, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2e]))]
#[case(StoredValue::Pickle(vec![0x80, 0x05, 0x95, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x8c, 0x00, 0x94, 0x2e]))]
#[case(StoredValue::Pickle(vec![0x80, 0x05, 0x5d, 0x94, 0x2e]))]
#[case(StoredValue::Pickle(vec![0x80, 0x05, 0x7d, 0x94, 0x2e]))]
#[case(StoredValue::Pickle(vec![0x80, 0x05, 0x29, 0x2e]))]
fn falsy_values_are_misses(sandbox: Sandbox, #[case] value: StoredValue) {
    sandbox.store().set("key", value, None, 0.0).unwrap();
    assert_eq!(
        sandbox
            .cache::<Value>()
            .get_cache("key", &ExactCacheContext::default())
            .unwrap(),
        None
    );
}

#[rstest]
#[case(Some(StoredValue::Integer(2)), 1.5, 3.5, "real")]
#[case(Some(StoredValue::Integer(2)), 1.0, 3.0, "integer")]
#[case(Some(StoredValue::Float(3.5)), 1.0, 1.0, "integer")]
#[case(Some(StoredValue::Text("not a number".into())), 2.0, 2.0, "integer")]
#[case(Some(StoredValue::Text("5".into())), 2.0, 7.0, "integer")]
#[case(Some(StoredValue::Text("3.5".into())), 2.0, 2.0, "integer")]
#[case(Some(StoredValue::Pickle(vec![0x80, 0x05, 0x88, 0x2e])), 1.0, 2.0, "integer")]
#[case(Some(StoredValue::Pickle(vec![0x80, 0x05, 0x4b, 0x02, 0x2e])), 1.0, 3.0, "integer")]
#[case(Some(StoredValue::Pickle(vec![0x80, 0x05, 0x95, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7d, 0x94, 0x8c, 0x01, 0x61, 0x94, 0x4b, 0x01, 0x73, 0x2e])), 1.0, 1.0, "integer")]
#[case(Some(StoredValue::Pickle(vec![0x80, 0x05, 0x4e, 0x2e])), 4.0, 4.0, "integer")]
fn counters_follow_python_initialization(
    sandbox: Sandbox,
    #[case] initial: Option<StoredValue>,
    #[case] amount: f64,
    #[case] expected: f64,
    #[case] sqlite_type: &str,
) {
    if let Some(initial) = initial {
        sandbox.store().set("counter", initial, None, 0.0).unwrap();
    }
    let cache = sandbox.cache::<f64>();
    assert_eq!(
        cache
            .increment_cache("counter", amount, ExactCacheContext::default())
            .unwrap(),
        expected
    );
    assert_eq!(
        sandbox
            .db()
            .query_row(
                "SELECT typeof(value) FROM Cache WHERE key = 'counter'",
                [],
                |row| row.get::<_, String>(0)
            )
            .unwrap(),
        sqlite_type
    );
}

#[rstest]
fn counters_are_atomic_across_concurrent_callers(sandbox: Sandbox) {
    let cache = Arc::new(sandbox.cache::<f64>());
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

#[rstest]
fn fractional_then_integer_increment_follows_python_behavior(sandbox: Sandbox) {
    let cache = sandbox.cache::<f64>();
    assert_eq!(
        cache
            .increment_cache("counter", 3.5, ExactCacheContext::default())
            .unwrap(),
        3.5
    );
    assert_eq!(
        cache
            .increment_cache("counter", 1.0, ExactCacheContext::default())
            .unwrap(),
        1.0
    );
}

#[rstest]
fn increment_ttl_replacement_clears_expiry_without_ttl(sandbox: Sandbox) {
    let cache = sandbox.cache::<f64>();
    cache
        .increment_cache(
            "counter",
            1.0,
            ExactCacheContext {
                ttl: Some(Duration::from_secs(60)),
            },
        )
        .unwrap();
    assert!(
        sandbox
            .db()
            .query_row(
                "SELECT expire_time IS NOT NULL FROM Cache WHERE key = 'counter'",
                [],
                |row| row.get::<_, bool>(0)
            )
            .unwrap()
    );
    cache
        .increment_cache("counter", 1.0, ExactCacheContext::default())
        .unwrap();
    assert!(
        !sandbox
            .db()
            .query_row(
                "SELECT expire_time IS NOT NULL FROM Cache WHERE key = 'counter'",
                [],
                |row| row.get::<_, bool>(0)
            )
            .unwrap()
    );
}

#[rstest]
fn custom_adapter_controls_storage_and_reads(sandbox: Sandbox) {
    let cache = DiskCache::with_adapter(sandbox.store(), TextAdapter, JsonCodec::<Value>::new());
    cache
        .set_cache("key", json!({"answer": 42}), &ExactCacheContext::default())
        .unwrap();
    assert!(matches!(
        sandbox.store().get("key", 0.0).unwrap(),
        Some(StoredValue::Text(_))
    ));
    assert_eq!(
        cache
            .get_cache("key", &ExactCacheContext::default())
            .unwrap(),
        Some(json!({"answer": 42}))
    );
}

#[rstest]
fn delete_flush_and_spilled_file_replacement_clean_up_storage(sandbox: Sandbox) {
    let large = vec![b'x'; 32 * 1024];
    sandbox
        .store()
        .set("large", StoredValue::Bytes(large.clone()), None, 0.0)
        .unwrap();
    assert_eq!(sandbox.value_files().len(), 1);
    sandbox
        .store()
        .set(
            "large",
            StoredValue::Bytes(vec![b'y'; 32 * 1024]),
            None,
            0.0,
        )
        .unwrap();
    assert_eq!(sandbox.value_files().len(), 1);
    sandbox.store().pop("large", 0.0).unwrap();
    assert!(sandbox.value_files().is_empty());
    sandbox
        .store()
        .set("a", StoredValue::Bytes(large.clone()), None, 0.0)
        .unwrap();
    sandbox
        .store()
        .set("b", StoredValue::Bytes(large), None, 0.0)
        .unwrap();
    sandbox.store().clear().unwrap();
    assert!(sandbox.value_files().is_empty());
}

#[rstest]
#[tokio::test]
async fn async_operations_disconnect_and_delete_match_sync_operations(sandbox: Sandbox) {
    let cache = sandbox.cache::<Value>();
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
    cache.disconnect().await.unwrap();
}

#[derive(Clone, Copy, Debug)]
enum Increment {
    Sync,
    Async { refresh_ttl: bool },
}

impl Increment {
    async fn apply(
        self,
        cache: &DiskCache<JsonCodec<Value>>,
        key: &str,
        amount: f64,
        context: ExactCacheContext,
    ) -> f64 {
        match self {
            Self::Sync => cache.increment_cache(key, amount, context).unwrap(),
            Self::Async { refresh_ttl } => cache
                .async_increment(key, amount, context, refresh_ttl)
                .await
                .unwrap(),
        }
    }
}

#[rstest]
#[case::sync_missing(Increment::Sync, None, 3.0, 3.0)]
#[case::sync_existing_int(Increment::Sync, Some(json!(7)), 5.0, 12.0)]
#[case::sync_non_int(Increment::Sync, Some(json!("not-a-number")), 4.0, 4.0)]
#[case::async_missing(Increment::Async { refresh_ttl: false }, None, 2.0, 2.0)]
#[case::async_existing_int(Increment::Async { refresh_ttl: false }, Some(json!(10)), 5.0, 15.0)]
#[case::async_non_int(Increment::Async { refresh_ttl: false }, Some(json!("corrupt")), 9.0, 9.0)]
#[case::async_refresh_ttl_is_ignored(Increment::Async { refresh_ttl: true }, Some(json!(1)), 1.0, 2.0)]
#[tokio::test]
async fn increments_read_back_through_get_cache(
    sandbox: Sandbox,
    #[case] increment: Increment,
    #[case] initial: Option<Value>,
    #[case] amount: f64,
    #[case] expected: f64,
) {
    let cache = sandbox.cache::<Value>();
    let context = ExactCacheContext::default();
    if let Some(initial) = initial {
        cache
            .async_set_cache("counter", initial, context.clone())
            .await
            .unwrap();
    }
    assert_eq!(
        increment
            .apply(&cache, "counter", amount, context.clone())
            .await,
        expected
    );
    assert_eq!(
        cache.get_cache("counter", &context).unwrap(),
        Some(json!(expected as i64))
    );
}

#[rstest]
#[case::without_refresh(false)]
#[case::with_refresh(true)]
#[tokio::test]
async fn async_increment_rewrites_ttl_on_every_write(sandbox: Sandbox, #[case] refresh_ttl: bool) {
    let cache = sandbox.cache::<Value>();
    let expiry = || {
        sandbox
            .db()
            .query_row(
                "SELECT expire_time IS NOT NULL FROM Cache WHERE key = 'counter'",
                [],
                |row| row.get::<_, bool>(0),
            )
            .unwrap()
    };
    let ttl = ExactCacheContext {
        ttl: Some(Duration::from_secs(60)),
    };
    cache
        .async_increment("counter", 1.0, ttl.clone(), refresh_ttl)
        .await
        .unwrap();
    assert!(expiry());
    cache
        .async_increment("counter", 1.0, ExactCacheContext::default(), refresh_ttl)
        .await
        .unwrap();
    assert!(!expiry());
    cache
        .async_increment("counter", 1.0, ttl, refresh_ttl)
        .await
        .unwrap();
    assert!(expiry());
}
