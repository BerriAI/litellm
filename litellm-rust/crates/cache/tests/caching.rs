use litellm_cache::{
    BaseCache, CacheConnectionResult, CacheControls, CacheEntry, CacheKeyContext, CacheKeyField,
    CacheKeyInput, CacheKwargs, Error, cache_key, get_cache_key,
};
use sha2::{Digest, Sha256};
use std::{sync::Mutex, time::Duration};

struct TestCache {
    default_ttl: Duration,
    writes: Mutex<Vec<(String, CacheEntry, CacheKwargs)>>,
}

impl BaseCache for TestCache {
    type Value = CacheEntry;

    fn default_ttl(&self) -> Duration {
        self.default_ttl
    }

    fn set_cache(&self, _: &str, _: Self::Value, _: CacheKwargs) -> Result<(), Error> {
        Err(Error::Unavailable)
    }

    async fn async_set_cache(
        &self,
        key: &str,
        value: Self::Value,
        kwargs: CacheKwargs,
    ) -> Result<(), Error> {
        if key == "unavailable" {
            return Err(Error::Unavailable);
        }
        self.writes
            .lock()
            .unwrap()
            .push((key.into(), value, kwargs));
        Ok(())
    }

    fn get_cache(&self, _: &str, _: &CacheKwargs) -> Result<Option<Self::Value>, Error> {
        Ok(None)
    }

    fn delete_cache(&self, _: &str) -> Result<(), Error> {
        Ok(())
    }

    fn flush_cache(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }

    async fn test_connection(&self) -> Result<CacheConnectionResult, Error> {
        unreachable!()
    }
}

#[test]
fn ttl_uses_default_and_allows_per_call_override() {
    let cache = TestCache {
        default_ttl: Duration::from_secs(60),
        writes: Mutex::default(),
    };
    assert_eq!(
        cache.get_ttl(&CacheKwargs::default()),
        Duration::from_secs(60)
    );
    assert_eq!(
        cache.get_ttl(&CacheKwargs {
            ttl: Some(Duration::from_secs(5)),
            ..Default::default()
        }),
        Duration::from_secs(5)
    );
}

#[tokio::test]
async fn default_batch_operations_use_async_writes_and_stop_on_failure() {
    let cache = TestCache {
        default_ttl: Duration::from_secs(60),
        writes: Mutex::default(),
    };
    let entry = CacheEntry {
        timestamp: 123.0,
        response: serde_json::json!("cached"),
    };
    let kwargs = CacheKwargs {
        ttl: Some(Duration::from_secs(5)),
        ..Default::default()
    };
    cache
        .batch_cache_write("single", entry.clone(), kwargs.clone())
        .await
        .unwrap();
    assert_eq!(
        cache
            .async_set_cache_pipeline(
                vec![
                    ("first".into(), entry.clone()),
                    ("unavailable".into(), entry.clone()),
                    ("skipped".into(), entry.clone()),
                ],
                kwargs.clone(),
            )
            .await,
        Err(Error::Unavailable)
    );
    assert_eq!(
        *cache.writes.lock().unwrap(),
        vec![
            ("single".into(), entry.clone(), kwargs.clone()),
            ("first".into(), entry, kwargs),
        ]
    );
}

#[test]
fn keys_match_python_order_groups_files_presets_and_namespaces() {
    let mut input = CacheKeyInput {
        fields: vec![
            CacheKeyField {
                name: "model".into(),
                value: Some("deployment".into()),
                api_parameter: true,
                internal_parameter: false,
            },
            CacheKeyField {
                name: "file".into(),
                value: None,
                api_parameter: true,
                internal_parameter: false,
            },
        ],
        namespace: Some("team".into()),
        ..Default::default()
    };
    CacheKeyContext {
        model_group: Some("group".into()),
        caching_groups: vec![(vec!["group".into()], "['group']".into())],
        file_checksum: Some("checksum".into()),
        ..Default::default()
    }
    .apply(&mut input);
    assert_eq!(
        cache_key(&input),
        format!(
            "team:{:x}",
            Sha256::digest(b"model: ['group']file: checksum")
        )
    );
    input.preset = Some("preset".into());
    assert_eq!(get_cache_key(&input), "preset");
}

#[test]
fn cache_controls_honor_default_modes_and_directives() {
    let enabled = CacheControls {
        supported_call_type: true,
        configured: true,
        default_on: true,
        ..Default::default()
    };
    assert!(enabled.reads());
    assert!(enabled.writes());
    assert!(
        !CacheControls {
            default_on: false,
            ..enabled
        }
        .reads()
    );
    assert!(
        CacheControls {
            default_on: false,
            use_cache: true,
            ..enabled
        }
        .reads()
    );
    assert!(
        !CacheControls {
            no_cache: true,
            ..enabled
        }
        .reads()
    );
    assert!(
        !CacheControls {
            no_store: true,
            ..enabled
        }
        .writes()
    );
}
