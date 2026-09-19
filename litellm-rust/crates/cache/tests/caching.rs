use litellm_cache::{
    BaseCache, CacheConnectionResult, CacheControls, CacheEntry, CacheFuture, CacheKeyContext,
    CacheKeyField, CacheKeyInput, CacheKwargs, Error, cache_key, get_cache_key,
};
use sha2::{Digest, Sha256};
use std::time::Duration;

struct TestCache {
    default_ttl: Duration,
}

impl BaseCache for TestCache {
    type Value = CacheEntry;

    fn default_ttl(&self) -> Duration {
        self.default_ttl
    }

    fn set_cache(&self, _: &str, _: Self::Value, _: CacheKwargs) -> Result<(), Error> {
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

    fn disconnect(&self) -> CacheFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }

    fn test_connection(&self) -> CacheFuture<'_, CacheConnectionResult> {
        unreachable!()
    }
}

#[test]
fn ttl_uses_default_and_allows_per_call_override() {
    let cache = TestCache {
        default_ttl: Duration::from_secs(60),
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
