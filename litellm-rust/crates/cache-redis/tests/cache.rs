use std::time::Duration;

use litellm_cache::{BaseCache, CacheCodec, CacheKwargs, Error, JsonCodec, get_cache, set_cache};
use litellm_cache_redis::RedisCache;
use redis_test::{MockCmd, MockRedisConnection};

struct TaggedByteCodec(u8);

impl CacheCodec for TaggedByteCodec {
    type Value = u8;

    fn encode(&self, value: &u8) -> Result<Vec<u8>, Error> {
        if *value > 127 {
            return Err(Error::InvalidEntry);
        }
        Ok(vec![self.0, *value])
    }

    fn decode(&self, bytes: &[u8]) -> Result<u8, Error> {
        match bytes {
            [tag, value] if *tag == self.0 => Ok(*value),
            _ => Err(Error::InvalidEntry),
        }
    }
}

#[test]
fn constructor_rejects_invalid_urls() {
    assert!(RedisCache::new("not a redis url", None, JsonCodec::<String>::new()).is_err());
}

#[test]
fn generic_helpers_use_the_injected_codec_and_ttl() {
    let connection = MockRedisConnection::new([
        MockCmd::new(
            redis::cmd("SETEX")
                .arg("counter")
                .arg(2)
                .arg([42u8, 7].as_slice()),
            Ok("OK"),
        ),
        MockCmd::new(redis::cmd("GET").arg("counter"), Ok(vec![42u8, 7])),
    ])
    .assert_all_commands_consumed();
    let cache = RedisCache::with_connection(connection, None, TaggedByteCodec(42));
    let kwargs = CacheKwargs {
        ttl: Some(Duration::from_millis(1500)),
        ..Default::default()
    };
    set_cache(&cache, "counter", 7, kwargs.clone()).unwrap();
    assert_eq!(get_cache(&cache, "counter", &kwargs).unwrap(), Some(7));
}

#[tokio::test]
async fn async_operations_preserve_codec_ttl_and_missing_values() {
    let connection = MockRedisConnection::new([
        MockCmd::new(
            redis::cmd("SETEX")
                .arg("counter")
                .arg(9)
                .arg([42u8, 7].as_slice()),
            Ok("OK"),
        ),
        MockCmd::new(redis::cmd("GET").arg("counter"), Ok(vec![42u8, 7])),
        MockCmd::new(
            redis::cmd("SETEX")
                .arg("batch")
                .arg(2)
                .arg([42u8, 8].as_slice()),
            Ok("OK"),
        ),
        MockCmd::new(redis::cmd("DEL").arg("counter"), Ok(1u32)),
        MockCmd::new(redis::cmd("GET").arg("counter"), Ok(redis::Value::Nil)),
    ])
    .assert_all_commands_consumed();
    let cache = RedisCache::with_connection(
        connection,
        Some(Duration::from_secs(9)),
        TaggedByteCodec(42),
    );
    let kwargs = CacheKwargs::default();
    cache
        .batch_cache_write("counter", 7, kwargs.clone())
        .await
        .unwrap();
    assert_eq!(
        cache.async_get_cache("counter", &kwargs).await.unwrap(),
        Some(7)
    );
    cache
        .async_set_cache_pipeline(
            vec![("batch".into(), 8)],
            CacheKwargs {
                ttl: Some(Duration::from_millis(1500)),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    cache.async_delete_cache("counter").await.unwrap();
    assert_eq!(
        cache.async_get_cache("counter", &kwargs).await.unwrap(),
        None
    );
}

#[tokio::test]
async fn codec_errors_propagate_without_writing_partial_batches() {
    let connection = MockRedisConnection::new([
        MockCmd::new(redis::cmd("GET").arg("invalid"), Ok(vec![99u8, 7])),
        MockCmd::new(redis::cmd("GET").arg("invalid"), Ok(vec![99u8, 7])),
    ])
    .assert_all_commands_consumed();
    let cache = RedisCache::with_connection(connection, None, TaggedByteCodec(42));
    let kwargs = CacheKwargs::default();
    assert_eq!(
        cache.set_cache("invalid", 255, kwargs.clone()),
        Err(Error::InvalidEntry)
    );
    assert_eq!(
        cache.async_set_cache("invalid", 255, kwargs.clone()).await,
        Err(Error::InvalidEntry)
    );
    assert_eq!(
        cache
            .async_set_cache_pipeline(
                vec![("valid".into(), 7), ("invalid".into(), 255)],
                kwargs.clone(),
            )
            .await,
        Err(Error::InvalidEntry)
    );
    assert_eq!(
        cache.get_cache("invalid", &kwargs),
        Err(Error::InvalidEntry)
    );
    assert_eq!(
        cache.async_get_cache("invalid", &kwargs).await,
        Err(Error::InvalidEntry)
    );
}

#[test]
fn namespaces_are_optional_and_existing_prefixes_are_not_duplicated() {
    let connection = MockRedisConnection::new([
        MockCmd::new(redis::cmd("GET").arg("team:key"), Ok(redis::Value::Nil)),
        MockCmd::new(redis::cmd("GET").arg("team:key"), Ok(redis::Value::Nil)),
    ])
    .assert_all_commands_consumed();
    let cache = RedisCache::with_connection(connection, None, JsonCodec::<String>::new())
        .with_namespace(Some("team".into()));
    assert_eq!(
        cache.get_cache("key", &CacheKwargs::default()).unwrap(),
        None
    );
    assert_eq!(
        cache
            .get_cache("team:key", &CacheKwargs::default())
            .unwrap(),
        None
    );
}

#[test]
fn flush_requires_a_namespace_and_escapes_glob_metacharacters() {
    let unscoped = RedisCache::with_connection(
        MockRedisConnection::new([]).assert_all_commands_consumed(),
        None,
        JsonCodec::<String>::new(),
    );
    assert_eq!(unscoped.flush_cache(), Err(Error::UnscopedFlush));
    let connection = MockRedisConnection::new([
        MockCmd::new(
            redis::cmd("SCAN")
                .cursor_arg(0)
                .arg("MATCH")
                .arg("team\\*:*"),
            Ok(redis_test::redis_value!(["0", ["team*:key"]])),
        ),
        MockCmd::new(redis::cmd("DEL").arg("team*:key"), Ok(1u32)),
    ])
    .assert_all_commands_consumed();
    let scoped = RedisCache::with_connection(connection, None, JsonCodec::<String>::new())
        .with_namespace(Some("team*".into()));
    scoped.flush_cache().unwrap();
}
