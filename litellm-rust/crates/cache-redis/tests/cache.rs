use std::time::Duration;

use litellm_cache::{
    BaseCache, BatchEntry, CacheCodec, CacheConnectionStatus, CacheKwargs, ClaimCache,
    CounterCache, Error, JsonCodec, get_cache, set_cache,
};
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
                .arg("team\\*:*")
                .arg("COUNT")
                .arg(1000),
            Ok(redis_test::redis_value!(["0", ["team*:key"]])),
        ),
        MockCmd::new(redis::cmd("DEL").arg("team*:key"), Ok(1u32)),
    ])
    .assert_all_commands_consumed();
    let scoped = RedisCache::with_connection(connection, None, JsonCodec::<String>::new())
        .with_namespace(Some("team*".into()));
    scoped.flush_cache().unwrap();
}

#[tokio::test]
async fn connection_failures_use_the_python_result_contract() {
    let error = redis::RedisError::from((redis::ErrorKind::Io, "connection refused"));
    let connection =
        MockRedisConnection::new([MockCmd::new(redis::cmd("PING"), Err::<String, _>(error))])
            .assert_all_commands_consumed();
    let cache = RedisCache::with_connection(connection, None, JsonCodec::<String>::new());

    let result = cache.test_connection().await.unwrap();
    assert_eq!(result.status, CacheConnectionStatus::Failed);
    assert!(result.message.starts_with("Redis connection failed:"));
    assert!(result.error.is_some());
}

#[tokio::test]
async fn batch_reads_keep_order_and_treat_invalid_values_as_invalid_entries() {
    let connection = MockRedisConnection::new([MockCmd::new(
        redis::cmd("MGET").arg("hit").arg("miss").arg("invalid"),
        Ok(vec![
            redis::Value::BulkString(vec![42, 7]),
            redis::Value::Nil,
            redis::Value::BulkString(vec![99, 7]),
        ]),
    )])
    .assert_all_commands_consumed();
    let cache = RedisCache::with_connection(connection, None, TaggedByteCodec(42));

    assert_eq!(
        cache
            .async_get_cache_batch(
                vec!["hit".into(), "miss".into(), "invalid".into()],
                CacheKwargs::default(),
            )
            .await
            .unwrap(),
        vec![BatchEntry::Hit(7), BatchEntry::Miss, BatchEntry::Invalid]
    );
}

#[tokio::test]
async fn async_flush_deletes_each_scan_page_separately() {
    let connection = MockRedisConnection::new([
        MockCmd::new(
            redis::cmd("SCAN")
                .cursor_arg(0)
                .arg("MATCH")
                .arg("team:*")
                .arg("COUNT")
                .arg(1000),
            Ok(redis_test::redis_value!(["7", ["team:a", "team:b"]])),
        ),
        MockCmd::new(redis::cmd("DEL").arg("team:a").arg("team:b"), Ok(2u32)),
        MockCmd::new(
            redis::cmd("SCAN")
                .cursor_arg(7)
                .arg("MATCH")
                .arg("team:*")
                .arg("COUNT")
                .arg(1000),
            Ok(redis_test::redis_value!(["0", ["team:c"]])),
        ),
        MockCmd::new(redis::cmd("DEL").arg("team:c"), Ok(1u32)),
    ])
    .assert_all_commands_consumed();
    let cache = RedisCache::with_connection(connection, None, JsonCodec::<String>::new())
        .with_namespace(Some("team".into()));

    cache.async_flush_cache().await.unwrap();
}

const CLAIM_SCRIPT: &str = concat!(
    "local current = redis.call('GET', KEYS[1]); ",
    "if ARGV[1] == '' then if current ~= false and current ~= '' then return 0; end; ",
    "elseif current ~= ARGV[1] then return 0; end; ",
    "if ARGV[3] ~= '' then redis.call('SET', KEYS[1], ARGV[3], 'EX', ARGV[2]); ",
    "elseif ARGV[4] == '1' then redis.call('EXPIRE', KEYS[1], ARGV[2]); end; return 1"
);

fn claim_eval(expected: &str, write: &str, refresh: bool) -> redis::Cmd {
    let mut cmd = redis::cmd("EVAL");
    cmd.arg(CLAIM_SCRIPT)
        .arg(1)
        .arg("pin")
        .arg(expected)
        .arg(600)
        .arg(write)
        .arg(u8::from(refresh));
    cmd
}

#[tokio::test]
async fn claims_match_eligible_values_written_by_another_encoder() {
    let python_payload = r#"{"model_id": "a", "deployment": "east"}"#;
    let stored = serde_json::json!({"deployment": "east", "model_id": "a"});
    let candidate = serde_json::json!({"model_id": "b"});
    let connection = MockRedisConnection::new([
        MockCmd::new(redis::cmd("GET").arg("pin"), Ok(python_payload)),
        MockCmd::new(claim_eval(python_payload, "", true), Ok(1)),
    ])
    .assert_all_commands_consumed();
    let cache =
        RedisCache::with_connection(connection, None, JsonCodec::<serde_json::Value>::new());

    assert_eq!(
        cache
            .async_claim_cache(
                "pin",
                candidate,
                vec![stored.clone()],
                CacheKwargs::default()
            )
            .await
            .unwrap(),
        stored
    );
}

#[test]
fn claims_retry_when_the_key_changes_and_replace_ineligible_winners() {
    let candidate = serde_json::json!({"model_id": "b"});
    let payload = r#"{"model_id":"b"}"#;
    let connection = MockRedisConnection::new([
        MockCmd::new(redis::cmd("GET").arg("pin"), Ok(redis::Value::Nil)),
        MockCmd::new(claim_eval("", payload, false), Ok(0)),
        MockCmd::new(redis::cmd("GET").arg("pin"), Ok(r#"{"model_id":"gone"}"#)),
        MockCmd::new(claim_eval(r#"{"model_id":"gone"}"#, payload, false), Ok(1)),
    ])
    .assert_all_commands_consumed();
    let cache =
        RedisCache::with_connection(connection, None, JsonCodec::<serde_json::Value>::new());

    assert_eq!(
        cache
            .claim_cache(
                "pin",
                candidate.clone(),
                &[serde_json::json!({"model_id": "a"})],
                CacheKwargs::default()
            )
            .unwrap(),
        candidate
    );
}

#[test]
fn claims_without_eligible_values_keep_the_winner_without_refreshing_its_ttl() {
    let stored = r#"{"model_id": "a"}"#;
    let connection = MockRedisConnection::new([
        MockCmd::new(redis::cmd("GET").arg("pin"), Ok(stored)),
        MockCmd::new(claim_eval(stored, "", false), Ok(1)),
    ])
    .assert_all_commands_consumed();
    let cache =
        RedisCache::with_connection(connection, None, JsonCodec::<serde_json::Value>::new());

    assert_eq!(
        cache
            .claim_cache(
                "pin",
                serde_json::json!({"model_id": "b"}),
                &[],
                CacheKwargs::default()
            )
            .unwrap(),
        serde_json::json!({"model_id": "a"})
    );
}

#[tokio::test]
async fn async_increment_runs_the_atomic_script() {
    let mut eval = redis::cmd("EVAL");
    eval.arg(concat!(
        "local value = redis.call('INCRBYFLOAT', KEYS[1], ARGV[1]); ",
        "if redis.call('TTL', KEYS[1]) == -1 then ",
        "redis.call('EXPIRE', KEYS[1], ARGV[2]); end; return value"
    ))
    .arg(1)
    .arg("counter")
    .arg(2.5f64)
    .arg(600);
    let connection =
        MockRedisConnection::new([MockCmd::new(eval, Ok("4.5"))]).assert_all_commands_consumed();
    let cache = RedisCache::with_connection(connection, None, JsonCodec::<f64>::new());

    assert_eq!(
        cache
            .async_increment_cache("counter", 2.5, CacheKwargs::default())
            .await
            .unwrap(),
        4.5
    );
}
