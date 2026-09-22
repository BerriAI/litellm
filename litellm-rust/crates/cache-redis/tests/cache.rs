use std::time::Duration;

use litellm_cache::{
    BaseCache, BatchCache, BatchEntry, CacheCodec, CacheConnectionStatus, CacheScript, ClaimCache,
    CounterCache, DeleteCache, Error, ExactCacheContext, FlushCache, IncrementOperation, JsonCodec,
    ScriptCache, get_cache, set_cache,
};
use litellm_cache_redis::{
    RedisArg, RedisCache, RedisLpopOperation, RedisLpopResult, RedisRpushOperation,
};
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
    let context = ExactCacheContext {
        ttl: Some(Duration::from_millis(1500)),
    };
    set_cache(&cache, "counter", 7, &context).unwrap();
    assert_eq!(get_cache(&cache, "counter", &context).unwrap(), Some(7));
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
    let context = ExactCacheContext::default();
    cache
        .batch_cache_write("counter", 7, context.clone())
        .await
        .unwrap();
    assert_eq!(
        cache.async_get_cache("counter", &context).await.unwrap(),
        Some(7)
    );
    cache
        .async_set_cache_pipeline(
            vec![("batch".into(), 8)],
            ExactCacheContext {
                ttl: Some(Duration::from_millis(1500)),
            },
        )
        .await
        .unwrap();
    cache.async_delete_cache("counter").await.unwrap();
    assert_eq!(
        cache.async_get_cache("counter", &context).await.unwrap(),
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
    let context = ExactCacheContext::default();
    assert_eq!(
        cache.set_cache("invalid", 255, &context),
        Err(Error::InvalidEntry)
    );
    assert_eq!(
        cache.async_set_cache("invalid", 255, context.clone()).await,
        Err(Error::InvalidEntry)
    );
    assert_eq!(
        cache
            .async_set_cache_pipeline(
                vec![("valid".into(), 7), ("invalid".into(), 255)],
                context.clone(),
            )
            .await,
        Err(Error::InvalidEntry)
    );
    assert_eq!(
        cache.get_cache("invalid", &context),
        Err(Error::InvalidEntry)
    );
    assert_eq!(
        cache.async_get_cache("invalid", &context).await,
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
        cache
            .get_cache("key", &ExactCacheContext::default())
            .unwrap(),
        None
    );
    assert_eq!(
        cache
            .get_cache("team:key", &ExactCacheContext::default())
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
            .async_batch_get_cache(
                vec!["hit".into(), "miss".into(), "invalid".into()],
                ExactCacheContext::default(),
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

#[tokio::test]
async fn direct_redis_operations_preserve_namespace_values_and_missing_ttls() {
    let mut sadd_pipeline = redis::pipe();
    sadd_pipeline
        .cmd("SADD")
        .arg("team:members")
        .arg("a")
        .arg("b")
        .cmd("EXPIRE")
        .arg("team:members")
        .arg(600u64)
        .ignore();
    let connection = MockRedisConnection::new([
        MockCmd::new(
            redis::cmd("MGET").arg("team:count").arg("team:missing"),
            Ok(redis_test::redis_value!(["7", nil])),
        ),
        MockCmd::new(
            redis::cmd("MGET").arg("team:count").arg("team:missing"),
            Ok(redis_test::redis_value!(["7", nil])),
        ),
        MockCmd::new(redis::cmd("PING"), Ok("PONG")),
        MockCmd::new(redis::cmd("PING"), Ok("PONG")),
        MockCmd::new(redis::cmd("TTL").arg("team:missing"), Ok(-2i64)),
        MockCmd::new(
            redis::cmd("SCAN")
                .cursor_arg(0)
                .arg("MATCH")
                .arg("team:job-*")
                .arg("COUNT")
                .arg(25),
            Ok(redis_test::redis_value!(["4", ["team:job-a"]])),
        ),
        MockCmd::new(
            redis::cmd("SCAN")
                .cursor_arg(4)
                .arg("MATCH")
                .arg("team:job-*")
                .arg("COUNT")
                .arg(25),
            Ok(redis_test::redis_value!(["0", ["team:job-b"]])),
        ),
        MockCmd::new(
            redis::cmd("DEL").arg("team:job-a").arg("team:job-b"),
            Ok(2u32),
        ),
        MockCmd::with_values(
            sadd_pipeline,
            Ok(vec![redis::Value::Int(2), redis::Value::Int(1)]),
        ),
        MockCmd::new(
            redis::cmd("RPUSH").arg("team:queue").arg("a").arg("b"),
            Ok(2u32),
        ),
        MockCmd::new(
            redis::cmd("LPOP").arg("team:queue").arg(2usize),
            Ok(redis_test::redis_value!(["a", "b"])),
        ),
        MockCmd::new(
            redis::cmd("EVAL")
                .arg("return KEYS[1]")
                .arg(1usize)
                .arg("team:key"),
            Ok("team:key"),
        ),
        MockCmd::new(
            redis::cmd("EVAL")
                .arg("return KEYS[1]")
                .arg(1usize)
                .arg("team:key"),
            Ok("team:key"),
        ),
        MockCmd::new(redis::cmd("CLIENT").arg("LIST"), Ok("id=1")),
        MockCmd::new(redis::cmd("INFO"), Ok("redis_version:7")),
        MockCmd::new(redis::cmd("FLUSHALL"), Ok("OK")),
    ])
    .assert_all_commands_consumed();
    let cache = RedisCache::with_connection(connection, None, JsonCodec::<String>::new())
        .with_namespace(Some("team".into()));

    assert_eq!(
        cache
            .batch_get_counts(&["count".into(), "missing".into()])
            .unwrap(),
        [Some(7), None]
    );
    assert_eq!(
        cache
            .async_batch_get_counts(vec!["count".into(), "missing".into()])
            .await
            .unwrap(),
        [Some(7), None]
    );
    assert!(cache.sync_ping().unwrap());
    assert!(cache.ping().await.unwrap());
    assert_eq!(cache.async_get_ttl("missing").await.unwrap(), None);
    assert_eq!(
        cache.async_scan_iter("job-", 25).await.unwrap(),
        ["team:job-a", "team:job-b"]
    );
    assert_eq!(
        cache
            .delete_cache_keys(vec!["job-a".into(), "job-b".into()])
            .await
            .unwrap(),
        2
    );
    assert_eq!(
        cache
            .async_set_cache_sadd("members", vec!["a".into(), "b".into()], None)
            .await
            .unwrap(),
        2
    );
    assert_eq!(
        cache
            .async_rpush("queue", vec!["a".into(), "b".into()])
            .await
            .unwrap(),
        2
    );
    assert_eq!(
        cache.async_lpop("queue", Some(2)).await.unwrap(),
        RedisLpopResult::Values(vec![b"a".to_vec(), b"b".to_vec()])
    );
    assert_eq!(
        cache
            .async_eval("return KEYS[1]".into(), vec!["key".into()], Vec::new())
            .await
            .unwrap(),
        redis::Value::BulkString(b"team:key".to_vec())
    );
    assert_eq!(
        cache
            .async_register_script("return KEYS[1]".into())
            .invoke(vec!["key".into()], Vec::new())
            .await
            .unwrap(),
        redis::Value::BulkString(b"team:key".to_vec())
    );
    assert_eq!(cache.client_list().unwrap(), "id=1");
    assert_eq!(cache.info().unwrap(), "redis_version:7");
    cache.flushall().unwrap();
}

#[tokio::test]
async fn direct_redis_pipelines_preserve_operation_order() {
    let mut rpush_pipeline = redis::pipe();
    rpush_pipeline
        .cmd("RPUSH")
        .arg("team:a")
        .arg("one")
        .cmd("RPUSH")
        .arg("team:b")
        .arg("two");
    let mut lpop_pipeline = redis::pipe();
    lpop_pipeline
        .cmd("LPOP")
        .arg("team:a")
        .arg(2usize)
        .cmd("LPOP")
        .arg("team:b");
    let connection = MockRedisConnection::new([
        MockCmd::with_values(
            rpush_pipeline,
            Ok(vec![redis::Value::Int(1), redis::Value::Int(2)]),
        ),
        MockCmd::with_values(
            lpop_pipeline,
            Ok(vec![redis_test::redis_value!(["one"]), redis::Value::Nil]),
        ),
    ])
    .assert_all_commands_consumed();
    let queue = RedisCache::with_connection(connection, None, JsonCodec::<String>::new())
        .with_namespace(Some("team".into()));

    assert_eq!(
        queue
            .async_rpush_pipeline(vec![
                RedisRpushOperation {
                    key: "a".into(),
                    values: vec![RedisArg::from("one")],
                },
                RedisRpushOperation {
                    key: "b".into(),
                    values: vec![RedisArg::from("two")],
                },
            ])
            .await
            .unwrap(),
        [1, 2]
    );
    assert_eq!(
        queue
            .async_lpop_pipeline(vec![
                RedisLpopOperation {
                    key: "a".into(),
                    count: Some(2),
                },
                RedisLpopOperation {
                    key: "b".into(),
                    count: None,
                },
            ])
            .await
            .unwrap(),
        [
            RedisLpopResult::Values(vec![b"one".to_vec()]),
            RedisLpopResult::Missing,
        ]
    );

    let mut increment_pipeline = redis::pipe();
    increment_pipeline
        .cmd("INCRBYFLOAT")
        .arg("team:counter")
        .arg(1.5f64)
        .cmd("EXPIRE")
        .arg("team:counter")
        .arg(10u64)
        .ignore()
        .cmd("INCRBYFLOAT")
        .arg("team:counter")
        .arg(2.0f64);
    let connection = MockRedisConnection::new([MockCmd::with_values(
        increment_pipeline,
        Ok(vec![
            redis::Value::BulkString(b"1.5".to_vec()),
            redis::Value::Int(1),
            redis::Value::BulkString(b"3.5".to_vec()),
        ]),
    )])
    .assert_all_commands_consumed();
    let counters = RedisCache::with_connection(connection, None, JsonCodec::<f64>::new())
        .with_namespace(Some("team".into()));
    assert_eq!(
        counters
            .async_increment_pipeline(vec![
                IncrementOperation {
                    key: "counter".into(),
                    amount: 1.5,
                    ttl: Some(Duration::from_secs(10)),
                },
                IncrementOperation {
                    key: "counter".into(),
                    amount: 2.0,
                    ttl: None,
                },
            ])
            .await
            .unwrap(),
        [1.5, 3.5]
    );
}

const INCREMENT_WITH_FLOOR_SCRIPT: &str = concat!(
    "local count = redis.call('INCRBY', KEYS[1], ARGV[1]); ",
    "if count < 0 then count = redis.call('INCRBY', KEYS[1], -count); end; ",
    "if redis.call('TTL', KEYS[1]) < 0 then redis.call('EXPIRE', KEYS[1], ARGV[2]); end; ",
    "return count"
);
const SET_MAX_SCRIPT: &str = concat!(
    "local current = redis.call('GET', KEYS[1]); ",
    "if current == false or tonumber(current) < tonumber(ARGV[1]) then ",
    "redis.call('SET', KEYS[1], ARGV[1]); ",
    "if tonumber(ARGV[2]) > 0 then redis.call('EXPIRE', KEYS[1], ARGV[2]); end; ",
    "return ARGV[1]; end; return current"
);

#[tokio::test]
async fn counter_repairs_are_atomic_and_use_default_ttl() {
    let floor = || {
        redis::cmd("EVAL")
            .arg(INCREMENT_WITH_FLOOR_SCRIPT)
            .arg(1)
            .arg("team:counter")
            .arg(-2i64)
            .arg(30u64)
            .clone()
    };
    let connection = MockRedisConnection::new([
        MockCmd::new(floor(), Ok(0i64)),
        MockCmd::new(floor(), Ok(0i64)),
        MockCmd::new(
            redis::cmd("EVAL")
                .arg(SET_MAX_SCRIPT)
                .arg(1)
                .arg("team:counter")
                .arg(4.5f64)
                .arg(600u64),
            Ok("4.5"),
        ),
    ])
    .assert_all_commands_consumed();
    let cache = RedisCache::with_connection(connection, None, JsonCodec::<f64>::new())
        .with_namespace(Some("team".into()));

    assert_eq!(
        cache
            .increment_with_floor("counter", -2, Duration::from_secs(30))
            .unwrap(),
        0
    );
    assert_eq!(
        cache
            .async_increment_with_floor("counter", -2, Duration::from_secs(30))
            .await
            .unwrap(),
        0
    );
    assert_eq!(
        cache.async_set_max("counter", 4.5, None).await.unwrap(),
        4.5
    );
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
                ExactCacheContext::default()
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
                ExactCacheContext::default()
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
                ExactCacheContext::default()
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
            .async_increment("counter", 2.5, ExactCacheContext::default())
            .await
            .unwrap(),
        4.5
    );
}
