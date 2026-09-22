mod support;

use std::time::Duration;

use litellm_cache::{
    BaseCache, BatchCache, BatchEntry, BoundedCounterCache, BulkDeleteCache, CacheCodec,
    CacheConnectionStatus, CacheScript, ClaimCache, ClientInfoCache, ConnectionCache,
    CountReadCache, CounterCache, DeleteCache, DisconnectCache, Error, ExactCacheContext,
    FlushAllCache, FlushCache, IncrementOperation, JsonCodec, PingCache, QueueCache,
    RefreshTtlCache, ScanCache, ScriptCache, SetCache, TtlCache, TtlPipelineCache, get_cache,
    set_cache,
};
use litellm_cache_redis::{
    RedisArg, RedisCache, RedisLpopOperation, RedisLpopResult, RedisRpushOperation,
};
use redis_test::{MockCmd, MockRedisConnection};
use rstest::{fixture, rstest};
use serde_json::json;
use support::TaggedByteCodec;

type Mocked<S> = RedisCache<S, MockRedisConnection>;

fn mock(commands: Vec<MockCmd>) -> MockRedisConnection {
    MockRedisConnection::new(commands).assert_all_commands_consumed()
}

fn tagged(commands: Vec<MockCmd>) -> Mocked<TaggedByteCodec> {
    RedisCache::with_connection(mock(commands), None, TaggedByteCodec(42))
}

fn json_cache(commands: Vec<MockCmd>) -> Mocked<JsonCodec<serde_json::Value>> {
    RedisCache::with_connection(mock(commands), None, JsonCodec::new())
}

fn team(commands: Vec<MockCmd>) -> Mocked<JsonCodec<serde_json::Value>> {
    json_cache(commands).with_namespace(Some("team".into()))
}

#[fixture]
fn context() -> ExactCacheContext {
    ExactCacheContext::default()
}

fn scan(pattern: &str, cursor: u64, count: usize, reply: redis::Value) -> MockCmd {
    MockCmd::new(
        redis::cmd("SCAN")
            .cursor_arg(cursor)
            .arg("MATCH")
            .arg(pattern)
            .arg("COUNT")
            .arg(count),
        Ok(reply),
    )
}

#[rstest]
fn constructor_rejects_invalid_urls() {
    assert!(RedisCache::new("not a redis url", None, JsonCodec::<String>::new()).is_err());
}

#[rstest]
#[case::zero_rounds_up_to_one(Some(Duration::ZERO), 1)]
#[case::fractions_round_up(Some(Duration::from_millis(1500)), 2)]
#[case::whole_seconds_are_kept(Some(Duration::from_secs(15)), 15)]
#[case::missing_ttl_uses_default(None, 600)]
fn writes_round_ttls_up_to_positive_seconds(#[case] ttl: Option<Duration>, #[case] seconds: u64) {
    let cache = tagged(vec![MockCmd::new(
        redis::cmd("SETEX")
            .arg("key")
            .arg(seconds)
            .arg([42u8, 7].as_slice()),
        Ok("OK"),
    )]);
    cache
        .set_cache("key", 7, &ExactCacheContext { ttl })
        .unwrap();
}

#[rstest]
fn generic_helpers_use_the_injected_codec_and_ttl() {
    let cache = tagged(vec![
        MockCmd::new(
            redis::cmd("SETEX")
                .arg("counter")
                .arg(2)
                .arg([42u8, 7].as_slice()),
            Ok("OK"),
        ),
        MockCmd::new(redis::cmd("GET").arg("counter"), Ok(vec![42u8, 7])),
    ]);
    let context = ExactCacheContext {
        ttl: Some(Duration::from_millis(1500)),
    };
    set_cache(&cache, "counter", 7, &context).unwrap();
    assert_eq!(get_cache(&cache, "counter", &context).unwrap(), Some(7));
}

#[rstest]
fn commands_round_trip_entries_and_delete_only_namespaced_keys(context: ExactCacheContext) {
    let value = json!({"deployment": "model-a", "cooldown_seconds": 30});
    let payload = JsonCodec::<serde_json::Value>::new()
        .encode(&value)
        .unwrap();
    let cache = json_cache(vec![
        MockCmd::new(
            redis::cmd("SETEX")
                .arg("litellm-cache:key")
                .arg(600)
                .arg(payload.clone()),
            Ok("OK"),
        ),
        MockCmd::new(redis::cmd("GET").arg("litellm-cache:key"), Ok(payload)),
        MockCmd::new(redis::cmd("DEL").arg("litellm-cache:key"), Ok(1u32)),
    ])
    .with_namespace(Some("litellm-cache".into()));

    cache.set_cache("key", value.clone(), &context).unwrap();
    assert_eq!(cache.get_cache("key", &context).unwrap(), Some(value));
    cache.delete_cache("key").unwrap();
}

#[rstest]
#[tokio::test]
async fn async_operations_preserve_codec_ttl_and_missing_values(context: ExactCacheContext) {
    let cache = RedisCache::with_connection(
        mock(vec![
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
        ]),
        Some(Duration::from_secs(9)),
        TaggedByteCodec(42),
    );
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

#[rstest]
#[tokio::test]
async fn codec_errors_propagate_without_writing_partial_batches(context: ExactCacheContext) {
    let cache = tagged(vec![
        MockCmd::new(redis::cmd("GET").arg("invalid"), Ok(vec![99u8, 7])),
        MockCmd::new(redis::cmd("GET").arg("invalid"), Ok(vec![99u8, 7])),
    ]);
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
        cache
            .async_set_cache_pipeline_with_ttls(vec![
                ("valid".into(), 7, None),
                ("invalid".into(), 255, None),
            ])
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

#[rstest]
#[case::bare_key("key")]
#[case::already_prefixed("team:key")]
fn namespaces_are_added_once(#[case] key: &str, context: ExactCacheContext) {
    let cache = team(vec![MockCmd::new(
        redis::cmd("GET").arg("team:key"),
        Ok(redis::Value::Nil),
    )]);
    assert_eq!(cache.get_cache(key, &context).unwrap(), None);
}

#[rstest]
#[case::empty(Some(String::new()))]
#[case::missing(None)]
fn empty_namespaces_leave_keys_unprefixed(
    #[case] namespace: Option<String>,
    context: ExactCacheContext,
) {
    let cache = json_cache(vec![MockCmd::new(
        redis::cmd("GET").arg("key"),
        Ok(redis::Value::Nil),
    )])
    .with_namespace(namespace);
    assert_eq!(cache.namespace(), None);
    assert_eq!(cache.get_cache("key", &context).unwrap(), None);
}

#[rstest]
#[tokio::test]
async fn flush_requires_a_namespace() {
    let unscoped = json_cache(Vec::new());
    assert_eq!(unscoped.flush_cache(), Err(Error::UnscopedFlush));
    assert_eq!(
        unscoped.async_flush_cache().await,
        Err(Error::UnscopedFlush)
    );
}

#[rstest]
#[case::plain_namespace("litellm-cache", "litellm-cache:*", "litellm-cache:key")]
#[case::glob_metacharacters_are_escaped("team*", "team\\*:*", "team*:key")]
fn flush_scans_and_deletes_only_namespaced_keys(
    #[case] namespace: &str,
    #[case] pattern: &str,
    #[case] key: &str,
) {
    let cache = json_cache(vec![
        scan(pattern, 0, 1000, redis_test::redis_value!(["0", [key]])),
        MockCmd::new(redis::cmd("DEL").arg(key), Ok(1u32)),
    ])
    .with_namespace(Some(namespace.into()));
    cache.flush_cache().unwrap();
}

#[rstest]
#[tokio::test]
async fn async_flush_deletes_each_scan_page_separately() {
    let cache = team(vec![
        scan(
            "team:*",
            0,
            1000,
            redis_test::redis_value!(["7", ["team:a", "team:b"]]),
        ),
        MockCmd::new(redis::cmd("DEL").arg("team:a").arg("team:b"), Ok(2u32)),
        scan(
            "team:*",
            7,
            1000,
            redis_test::redis_value!(["0", ["team:c"]]),
        ),
        MockCmd::new(redis::cmd("DEL").arg("team:c"), Ok(1u32)),
    ]);
    cache.async_flush_cache().await.unwrap();
}

#[rstest]
fn flushall_ignores_the_namespace() {
    team(vec![MockCmd::new(redis::cmd("FLUSHALL"), Ok("OK"))])
        .flushall()
        .unwrap();
}

#[rstest]
#[tokio::test]
async fn batch_reads_keep_order_and_treat_invalid_values_as_invalid_entries(
    context: ExactCacheContext,
) {
    let cache = tagged(vec![MockCmd::new(
        redis::cmd("MGET").arg("hit").arg("miss").arg("invalid"),
        Ok(vec![
            redis::Value::BulkString(vec![42, 7]),
            redis::Value::Nil,
            redis::Value::BulkString(vec![99, 7]),
        ]),
    )]);

    assert_eq!(
        cache
            .async_batch_get_cache(vec!["hit".into(), "miss".into(), "invalid".into()], context)
            .await
            .unwrap(),
        vec![BatchEntry::Hit(7), BatchEntry::Miss, BatchEntry::Invalid]
    );
}

#[rstest]
#[tokio::test]
async fn ttl_pipeline_keeps_each_entry_ttl_and_defaults_missing_ones() {
    let mut pipeline = redis::pipe();
    pipeline
        .cmd("SETEX")
        .arg("ns:team_id:t1")
        .arg(60u64)
        .arg(r#"{"team_id":"t1"}"#)
        .cmd("SETEX")
        .arg("ns:u1")
        .arg(7u64)
        .arg(r#"{"user_id":"u1"}"#)
        .cmd("SETEX")
        .arg("ns:org_id:o1")
        .arg(300u64)
        .arg(r#"{"a":1}"#);
    let cache = RedisCache::with_connection(
        mock(vec![MockCmd::with_values(
            pipeline,
            Ok(vec!["OK", "OK", "OK"]),
        )]),
        Some(Duration::from_secs(300)),
        JsonCodec::<serde_json::Value>::new(),
    )
    .with_namespace(Some("ns".into()));

    cache
        .async_set_cache_pipeline_with_ttls(vec![
            (
                "team_id:t1".into(),
                json!({"team_id": "t1"}),
                Some(Duration::from_secs(60)),
            ),
            (
                "u1".into(),
                json!({"user_id": "u1"}),
                Some(Duration::from_secs(7)),
            ),
            ("org_id:o1".into(), json!({"a": 1}), None),
        ])
        .await
        .unwrap();
}

#[rstest]
#[tokio::test]
async fn empty_pipelines_skip_the_round_trip(context: ExactCacheContext) {
    let cache = json_cache(Vec::new());
    cache
        .async_set_cache_pipeline(Vec::new(), context)
        .await
        .unwrap();
    cache
        .async_set_cache_pipeline_with_ttls(Vec::new())
        .await
        .unwrap();
    assert_eq!(cache.delete_cache_keys(Vec::new()).await.unwrap(), 0);
    assert_eq!(
        cache.async_rpush_pipeline(Vec::new()).await.unwrap(),
        Vec::<usize>::new()
    );
    assert_eq!(
        cache.async_lpop_pipeline(Vec::new()).await.unwrap(),
        Vec::<RedisLpopResult>::new()
    );
    assert_eq!(
        cache.async_increment_pipeline(Vec::new()).await.unwrap(),
        Vec::<f64>::new()
    );
}

#[rstest]
#[tokio::test]
async fn count_reads_parse_integers_and_keep_missing_counters() {
    let mget = || {
        MockCmd::new(
            redis::cmd("MGET").arg("team:count").arg("team:missing"),
            Ok(redis_test::redis_value!(["7", nil])),
        )
    };
    let cache = team(vec![mget(), mget()]);
    let keys = vec!["count".to_string(), "missing".to_string()];

    assert_eq!(cache.batch_get_counts(&keys).unwrap(), [Some(7), None]);
    assert_eq!(
        cache.async_batch_get_counts(keys).await.unwrap(),
        [Some(7), None]
    );
}

#[rstest]
#[tokio::test]
async fn pings_run_on_both_paths() {
    let cache = team(vec![
        MockCmd::new(redis::cmd("PING"), Ok("PONG")),
        MockCmd::new(redis::cmd("PING"), Ok("PONG")),
    ]);
    assert!(cache.sync_ping().unwrap());
    assert!(cache.ping().await.unwrap());
}

#[rstest]
#[case::remaining(12, Some(Duration::from_secs(12)))]
#[case::no_expiry(-1, None)]
#[case::missing(-2, None)]
#[tokio::test]
async fn ttl_reads_hide_negative_replies(#[case] reply: i64, #[case] ttl: Option<Duration>) {
    let cache = team(vec![MockCmd::new(
        redis::cmd("TTL").arg("team:key"),
        Ok(reply),
    )]);
    assert_eq!(cache.async_get_ttl("key").await.unwrap(), ttl);
}

#[rstest]
#[case::explicit_ttl(Some(Duration::from_secs(30)), 30, 1, true)]
#[case::default_ttl(None, 600, 1, true)]
#[case::missing_key(Some(Duration::from_secs(30)), 30, 0, false)]
#[tokio::test]
async fn refresh_ttl_expires_existing_keys_only(
    #[case] ttl: Option<Duration>,
    #[case] seconds: u64,
    #[case] reply: i64,
    #[case] refreshed: bool,
) {
    let cache = team(vec![MockCmd::new(
        redis::cmd("EXPIRE").arg("team:key").arg(seconds),
        Ok(reply),
    )]);
    assert_eq!(
        cache.async_refresh_ttl("key", ttl).await.unwrap(),
        refreshed
    );
}

#[rstest]
#[tokio::test]
async fn scan_stops_at_count_and_bulk_delete_reports_existing_keys() {
    let cache = team(vec![
        scan(
            "team:job-*",
            0,
            25,
            redis_test::redis_value!(["4", ["team:job-a"]]),
        ),
        scan(
            "team:job-*",
            4,
            25,
            redis_test::redis_value!(["0", ["team:job-b"]]),
        ),
        MockCmd::new(
            redis::cmd("DEL").arg("team:job-a").arg("team:job-b"),
            Ok(2u32),
        ),
    ]);
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
}

#[rstest]
#[tokio::test]
async fn sets_add_members_and_arm_the_default_ttl() {
    let mut pipeline = redis::pipe();
    pipeline
        .cmd("SADD")
        .arg("team:members")
        .arg("a")
        .arg("b")
        .cmd("EXPIRE")
        .arg("team:members")
        .arg(600u64)
        .ignore();
    let cache = team(vec![MockCmd::with_values(
        pipeline,
        Ok(vec![redis::Value::Int(2), redis::Value::Int(1)]),
    )]);
    assert_eq!(
        cache
            .async_set_cache_sadd("members", vec!["a".into(), "b".into()], None)
            .await
            .unwrap(),
        2
    );
    assert_eq!(
        cache
            .async_set_cache_sadd("members", Vec::new(), None)
            .await,
        Err(Error::InvalidEntry)
    );
}

#[rstest]
#[tokio::test]
async fn queues_push_and_pop_namespaced_lists() {
    let cache = team(vec![
        MockCmd::new(
            redis::cmd("RPUSH").arg("team:queue").arg("a").arg("b"),
            Ok(2u32),
        ),
        MockCmd::new(
            redis::cmd("INFO"),
            Ok("# Server\r\nredis_version:7.2.4\r\n"),
        ),
        MockCmd::new(
            redis::cmd("LPOP").arg("team:queue").arg(2usize),
            Ok(redis_test::redis_value!(["a", "b"])),
        ),
        MockCmd::new(redis::cmd("LPOP").arg("team:queue"), Ok("c")),
    ]);
    assert_eq!(
        cache
            .async_rpush("queue", vec!["a".into(), "b".into()])
            .await
            .unwrap(),
        2
    );
    assert_eq!(
        cache.async_rpush("queue", Vec::new()).await,
        Err(Error::InvalidEntry)
    );
    assert_eq!(
        cache.async_lpop("queue", Some(2)).await.unwrap(),
        RedisLpopResult::Values(vec![b"a".to_vec(), b"b".to_vec()])
    );
    assert_eq!(
        cache.async_lpop("queue", None).await.unwrap(),
        RedisLpopResult::Value(b"c".to_vec())
    );
}

/// Python `RedisCache.async_lpop` checks `redis_version` from `INFO` and, below major version 7,
/// pops a counted batch as `count` single-command `LPOP` pipelines, dropping `None` replies.
#[rstest]
#[tokio::test]
async fn counted_lpop_falls_back_to_single_pops_below_redis_7() {
    let single_pop = || {
        let mut pipeline = redis::pipe();
        pipeline.cmd("LPOP").arg("team:queue");
        pipeline
    };
    let cache = team(vec![
        MockCmd::new(
            redis::cmd("INFO"),
            Ok("# Server\r\nredis_version:6.2.14\r\n"),
        ),
        MockCmd::with_values(single_pop(), Ok(vec![redis_test::redis_value!("a")])),
        MockCmd::with_values(single_pop(), Ok(vec![redis_test::redis_value!("b")])),
        MockCmd::with_values(single_pop(), Ok(vec![redis::Value::Nil])),
    ]);

    assert_eq!(
        cache.async_lpop("queue", Some(3)).await.unwrap(),
        RedisLpopResult::Values(vec![b"a".to_vec(), b"b".to_vec()])
    );
}

/// Python keeps `redis_version = "Unknown"` when `INFO` fails and then assumes
/// `DEFAULT_REDIS_MAJOR_VERSION` (7), so a counted pop is one `LPOP key count`. The version is read
/// once: the second pop sends no second `INFO`.
#[rstest]
#[tokio::test]
async fn counted_lpop_assumes_redis_7_when_info_fails() {
    let cache = team(vec![
        MockCmd::new(
            redis::cmd("INFO"),
            Err::<String, _>(redis::RedisError::from((redis::ErrorKind::Io, "down"))),
        ),
        MockCmd::new(
            redis::cmd("LPOP").arg("team:queue").arg(2usize),
            Ok(redis_test::redis_value!(["a", "b"])),
        ),
        MockCmd::new(
            redis::cmd("LPOP").arg("team:queue").arg(1usize),
            Ok(redis_test::redis_value!(["c"])),
        ),
    ]);

    assert_eq!(
        cache.async_lpop("queue", Some(2)).await.unwrap(),
        RedisLpopResult::Values(vec![b"a".to_vec(), b"b".to_vec()])
    );
    assert_eq!(
        cache.async_lpop("queue", Some(1)).await.unwrap(),
        RedisLpopResult::Values(vec![b"c".to_vec()])
    );
}

fn push_and_trim(start: i64) -> redis::Pipeline {
    let mut pipeline = redis::pipe();
    pipeline
        .atomic()
        .cmd("RPUSH")
        .arg("ns:buf")
        .arg("c")
        .arg("d")
        .cmd("LTRIM")
        .arg("ns:buf")
        .arg(start)
        .arg(-1);
    pipeline
}

#[rstest]
#[case::keeps_newest_entries(3, -3)]
#[case::zero_keeps_everything(0, 0)]
#[tokio::test]
async fn rpush_and_trim_runs_push_and_trim_in_one_transaction(
    #[case] max_len: usize,
    #[case] start: i64,
) {
    let cache = json_cache(vec![MockCmd::with_values(
        push_and_trim(start),
        Ok(vec![redis::Value::Array(vec![
            redis::Value::Int(4),
            redis::Value::Okay,
        ])]),
    )])
    .with_namespace(Some("ns".into()));

    assert_eq!(
        cache
            .async_rpush_and_trim("buf", vec!["c".into(), "d".into()], max_len)
            .await
            .unwrap(),
        4
    );
}

#[rstest]
#[tokio::test]
async fn rpush_and_trim_raises_when_a_queued_command_fails() {
    let wrong_type = redis::parse_redis_value(
        b"-WRONGTYPE Operation against a key holding the wrong kind of value\r\n",
    )
    .unwrap();
    let cache = json_cache(vec![MockCmd::with_values(
        push_and_trim(-3),
        Ok(vec![redis::Value::Array(vec![
            wrong_type,
            redis::Value::Okay,
        ])]),
    )])
    .with_namespace(Some("ns".into()));

    assert_eq!(
        cache
            .async_rpush_and_trim("buf", vec!["c".into(), "d".into()], 3)
            .await,
        Err(Error::Unavailable)
    );
    assert_eq!(
        cache.async_rpush_and_trim("buf", Vec::new(), 3).await,
        Err(Error::InvalidEntry)
    );
}

#[rstest]
#[tokio::test]
async fn scripts_and_eval_namespace_their_keys() {
    let eval = || {
        MockCmd::new(
            redis::cmd("EVAL")
                .arg("return KEYS[1]")
                .arg(1usize)
                .arg("team:key"),
            Ok("team:key"),
        )
    };
    let cache = team(vec![eval(), eval()]);
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
}

#[rstest]
fn client_list_and_info_return_server_text() {
    let cache = team(vec![
        MockCmd::new(redis::cmd("CLIENT").arg("LIST"), Ok("id=1")),
        MockCmd::new(redis::cmd("INFO"), Ok("redis_version:7")),
    ]);
    assert_eq!(cache.client_list().unwrap(), "id=1");
    assert_eq!(cache.info().unwrap(), "redis_version:7");
}

#[rstest]
#[tokio::test]
async fn pipelines_preserve_operation_order() {
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
    let queue = team(vec![
        MockCmd::with_values(
            rpush_pipeline,
            Ok(vec![redis::Value::Int(1), redis::Value::Int(2)]),
        ),
        MockCmd::with_values(
            lpop_pipeline,
            Ok(vec![redis_test::redis_value!(["one"]), redis::Value::Nil]),
        ),
    ]);

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
}

#[rstest]
#[tokio::test]
async fn increment_pipeline_expires_only_operations_with_a_ttl() {
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
    let counters = team(vec![MockCmd::with_values(
        increment_pipeline,
        Ok(vec![
            redis::Value::BulkString(b"1.5".to_vec()),
            redis::Value::Int(1),
            redis::Value::BulkString(b"3.5".to_vec()),
        ]),
    )]);
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

const INCREMENT_SCRIPT: &str = concat!(
    "local value = redis.call('INCRBYFLOAT', KEYS[1], ARGV[1]); ",
    "if redis.call('TTL', KEYS[1]) == -1 then ",
    "redis.call('EXPIRE', KEYS[1], ARGV[2]); end; return value"
);
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

fn increment_script(amount: f64) -> MockCmd {
    MockCmd::new(
        redis::cmd("EVAL")
            .arg(INCREMENT_SCRIPT)
            .arg(1)
            .arg("counter")
            .arg(amount)
            .arg(600),
        Ok("4.5"),
    )
}

#[rstest]
#[tokio::test]
async fn increments_keep_an_existing_ttl_in_one_atomic_script(context: ExactCacheContext) {
    let cache = RedisCache::with_connection(
        mock(vec![increment_script(2.5), increment_script(2.5)]),
        None,
        JsonCodec::<f64>::new(),
    );

    assert_eq!(
        cache
            .increment_cache("counter", 2.5, context.clone())
            .unwrap(),
        4.5
    );
    assert_eq!(
        cache
            .async_increment("counter", 2.5, context, false)
            .await
            .unwrap(),
        4.5
    );
}

#[rstest]
#[case::explicit_ttl(Some(Duration::from_secs(60)), 60u64)]
#[case::default_ttl(None, 600u64)]
#[tokio::test]
async fn refresh_ttl_increments_rearm_the_ttl_in_the_same_round_trip(
    #[case] ttl: Option<Duration>,
    #[case] seconds: u64,
) {
    let mut pipeline = redis::pipe();
    pipeline
        .cmd("INCRBYFLOAT")
        .arg("ns:spend:key:k")
        .arg(1.5f64)
        .cmd("EXPIRE")
        .arg("ns:spend:key:k")
        .arg(seconds);
    let cache = json_cache(vec![MockCmd::with_values(
        pipeline,
        Ok(vec![
            redis::Value::BulkString(b"1.5".to_vec()),
            redis::Value::Int(1),
        ]),
    )])
    .with_namespace(Some("ns".into()));

    assert_eq!(
        cache
            .async_increment("spend:key:k", 1.5, ExactCacheContext { ttl }, true)
            .await
            .unwrap(),
        1.5
    );
}

#[rstest]
#[tokio::test]
async fn counter_repairs_are_atomic_and_use_default_ttl() {
    let floor = || {
        MockCmd::new(
            redis::cmd("EVAL")
                .arg(INCREMENT_WITH_FLOOR_SCRIPT)
                .arg(1)
                .arg("team:counter")
                .arg(-2i64)
                .arg(30u64),
            Ok(0i64),
        )
    };
    let cache = team(vec![
        floor(),
        floor(),
        MockCmd::new(
            redis::cmd("EVAL")
                .arg(SET_MAX_SCRIPT)
                .arg(1)
                .arg("team:counter")
                .arg(4.5f64)
                .arg(600u64),
            Ok("4.5"),
        ),
    ]);

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

fn claim_eval(expected: &str, write: &str, refresh: bool, applied: i64) -> MockCmd {
    MockCmd::new(
        redis::cmd("EVAL")
            .arg(CLAIM_SCRIPT)
            .arg(1)
            .arg("pin")
            .arg(expected)
            .arg(600)
            .arg(write)
            .arg(u8::from(refresh)),
        Ok(applied),
    )
}

#[rstest]
#[tokio::test]
async fn claims_match_eligible_values_written_by_another_encoder(context: ExactCacheContext) {
    let python_payload = r#"{"model_id": "a", "deployment": "east"}"#;
    let stored = json!({"deployment": "east", "model_id": "a"});
    let cache = json_cache(vec![
        MockCmd::new(redis::cmd("GET").arg("pin"), Ok(python_payload)),
        claim_eval(python_payload, "", true, 1),
    ]);

    assert_eq!(
        cache
            .async_claim_cache(
                "pin",
                json!({"model_id": "b"}),
                vec![stored.clone()],
                context
            )
            .await
            .unwrap(),
        stored
    );
}

#[rstest]
fn claims_retry_when_the_key_changes_and_replace_ineligible_winners(context: ExactCacheContext) {
    let candidate = json!({"model_id": "b"});
    let payload = r#"{"model_id":"b"}"#;
    let cache = json_cache(vec![
        MockCmd::new(redis::cmd("GET").arg("pin"), Ok(redis::Value::Nil)),
        claim_eval("", payload, false, 0),
        MockCmd::new(redis::cmd("GET").arg("pin"), Ok(r#"{"model_id":"gone"}"#)),
        claim_eval(r#"{"model_id":"gone"}"#, payload, false, 1),
    ]);

    assert_eq!(
        cache
            .claim_cache(
                "pin",
                candidate.clone(),
                &[json!({"model_id": "a"})],
                context
            )
            .unwrap(),
        candidate
    );
}

#[rstest]
fn claims_without_eligible_values_keep_the_winner_without_refreshing_its_ttl(
    context: ExactCacheContext,
) {
    let stored = r#"{"model_id": "a"}"#;
    let cache = json_cache(vec![
        MockCmd::new(redis::cmd("GET").arg("pin"), Ok(stored)),
        claim_eval(stored, "", false, 1),
    ]);

    assert_eq!(
        cache
            .claim_cache("pin", json!({"model_id": "b"}), &[], context)
            .unwrap(),
        json!({"model_id": "a"})
    );
}

#[rstest]
#[tokio::test]
async fn test_connection_reports_success_with_the_python_message() {
    let cache = team(vec![MockCmd::new(redis::cmd("PING"), Ok("PONG"))]);

    let result = cache.test_connection().await.unwrap();
    assert_eq!(result.status, CacheConnectionStatus::Success);
    assert_eq!(result.message, "Redis connection test successful");
    assert_eq!(result.error, None);
}

#[rstest]
#[case::unexpected_reply(Ok("NOPE"), "Redis ping returned False", false)]
#[case::connection_refused(
    Err(redis::RedisError::from((redis::ErrorKind::Io, "connection refused"))),
    "Redis connection failed:",
    true
)]
#[tokio::test]
async fn test_connection_failures_use_the_python_result_contract(
    #[case] reply: redis::RedisResult<&'static str>,
    #[case] message: &str,
    #[case] has_error: bool,
) {
    let cache = json_cache(vec![MockCmd::new(redis::cmd("PING"), reply)]);

    let result = cache.test_connection().await.unwrap();
    assert_eq!(result.status, CacheConnectionStatus::Failed);
    assert!(result.message.starts_with(message), "{}", result.message);
    assert_eq!(result.error.is_some(), has_error);
}

#[rstest]
#[tokio::test]
async fn disconnect_keeps_a_caller_owned_connection_usable() {
    let cache = team(vec![MockCmd::new(redis::cmd("PING"), Ok("PONG"))]);
    cache.disconnect().await.unwrap();
    assert!(cache.ping().await.unwrap());
}

#[rstest]
#[tokio::test]
async fn disconnect_drains_an_idle_pool_without_connecting() {
    let cache = RedisCache::new("redis://127.0.0.1:1", None, JsonCodec::<String>::new()).unwrap();
    cache.disconnect().await.unwrap();
}
