mod support;

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_cache::{
    BaseCache, Error, JsonCodec, SemanticCacheContext,
    semantic::{PreparedEmbedding, SemanticCache, SemanticLookup},
};
use litellm_cache_valkey_semantic::{
    DEFAULT_INDEX_NAME, ValkeySemanticCache, ValkeySemanticConfig,
};
use redis_test::MockRedisConnection;
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use support::{EmbedCalls, FakeEmbedder, RecordingConnection};

type Requests = Arc<Mutex<Vec<Vec<u8>>>>;
type RecordingCache = ValkeySemanticCache<FakeEmbedder, JsonCodec<Value>, RecordingConnection>;

const KEY_SCOPE: &str = "2c70e12b7a0646f92279f427c7b38e7334d8e5389cff167a1dc30e73f826b683";

struct Recording {
    cache: RecordingCache,
    requests: Requests,
    calls: EmbedCalls,
}

impl Recording {
    fn text(&self) -> String {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .map(|request| String::from_utf8_lossy(request).into_owned())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn config() -> ValkeySemanticConfig {
    ValkeySemanticConfig {
        similarity_threshold: 0.8,
        index_name: "test".into(),
    }
}

fn recording(replies: impl IntoIterator<Item = redis::RedisResult<redis::Value>>) -> Recording {
    let connection = RecordingConnection::new(replies);
    let requests = connection.requests();
    let embedder = FakeEmbedder::new(&[]);
    let calls = Arc::clone(&embedder.calls);
    Recording {
        cache: ValkeySemanticCache::with_connection(
            connection,
            embedder,
            JsonCodec::new(),
            config(),
        ),
        requests,
        calls,
    }
}

#[fixture]
fn entry() -> Value {
    json!({"timestamp": 1.0, "response": {"answer": "ok"}})
}

#[fixture]
fn context() -> SemanticCacheContext {
    SemanticCacheContext {
        messages: Some(json!([{"role": "user", "content": "hello"}])),
        metadata: Some(json!({"source": "test"})),
        ..Default::default()
    }
}

fn ok() -> redis::RedisResult<redis::Value> {
    Ok(redis::Value::SimpleString("OK".into()))
}

fn already_exists() -> redis::RedisResult<redis::Value> {
    Err(redis::RedisError::from((
        redis::ErrorKind::Io,
        "already exists",
    )))
}

fn bulk(value: &[u8]) -> redis::Value {
    redis::Value::BulkString(value.to_vec())
}

fn search_hit(response: &[u8], distance: &str) -> redis::Value {
    redis::Value::Array(vec![
        redis::Value::Int(1),
        bulk(b"test:document"),
        redis::Value::Array(vec![
            bulk(b"response"),
            bulk(response),
            bulk(b"vector_distance"),
            bulk(distance.as_bytes()),
        ]),
    ])
}

fn encoded(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).unwrap()
}

/// `FT.INFO` with the vector field's dimension nested one level down, as valkey-search reports.
fn nested_dimension_info(dimension: i64) -> redis::Value {
    redis::Value::Array(vec![
        redis::Value::SimpleString("attributes".into()),
        redis::Value::Array(vec![redis::Value::Array(vec![
            redis::Value::SimpleString("embedding".into()),
            redis::Value::Array(vec![
                redis::Value::SimpleString("dimensions".into()),
                redis::Value::Int(dimension),
            ]),
        ])]),
    ])
}

/// `FT.INFO` with `dimensions` as a sibling string of the identifier.
fn flat_dimension_info(dimension: &str) -> redis::Value {
    redis::Value::Array(vec![
        redis::Value::SimpleString("attributes".into()),
        redis::Value::Array(vec![redis::Value::Array(vec![
            redis::Value::SimpleString("identifier".into()),
            redis::Value::SimpleString("embedding".into()),
            redis::Value::Array(vec![
                redis::Value::SimpleString("dimensions".into()),
                redis::Value::SimpleString(dimension.into()),
            ]),
        ])]),
    ])
}

#[rstest]
#[case::string_content(json!([{"content": "hello"}]), None, Some("hello"))]
#[case::text_parts(json!([{"content": [{"text": "hello"}, {"text": " world"}]}]), None, Some("hello world"))]
#[case::non_object_parts_are_skipped(json!([{"content": ["raw", {"text": "hello"}]}]), None, Some("hello"))]
#[case::search_results(json!([{"search_results": [{"source": "s", "title": "t", "content": [{"text": "c"}], "citations": ["x"]}]}]), None, Some(r#"stc["x"]"#))]
#[case::responses_string_input(json!([]), Some(json!(" hello ")), Some("hello"))]
#[case::responses_item_input(json!([]), Some(json!([{"content": "first"}, {"text": "second"}])), Some("first\nsecond"))]
#[case::blank_input(json!([]), Some(json!("   ")), None)]
fn prompt_shapes_follow_redis_semantic_extraction(
    #[case] messages: Value,
    #[case] input: Option<Value>,
    #[case] expected: Option<&str>,
) {
    let recording = recording([ok(), Ok(redis::Value::Array(vec![redis::Value::Int(0)]))]);
    let context = SemanticCacheContext {
        messages: Some(messages),
        input,
        ..Default::default()
    };

    assert_eq!(recording.cache.get_cache("key", &context).unwrap(), None);

    let prompts = recording
        .calls
        .lock()
        .unwrap()
        .iter()
        .map(|(prompt, _)| prompt.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        prompts,
        expected.into_iter().map(str::to_owned).collect::<Vec<_>>()
    );
    assert_eq!(
        recording.requests.lock().unwrap().is_empty(),
        expected.is_none()
    );
}

#[rstest]
#[case::key("key", KEY_SCOPE)]
#[case::empty_key("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")]
fn documents_are_scoped_by_the_keys_sha256(
    #[case] key: &str,
    #[case] scope: &str,
    entry: Value,
    context: SemanticCacheContext,
) {
    let recording = recording([ok()]);

    recording.cache.set_cache(key, entry, &context).unwrap();

    let text = recording.text();
    assert!(text.contains(&format!("test:{scope}:")));
    assert!(text.contains(&format!("litellm_cache_key\r\n$64\r\n{scope}\r\n")));
}

#[rstest]
#[case::no_ttl(None, None)]
#[case::whole_seconds(Some(Duration::from_secs(5)), Some("5"))]
#[case::fractional_seconds_truncate(Some(Duration::from_millis(1900)), Some("1"))]
fn set_writes_hset_and_expires_only_with_a_ttl(
    #[case] ttl: Option<Duration>,
    #[case] expire: Option<&str>,
    entry: Value,
    context: SemanticCacheContext,
) {
    let recording = recording([ok()]);

    recording
        .cache
        .set_cache(
            "key",
            entry,
            &SemanticCacheContext {
                ttl,
                ..context.clone()
            },
        )
        .unwrap();

    let text = recording.text();
    assert!(text.contains("FT.CREATE"));
    assert!(text.contains("HSET"));
    let expire_seconds = text
        .split_once("EXPIRE\r\n")
        .and_then(|(_, rest)| rest.split("\r\n").nth(3));
    assert_eq!(expire_seconds, expire);
    assert_eq!(
        *recording.calls.lock().unwrap(),
        vec![("hello".to_owned(), context.metadata)]
    );
}

#[rstest]
fn second_set_skips_create_after_dimension_is_cached(entry: Value, context: SemanticCacheContext) {
    let recording = recording([ok()]);

    recording
        .cache
        .set_cache("key", entry.clone(), &context)
        .unwrap();
    recording.cache.set_cache("key", entry, &context).unwrap();

    let text = recording.text();
    assert_eq!(text.matches("FT.CREATE").count(), 1);
    assert_eq!(text.matches("HSET").count(), 2);
}

#[rstest]
#[case::nested_matching(nested_dimension_info(3), Ok(()))]
#[case::nested_mismatch(nested_dimension_info(2), Err(Error::Unavailable))]
#[case::flat_matching(flat_dimension_info("3"), Ok(()))]
#[case::flat_mismatch(flat_dimension_info("2"), Err(Error::Unavailable))]
#[case::unreported_dimension_is_accepted(redis::Value::Array(vec![]), Ok(()))]
fn existing_index_dimension_must_match_embedding(
    #[case] info: redis::Value,
    #[case] expected: Result<(), Error>,
    entry: Value,
    context: SemanticCacheContext,
) {
    let recording = recording([already_exists(), Ok(info)]);

    assert_eq!(recording.cache.set_cache("key", entry, &context), expected);
}

#[rstest]
#[case::create_failure(Err(redis::RedisError::from((redis::ErrorKind::Io, "boom"))))]
fn index_creation_failures_are_unavailable(
    #[case] reply: redis::RedisResult<redis::Value>,
    entry: Value,
    context: SemanticCacheContext,
) {
    let recording = recording([reply]);

    assert_eq!(
        recording.cache.set_cache("key", entry, &context),
        Err(Error::Unavailable)
    );
}

#[rstest]
#[case::within_threshold(search_hit(&encoded(&entry()), "0.1"), Ok(Some(entry())))]
#[case::at_threshold(search_hit(&encoded(&entry()), "0.2"), Ok(Some(entry())))]
#[case::beyond_threshold(search_hit(&encoded(&entry()), "0.5"), Ok(None))]
#[case::zero_documents(redis::Value::Array(vec![redis::Value::Int(0)]), Ok(None))]
#[case::missing_response(
    redis::Value::Array(vec![
        redis::Value::Int(1),
        bulk(b"document"),
        redis::Value::Array(vec![bulk(b"vector_distance"), bulk(b"0.1")]),
    ]),
    Err(Error::InvalidEntry)
)]
#[case::unparsable_distance(search_hit(b"not-json", "abc"), Err(Error::InvalidEntry))]
#[case::undecodable_response(search_hit(b"not-json", "0.1"), Err(Error::InvalidEntry))]
fn get_applies_threshold_and_decodes_entry(
    #[case] reply: redis::Value,
    #[case] expected: Result<Option<Value>, Error>,
    context: SemanticCacheContext,
) {
    let recording = recording([ok(), Ok(reply)]);

    assert_eq!(recording.cache.get_cache("key", &context), expected);
}

#[rstest]
#[case::hit(context(), Some(search_hit(&encoded(&entry()), "0.1")), Some(entry()), Some(1.0 - 0.1))]
#[case::below_threshold(context(), Some(search_hit(&encoded(&entry()), "0.5")), None, Some(1.0 - 0.5))]
#[case::no_results(context(), Some(redis::Value::Array(vec![redis::Value::Int(0)])), None, Some(0.0))]
#[case::no_prompt(SemanticCacheContext::default(), None, None, Some(0.0))]
#[tokio::test]
async fn lookup_reports_python_semantic_similarity(
    #[case] context: SemanticCacheContext,
    #[case] reply: Option<redis::Value>,
    #[case] value: Option<Value>,
    #[case] similarity: Option<f64>,
    #[values(false, true)] use_async: bool,
) {
    let searched = reply.is_some();
    let recording = recording(reply.map_or_else(Vec::new, |reply| vec![ok(), Ok(reply)]));

    let lookup = if use_async {
        recording
            .cache
            .async_get_cache_with_similarity("key", &context)
            .await
    } else {
        recording.cache.get_cache_with_similarity("key", &context)
    };

    assert_eq!(lookup, Ok(SemanticLookup { value, similarity }));
    assert_eq!(recording.text().contains("FT.SEARCH"), searched);
}

#[rstest]
#[tokio::test]
async fn missing_prompt_does_not_touch_valkey(entry: Value) {
    let cache = ValkeySemanticCache::with_connection(
        MockRedisConnection::new([]).assert_all_commands_consumed(),
        FakeEmbedder::new(&[]),
        JsonCodec::new(),
        config(),
    );
    let context = SemanticCacheContext::default();

    cache.set_cache("key", entry.clone(), &context).unwrap();
    assert_eq!(cache.get_cache("key", &context).unwrap(), None);
    cache
        .async_set_cache("key", entry, context.clone())
        .await
        .unwrap();
    assert_eq!(cache.async_get_cache("key", &context).await.unwrap(), None);
    assert_eq!(cache.get_ttl(&context), None);
}

#[rstest]
fn with_embedder_shares_index_state_and_connections(entry: Value, context: SemanticCacheContext) {
    let recording = recording([ok(), ok(), Ok(search_hit(&encoded(&entry), "0.1"))]);
    recording
        .cache
        .set_cache("key", entry.clone(), &context)
        .unwrap();

    let prepared = recording
        .cache
        .with_embedder(PreparedEmbedding(vec![0.1, 0.2, 0.3]));

    assert_eq!(prepared.get_cache("key", &context).unwrap(), Some(entry));
    assert_eq!(recording.text().matches("FT.CREATE").count(), 1);
}

#[rstest]
fn accessors_report_the_config() {
    let recording = recording([]);

    assert_eq!(recording.cache.index_name(), "test");
    assert_eq!(recording.cache.similarity_threshold(), 0.8);
    assert_eq!(DEFAULT_INDEX_NAME, "litellm_semantic_cache_index");
}

#[rstest]
#[tokio::test]
async fn async_set_and_get_use_shared_document_helpers(
    entry: Value,
    context: SemanticCacheContext,
) {
    let recording = recording([ok(), ok(), ok(), Ok(search_hit(&encoded(&entry), "0.1"))]);
    let context = SemanticCacheContext {
        ttl: Some(Duration::from_millis(1900)),
        ..context
    };

    recording
        .cache
        .async_set_cache("key", entry.clone(), context.clone())
        .await
        .unwrap();
    assert_eq!(
        recording
            .cache
            .async_get_cache("key", &context)
            .await
            .unwrap(),
        Some(entry)
    );

    let text = recording.text();
    assert!(text.contains("FT.CREATE"));
    assert!(text.contains("HSET"));
    assert!(text.contains("EXPIRE"));
    assert_eq!(
        *recording.calls.lock().unwrap(),
        vec![("hello".to_owned(), context.metadata.clone()); 2]
    );
}
