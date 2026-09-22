use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_cache::{BaseCache, CacheCodec, Error, SemanticCacheContext};
use litellm_cache_redis_semantic::{Embedder, RedisSemanticCache, RedisSemanticConfig};
use litellm_cache_response::{CacheEntry, ResponseCacheCodec};
use redis_test::{MockCmd, MockRedisConnection};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const INDEX: &str = "litellm_semantic_cache_index";

struct FakeEmbedder {
    vectors: HashMap<String, Vec<f32>>,
    calls: Arc<Mutex<Vec<String>>>,
}

impl FakeEmbedder {
    fn new(vectors: &[(&str, &[f32])]) -> (Self, Arc<Mutex<Vec<String>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                vectors: vectors
                    .iter()
                    .map(|(prompt, vector)| (prompt.to_string(), vector.to_vec()))
                    .collect(),
                calls: Arc::clone(&calls),
            },
            calls,
        )
    }
}

impl Embedder for FakeEmbedder {
    fn embed(&self, prompt: &str, _: Option<&Value>) -> Result<Vec<f32>, Error> {
        self.calls.lock().unwrap().push(prompt.to_string());

        Ok(self
            .vectors
            .get(prompt)
            .cloned()
            .unwrap_or_else(|| vec![0.1, 0.2, 0.3]))
    }

    async fn async_embed(&self, prompt: &str, metadata: Option<&Value>) -> Result<Vec<f32>, Error> {
        self.embed(prompt, metadata)
    }
}

fn config() -> RedisSemanticConfig {
    RedisSemanticConfig {
        index_name: INDEX.into(),
        similarity_threshold: 0.9,
    }
}

fn messages_context(messages: Vec<Value>) -> SemanticCacheContext {
    SemanticCacheContext {
        messages: Some(Value::Array(messages)),
        ..Default::default()
    }
}

fn entry() -> CacheEntry {
    CacheEntry {
        timestamp: Some(1.0),
        response: json!({"answer": "yes"}),
    }
}

fn encoded(entry: &CacheEntry) -> Vec<u8> {
    ResponseCacheCodec.encode(entry).unwrap()
}

fn vector_bytes(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|component| component.to_le_bytes())
        .collect()
}

fn entry_id(prompt: &str, tag: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(prompt.as_bytes());
    digest.update(b"litellm_cache_key");
    digest.update(tag.as_bytes());
    format!("{:x}", digest.finalize())
}

fn s(value: &str) -> redis::Value {
    redis::Value::BulkString(value.as_bytes().to_vec())
}

fn unknown_index_error() -> redis::RedisError {
    redis::RedisError::from((redis::ErrorKind::Extension, "Unknown index name"))
}

fn attribute(name: &str, field_type: &str, extra: Vec<redis::Value>) -> redis::Value {
    let mut parts = vec![
        s("identifier"),
        s(name),
        s("attribute"),
        s(name),
        s("type"),
        s(field_type),
    ];
    parts.extend(extra);
    redis::Value::Array(parts)
}

fn index_info(attributes: Vec<redis::Value>) -> redis::Value {
    redis::Value::Array(vec![
        s("index_name"),
        s(INDEX),
        s("attributes"),
        redis::Value::Array(attributes),
    ])
}

fn vector_attribute_with(dims: i64, data_type: &str, distance_metric: &str) -> redis::Value {
    attribute(
        "prompt_vector",
        "VECTOR",
        vec![
            s("algorithm"),
            s("FLAT"),
            s("data_type"),
            s(data_type),
            s("dim"),
            redis::Value::Int(dims),
            s("distance_metric"),
            s(distance_metric),
        ],
    )
}

fn vector_attribute(dims: i64) -> redis::Value {
    vector_attribute_with(dims, "FLOAT32", "COSINE")
}

fn info_with_vector(vector: redis::Value) -> redis::Value {
    index_info(vec![
        attribute("prompt", "TEXT", vec![]),
        attribute("response", "TEXT", vec![]),
        attribute("inserted_at", "NUMERIC", vec![]),
        attribute("updated_at", "NUMERIC", vec![]),
        vector,
        attribute("litellm_cache_key", "TAG", vec![]),
    ])
}

fn compatible_info(dims: i64) -> redis::Value {
    info_with_vector(vector_attribute(dims))
}

fn unscoped_info(dims: i64) -> redis::Value {
    index_info(vec![
        attribute("prompt", "TEXT", vec![]),
        attribute("response", "TEXT", vec![]),
        attribute("inserted_at", "NUMERIC", vec![]),
        attribute("updated_at", "NUMERIC", vec![]),
        vector_attribute(dims),
    ])
}

fn create_index_command(name: &str, dims: usize) -> redis::Cmd {
    let mut command = redis::cmd("FT.CREATE");
    command
        .arg(name)
        .arg("ON")
        .arg("HASH")
        .arg("PREFIX")
        .arg(1)
        .arg(name)
        .arg("SCORE")
        .arg(1.0)
        .arg("SCHEMA")
        .arg("prompt")
        .arg("TEXT")
        .arg("WEIGHT")
        .arg(1)
        .arg("response")
        .arg("TEXT")
        .arg("WEIGHT")
        .arg(1)
        .arg("inserted_at")
        .arg("NUMERIC")
        .arg("updated_at")
        .arg("NUMERIC")
        .arg("prompt_vector")
        .arg("VECTOR")
        .arg("FLAT")
        .arg(6)
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dims)
        .arg("DISTANCE_METRIC")
        .arg("COSINE")
        .arg("litellm_cache_key")
        .arg("TAG")
        .arg("SEPARATOR")
        .arg(",");
    command
}

fn search_command(index: &str, tag: &str, vector: &[f32]) -> redis::Cmd {
    let mut command = redis::cmd("FT.SEARCH");
    command
        .arg(index)
        .arg(format!(
            "(@litellm_cache_key:{{{tag}}})=>[KNN 1 @prompt_vector $vector AS vector_distance]"
        ))
        .arg("RETURN")
        .arg(8)
        .arg("entry_id")
        .arg("prompt")
        .arg("response")
        .arg("inserted_at")
        .arg("updated_at")
        .arg("metadata")
        .arg("litellm_cache_key")
        .arg("vector_distance")
        .arg("SORTBY")
        .arg("vector_distance")
        .arg("ASC")
        .arg("DIALECT")
        .arg(2)
        .arg("LIMIT")
        .arg(0)
        .arg(1)
        .arg("PARAMS")
        .arg(2)
        .arg("vector")
        .arg(vector_bytes(vector));
    command
}

fn hit_fields(tag: &str, distance: &str, response: Vec<u8>) -> redis::Value {
    redis::Value::Array(vec![
        s("entry_id"),
        s("stored-id"),
        s("prompt"),
        s("hello prompt"),
        s("response"),
        redis::Value::BulkString(response),
        s("inserted_at"),
        s("1700000000.5"),
        s("updated_at"),
        s("1700000000.5"),
        s("litellm_cache_key"),
        s(tag),
        s("vector_distance"),
        s(distance),
    ])
}

fn search_result(fields: redis::Value) -> redis::Value {
    redis::Value::Array(vec![
        redis::Value::Int(1),
        s("litellm_semantic_cache_index:stored-id"),
        fields,
    ])
}

fn empty_result() -> redis::Value {
    redis::Value::Array(vec![redis::Value::Int(0)])
}

#[test]
fn store_creates_index_and_writes_hash_with_expire() {
    let vector = vec![0.1f32, 0.2, 0.3];
    let prompt = "hello prompt";
    let tag = "key1";
    let hash_key = format!("{INDEX}:{}", entry_id(prompt, tag));
    let value = entry();
    let connection = MockRedisConnection::new([
        MockCmd::new(
            redis::cmd("FT.INFO").arg(INDEX),
            Err::<redis::Value, _>(unknown_index_error()),
        ),
        MockCmd::new(create_index_command(INDEX, 3), Ok("OK")),
        MockCmd::new(
            redis::cmd("HSET")
                .arg(&hash_key)
                .arg("entry_id")
                .arg(entry_id(prompt, tag))
                .arg("prompt")
                .arg(prompt)
                .arg("response")
                .arg(encoded(&value))
                .arg("prompt_vector")
                .arg(vector_bytes(&vector))
                .arg("inserted_at")
                .arg("1700000000.5")
                .arg("updated_at")
                .arg("1700000000.5")
                .arg("litellm_cache_key")
                .arg(tag),
            Ok(7),
        ),
        MockCmd::new(redis::cmd("EXPIRE").arg(&hash_key).arg(5), Ok(1)),
    ])
    .assert_all_commands_consumed();
    let (embedder, _) = FakeEmbedder::new(&[(prompt, &vector)]);
    let cache = RedisSemanticCache::with_connection(connection, embedder, config())
        .with_clock(|| 1700000000.5);

    let context = SemanticCacheContext {
        ttl: Some(Duration::from_secs(5)),
        ..messages_context(vec![json!({"role": "user", "content": prompt})])
    };
    cache.set_cache(tag, value, &context).unwrap();
}

#[test]
fn store_without_ttl_skips_expire() {
    let prompt = "hello prompt";
    let connection = MockRedisConnection::new([
        MockCmd::new(redis::cmd("FT.INFO").arg(INDEX), Ok(compatible_info(3))),
        MockCmd::new(
            redis::cmd("HSET")
                .arg(format!("{INDEX}:{}", entry_id(prompt, "key1")))
                .arg("entry_id")
                .arg(entry_id(prompt, "key1"))
                .arg("prompt")
                .arg(prompt)
                .arg("response")
                .arg(encoded(&entry()))
                .arg("prompt_vector")
                .arg(vector_bytes(&[0.1f32, 0.2, 0.3]))
                .arg("inserted_at")
                .arg("1700000000.5")
                .arg("updated_at")
                .arg("1700000000.5")
                .arg("litellm_cache_key")
                .arg("key1"),
            Ok(7),
        ),
    ])
    .assert_all_commands_consumed();
    let (embedder, _) = FakeEmbedder::new(&[]);
    let cache = RedisSemanticCache::with_connection(connection, embedder, config())
        .with_clock(|| 1700000000.5);

    cache
        .set_cache(
            "key1",
            entry(),
            &messages_context(vec![json!({"role": "user", "content": prompt})]),
        )
        .unwrap();
}

#[test]
fn lookup_returns_hit_below_distance_threshold() {
    let vector = vec![0.1f32, 0.2, 0.3];
    let value = entry();
    let connection = MockRedisConnection::new([
        MockCmd::new(redis::cmd("FT.INFO").arg(INDEX), Ok(compatible_info(3))),
        MockCmd::new(
            search_command(INDEX, "key1", &vector),
            Ok(search_result(hit_fields("key1", "0.05", encoded(&value)))),
        ),
    ])
    .assert_all_commands_consumed();
    let (embedder, _) = FakeEmbedder::new(&[]);
    let cache = RedisSemanticCache::with_connection(connection, embedder, config());

    let hit = cache
        .get_cache(
            "key1",
            &messages_context(vec![json!({"role": "user", "content": "hello prompt"})]),
        )
        .unwrap();
    assert_eq!(hit, Some(value));
}

#[test]
fn lookup_misses_above_distance_threshold_and_on_tag_mismatch() {
    let vector = vec![0.1f32, 0.2, 0.3];
    let connection = MockRedisConnection::new([
        MockCmd::new(redis::cmd("FT.INFO").arg(INDEX), Ok(compatible_info(3))),
        MockCmd::new(
            search_command(INDEX, "key1", &vector),
            Ok(search_result(hit_fields("key1", "0.5", encoded(&entry())))),
        ),
        MockCmd::new(
            search_command(INDEX, "key1", &vector),
            Ok(search_result(hit_fields(
                "other",
                "0.05",
                encoded(&entry()),
            ))),
        ),
    ])
    .assert_all_commands_consumed();
    let (embedder, _) = FakeEmbedder::new(&[]);
    let cache = RedisSemanticCache::with_connection(connection, embedder, config());
    let context = messages_context(vec![json!({"role": "user", "content": "hello prompt"})]);

    assert_eq!(cache.get_cache("key1", &context).unwrap(), None);
    assert_eq!(cache.get_cache("key1", &context).unwrap(), None);
}

#[test]
fn lookup_returns_invalid_entry_on_malformed_response() {
    let vector = vec![0.1f32, 0.2, 0.3];
    let connection = MockRedisConnection::new([
        MockCmd::new(redis::cmd("FT.INFO").arg(INDEX), Ok(compatible_info(3))),
        MockCmd::new(
            search_command(INDEX, "key1", &vector),
            Ok(search_result(hit_fields(
                "key1",
                "0.05",
                b"not json!".to_vec(),
            ))),
        ),
    ])
    .assert_all_commands_consumed();
    let (embedder, _) = FakeEmbedder::new(&[]);
    let cache = RedisSemanticCache::with_connection(connection, embedder, config());

    assert_eq!(
        cache
            .get_cache(
                "key1",
                &messages_context(vec![json!({"role": "user", "content": "hello prompt"})])
            )
            .unwrap_err(),
        Error::InvalidEntry
    );
}

#[test]
fn missing_prompt_is_noop_and_never_embeds() {
    let connection = MockRedisConnection::new(Vec::<MockCmd>::new()).assert_all_commands_consumed();
    let (embedder, calls) = FakeEmbedder::new(&[]);
    let cache = RedisSemanticCache::with_connection(connection, embedder, config());

    let context = SemanticCacheContext::default();
    cache.set_cache("key1", entry(), &context).unwrap();
    assert_eq!(cache.get_cache("key1", &context).unwrap(), None);
    assert!(calls.lock().unwrap().is_empty());
}

#[test]
fn scope_overrides_key_as_filter_tag() {
    let vector = vec![0.1f32, 0.2, 0.3];
    let prompt = "hello prompt";
    let value = entry();
    let connection = MockRedisConnection::new([
        MockCmd::new(redis::cmd("FT.INFO").arg(INDEX), Ok(compatible_info(3))),
        MockCmd::new(
            redis::cmd("HSET")
                .arg(format!("{INDEX}:{}", entry_id(prompt, "scope-a")))
                .arg("entry_id")
                .arg(entry_id(prompt, "scope-a"))
                .arg("prompt")
                .arg(prompt)
                .arg("response")
                .arg(encoded(&value))
                .arg("prompt_vector")
                .arg(vector_bytes(&vector))
                .arg("inserted_at")
                .arg("1700000000.5")
                .arg("updated_at")
                .arg("1700000000.5")
                .arg("litellm_cache_key")
                .arg("scope-a"),
            Ok(7),
        ),
        MockCmd::new(
            search_command(INDEX, "scope\\-a", &vector),
            Ok(search_result(hit_fields(
                "scope-a",
                "0.05",
                encoded(&value),
            ))),
        ),
    ])
    .assert_all_commands_consumed();
    let (embedder, _) = FakeEmbedder::new(&[]);
    let cache = RedisSemanticCache::with_connection(connection, embedder, config())
        .with_clock(|| 1700000000.5);
    let context = SemanticCacheContext {
        scope: Some("scope-a".into()),
        ..messages_context(vec![json!({"role": "user", "content": prompt})])
    };

    cache.set_cache("key1", value.clone(), &context).unwrap();
    assert_eq!(cache.get_cache("key1", &context).unwrap(), Some(value));
}

#[test]
fn incompatible_schema_falls_back_to_isolated_index() {
    let prompt = "hello prompt";
    let tag = "key1";
    let isolated = format!("{INDEX}_isolated");
    let connection = MockRedisConnection::new([
        MockCmd::new(redis::cmd("FT.INFO").arg(INDEX), Ok(unscoped_info(3))),
        MockCmd::new(
            redis::cmd("FT.INFO").arg(&isolated),
            Err::<redis::Value, _>(unknown_index_error()),
        ),
        MockCmd::new(create_index_command(&isolated, 3), Ok("OK")),
        MockCmd::new(
            redis::cmd("HSET")
                .arg(format!("{isolated}:{}", entry_id(prompt, tag)))
                .arg("entry_id")
                .arg(entry_id(prompt, tag))
                .arg("prompt")
                .arg(prompt)
                .arg("response")
                .arg(encoded(&entry()))
                .arg("prompt_vector")
                .arg(vector_bytes(&[0.1f32, 0.2, 0.3]))
                .arg("inserted_at")
                .arg("1700000000.5")
                .arg("updated_at")
                .arg("1700000000.5")
                .arg("litellm_cache_key")
                .arg(tag),
            Ok(7),
        ),
    ])
    .assert_all_commands_consumed();
    let (embedder, _) = FakeEmbedder::new(&[]);
    let cache = RedisSemanticCache::with_connection(connection, embedder, config())
        .with_clock(|| 1700000000.5);

    cache
        .set_cache(
            tag,
            entry(),
            &messages_context(vec![json!({"role": "user", "content": prompt})]),
        )
        .unwrap();
}

#[test]
fn create_index_race_rechecks_schema_and_stores() {
    let prompt = "hello prompt";
    let tag = "key1";
    let connection = MockRedisConnection::new([
        MockCmd::new(
            redis::cmd("FT.INFO").arg(INDEX),
            Err::<redis::Value, _>(unknown_index_error()),
        ),
        MockCmd::new(
            create_index_command(INDEX, 3),
            Err::<&str, _>(redis::RedisError::from((
                redis::ErrorKind::Extension,
                "Index already exists",
            ))),
        ),
        MockCmd::new(redis::cmd("FT.INFO").arg(INDEX), Ok(compatible_info(3))),
        MockCmd::new(
            redis::cmd("HSET")
                .arg(format!("{INDEX}:{}", entry_id(prompt, tag)))
                .arg("entry_id")
                .arg(entry_id(prompt, tag))
                .arg("prompt")
                .arg(prompt)
                .arg("response")
                .arg(encoded(&entry()))
                .arg("prompt_vector")
                .arg(vector_bytes(&[0.1f32, 0.2, 0.3]))
                .arg("inserted_at")
                .arg("1700000000.5")
                .arg("updated_at")
                .arg("1700000000.5")
                .arg("litellm_cache_key")
                .arg(tag),
            Ok(7),
        ),
    ])
    .assert_all_commands_consumed();
    let (embedder, _) = FakeEmbedder::new(&[]);
    let cache = RedisSemanticCache::with_connection(connection, embedder, config())
        .with_clock(|| 1700000000.5);

    cache
        .set_cache(
            tag,
            entry(),
            &messages_context(vec![json!({"role": "user", "content": prompt})]),
        )
        .unwrap();
}

#[test]
fn wrong_distance_metric_falls_back_to_isolated_index() {
    let prompt = "hello prompt";
    let tag = "key1";
    let isolated = format!("{INDEX}_isolated");
    let connection = MockRedisConnection::new([
        MockCmd::new(
            redis::cmd("FT.INFO").arg(INDEX),
            Ok(info_with_vector(vector_attribute_with(3, "FLOAT32", "L2"))),
        ),
        MockCmd::new(
            redis::cmd("FT.INFO").arg(&isolated),
            Err::<redis::Value, _>(unknown_index_error()),
        ),
        MockCmd::new(create_index_command(&isolated, 3), Ok("OK")),
        MockCmd::new(
            redis::cmd("HSET")
                .arg(format!("{isolated}:{}", entry_id(prompt, tag)))
                .arg("entry_id")
                .arg(entry_id(prompt, tag))
                .arg("prompt")
                .arg(prompt)
                .arg("response")
                .arg(encoded(&entry()))
                .arg("prompt_vector")
                .arg(vector_bytes(&[0.1f32, 0.2, 0.3]))
                .arg("inserted_at")
                .arg("1700000000.5")
                .arg("updated_at")
                .arg("1700000000.5")
                .arg("litellm_cache_key")
                .arg(tag),
            Ok(7),
        ),
    ])
    .assert_all_commands_consumed();
    let (embedder, _) = FakeEmbedder::new(&[]);
    let cache = RedisSemanticCache::with_connection(connection, embedder, config())
        .with_clock(|| 1700000000.5);

    cache
        .set_cache(
            tag,
            entry(),
            &messages_context(vec![json!({"role": "user", "content": prompt})]),
        )
        .unwrap();
}

#[test]
fn tag_special_characters_are_escaped_in_search_filter() {
    let vector = vec![0.1f32, 0.2, 0.3];
    let tag = "a:b, c|d";
    let connection = MockRedisConnection::new([
        MockCmd::new(redis::cmd("FT.INFO").arg(INDEX), Ok(compatible_info(3))),
        MockCmd::new(
            search_command(INDEX, "a\\:b\\,\\ c\\|d", &vector),
            Ok(empty_result()),
        ),
    ])
    .assert_all_commands_consumed();
    let (embedder, _) = FakeEmbedder::new(&[]);
    let cache = RedisSemanticCache::with_connection(connection, embedder, config());

    assert_eq!(
        cache
            .get_cache(
                tag,
                &messages_context(vec![json!({"role": "user", "content": "hello prompt"})])
            )
            .unwrap(),
        None
    );
}

#[test]
fn prompt_extraction_matches_python_message_and_input_shapes() {
    let vector = vec![0.1f32, 0.2, 0.3];
    let lookups = 5;
    let mut commands = vec![MockCmd::new(
        redis::cmd("FT.INFO").arg(INDEX),
        Ok(compatible_info(3)),
    )];
    for _ in 0..lookups {
        commands.push(MockCmd::new(
            search_command(INDEX, "key1", &vector),
            Ok(empty_result()),
        ));
    }
    let connection = MockRedisConnection::new(commands).assert_all_commands_consumed();
    let (embedder, calls) = FakeEmbedder::new(&[]);
    let cache = RedisSemanticCache::with_connection(connection, embedder, config());

    cache
        .get_cache(
            "key1",
            &messages_context(vec![
                json!({"role": "user", "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]}),
                json!({"role": "assistant", "content": "reply"}),
            ]),
        )
        .unwrap();
    cache
        .get_cache(
            "key1",
            &SemanticCacheContext {
                input: Some(json!("  plain input  ")),
                ..Default::default()
            },
        )
        .unwrap();
    cache
        .get_cache(
            "key1",
            &SemanticCacheContext {
                input: Some(
                    json!([{"content": [{"type": "input_text", "text": "nested"}]}, "tail"]),
                ),
                ..Default::default()
            },
        )
        .unwrap();
    cache
        .get_cache(
            "key1",
            &SemanticCacheContext {
                input: Some(json!({"output_text": "  result text  "})),
                ..Default::default()
            },
        )
        .unwrap();
    cache
        .get_cache(
            "key1",
            &messages_context(vec![json!({
                "role": "user",
                "content": "question",
                "search_results": [{"source": "src", "title": "t", "content": [{"text": "found"}], "citations": {"a": 1}}],
            })]),
        )
        .unwrap();

    assert_eq!(
        *calls.lock().unwrap(),
        vec![
            "firstsecondreply",
            "plain input",
            "nested\ntail",
            "result text",
            "questionsrctfound{\"a\":1}",
        ]
    );
}

#[test]
fn ttl_passes_through_context_only() {
    let (embedder, _) = FakeEmbedder::new(&[]);
    let cache = RedisSemanticCache::with_connection(
        MockRedisConnection::new(Vec::<MockCmd>::new()),
        embedder,
        config(),
    );
    assert_eq!(cache.get_ttl(&SemanticCacheContext::default()), None);
    assert_eq!(
        cache.get_ttl(&SemanticCacheContext {
            ttl: Some(Duration::from_secs(9)),
            ..Default::default()
        }),
        Some(Duration::from_secs(9))
    );
}

#[tokio::test]
async fn async_paths_embed_then_run_blocking_redis_work() {
    let vector = vec![0.1f32, 0.2, 0.3];
    let prompt = "hello prompt";
    let tag = "key1";
    let hash_key = format!("{INDEX}:{}", entry_id(prompt, tag));
    let value = entry();
    let connection = MockRedisConnection::new([
        MockCmd::new(redis::cmd("FT.INFO").arg(INDEX), Ok(compatible_info(3))),
        MockCmd::new(
            redis::cmd("HSET")
                .arg(&hash_key)
                .arg("entry_id")
                .arg(entry_id(prompt, tag))
                .arg("prompt")
                .arg(prompt)
                .arg("response")
                .arg(encoded(&value))
                .arg("prompt_vector")
                .arg(vector_bytes(&vector))
                .arg("inserted_at")
                .arg("1700000000.5")
                .arg("updated_at")
                .arg("1700000000.5")
                .arg("litellm_cache_key")
                .arg(tag),
            Ok(7),
        ),
        MockCmd::new(
            search_command(INDEX, tag, &vector),
            Ok(search_result(hit_fields(tag, "0.05", encoded(&value)))),
        ),
    ])
    .assert_all_commands_consumed();
    let (embedder, _) = FakeEmbedder::new(&[]);
    let cache = RedisSemanticCache::with_connection(connection, embedder, config())
        .with_clock(|| 1700000000.5);
    let context = messages_context(vec![json!({"role": "user", "content": prompt})]);

    cache
        .async_set_cache(tag, value.clone(), context.clone())
        .await
        .unwrap();
    assert_eq!(
        cache.async_get_cache(tag, &context).await.unwrap(),
        Some(value)
    );
}

#[test]
fn shared_base_index_across_dimensions_replaces_the_isolated_index() {
    // Pins parity with Python's `_isolated` + overwrite=True flow.
    let prompt = "shared prompt";
    let tag = "key1";
    let isolated = format!("{INDEX}_isolated");
    let value = entry();
    let context = || messages_context(vec![json!({"role": "user", "content": prompt})]);
    let store_hash = |index: &str, vector: &[f32]| {
        MockCmd::new(
            redis::cmd("HSET")
                .arg(format!("{index}:{}", entry_id(prompt, tag)))
                .arg("entry_id")
                .arg(entry_id(prompt, tag))
                .arg("prompt")
                .arg(prompt)
                .arg("response")
                .arg(encoded(&value))
                .arg("prompt_vector")
                .arg(vector_bytes(vector))
                .arg("inserted_at")
                .arg("1700000000.5")
                .arg("updated_at")
                .arg("1700000000.5")
                .arg("litellm_cache_key")
                .arg(tag),
            Ok(7),
        )
    };

    let vector_a = vec![0.1f32; 8];
    let connection_a = MockRedisConnection::new([
        MockCmd::new(
            redis::cmd("FT.INFO").arg(INDEX),
            Err::<redis::Value, _>(unknown_index_error()),
        ),
        MockCmd::new(create_index_command(INDEX, 8), Ok("OK")),
        store_hash(INDEX, &vector_a),
    ])
    .assert_all_commands_consumed();
    let (embedder_a, _) = FakeEmbedder::new(&[(prompt, &vector_a)]);
    let worker_a = RedisSemanticCache::with_connection(connection_a, embedder_a, config())
        .with_clock(|| 1700000000.5);
    worker_a.set_cache(tag, value.clone(), &context()).unwrap();

    let vector_b = vec![0.2f32; 4];
    let connection_b = MockRedisConnection::new([
        MockCmd::new(redis::cmd("FT.INFO").arg(INDEX), Ok(compatible_info(8))),
        MockCmd::new(
            redis::cmd("FT.INFO").arg(&isolated),
            Err::<redis::Value, _>(unknown_index_error()),
        ),
        MockCmd::new(create_index_command(&isolated, 4), Ok("OK")),
        store_hash(&isolated, &vector_b),
        MockCmd::new(
            search_command(&isolated, tag, &vector_b),
            Ok(search_result(hit_fields(tag, "0.0", encoded(&value)))),
        ),
        MockCmd::new(
            search_command(&isolated, tag, &vector_b),
            Err::<redis::Value, _>(redis::RedisError::from((
                redis::ErrorKind::Extension,
                "Vector dimension mismatch",
            ))),
        ),
    ])
    .assert_all_commands_consumed();
    let (embedder_b, _) = FakeEmbedder::new(&[(prompt, &vector_b)]);
    let worker_b = RedisSemanticCache::with_connection(connection_b, embedder_b, config())
        .with_clock(|| 1700000000.5);
    worker_b.set_cache(tag, value.clone(), &context()).unwrap();
    assert_eq!(
        worker_b.get_cache(tag, &context()).unwrap(),
        Some(value.clone())
    );

    let vector_c = vec![0.3f32; 16];
    let connection_c = MockRedisConnection::new([
        MockCmd::new(redis::cmd("FT.INFO").arg(INDEX), Ok(compatible_info(8))),
        MockCmd::new(redis::cmd("FT.INFO").arg(&isolated), Ok(compatible_info(4))),
        MockCmd::new(redis::cmd("FT.DROPINDEX").arg(&isolated), Ok("OK")),
        MockCmd::new(create_index_command(&isolated, 16), Ok("OK")),
        store_hash(&isolated, &vector_c),
    ])
    .assert_all_commands_consumed();
    let (embedder_c, _) = FakeEmbedder::new(&[(prompt, &vector_c)]);
    let worker_c = RedisSemanticCache::with_connection(connection_c, embedder_c, config())
        .with_clock(|| 1700000000.5);
    worker_c.set_cache(tag, value.clone(), &context()).unwrap();

    assert_eq!(
        worker_b.get_cache(tag, &context()).unwrap_err(),
        Error::Unavailable
    );
}

#[test]
fn live_shared_index_is_replaced_across_dimensions() {
    let Ok(url) = std::env::var("LITELLM_REDIS_STACK_URL") else {
        return;
    };
    // Pins parity with Python's `_isolated` + overwrite=True flow.
    let base = format!("rust_semantic_shared_{}", std::process::id());
    let isolated = format!("{base}_isolated");
    let prompt = "shared live prompt";
    let tag = "key1";
    let context = || messages_context(vec![json!({"role": "user", "content": prompt})]);
    let value = entry();
    let worker = |vector: Vec<f32>| {
        let (embedder, _) = FakeEmbedder::new(&[(prompt, vector.as_slice())]);
        RedisSemanticCache::new(
            &url,
            embedder,
            RedisSemanticConfig {
                index_name: base.clone(),
                similarity_threshold: 0.9,
            },
        )
        .unwrap()
    };

    let worker_a = worker(vec![0.1f32; 8]);
    worker_a.set_cache(tag, value.clone(), &context()).unwrap();

    let worker_b = worker(vec![0.2f32; 4]);
    worker_b.set_cache(tag, value.clone(), &context()).unwrap();
    assert_eq!(
        worker_b.get_cache(tag, &context()).unwrap(),
        Some(value.clone())
    );

    let worker_c = worker(vec![0.3f32; 16]);
    worker_c.set_cache(tag, value.clone(), &context()).unwrap();

    assert_eq!(
        worker_b.get_cache(tag, &context()).unwrap_err(),
        Error::Unavailable
    );

    let mut connection = redis::Client::open(url).unwrap().get_connection().unwrap();
    for index in [&base, &isolated] {
        let _: Result<(), _> = redis::cmd("FT.DROPINDEX")
            .arg(index)
            .arg("DD")
            .query(&mut connection);
    }
}

#[test]
fn live_store_lookup_and_ttl_against_redis_stack() {
    let Ok(url) = std::env::var("LITELLM_REDIS_STACK_URL") else {
        return;
    };
    let vector = vec![0.1f32, 0.2, 0.3, 0.4];
    let prompt = "rust semantic cache live prompt";
    let tag = "live-key";
    let index_name = format!("rust_semantic_test_{}", std::process::id());
    let (embedder, _) = FakeEmbedder::new(&[(prompt, &vector)]);
    let cache = RedisSemanticCache::new(
        &url,
        embedder,
        RedisSemanticConfig {
            index_name: index_name.clone(),
            similarity_threshold: 0.9,
        },
    )
    .unwrap();
    let context = SemanticCacheContext {
        ttl: Some(Duration::from_secs(120)),
        ..messages_context(vec![json!({"role": "user", "content": prompt})])
    };
    let value = entry();

    cache.set_cache(tag, value.clone(), &context).unwrap();
    assert_eq!(cache.get_cache(tag, &context).unwrap(), Some(value));
    assert_eq!(cache.get_cache("other-key", &context).unwrap(), None);

    let mut connection = redis::Client::open(url).unwrap().get_connection().unwrap();
    let ttl: i64 = redis::Commands::ttl(
        &mut connection,
        format!("{index_name}:{}", entry_id(prompt, tag)),
    )
    .unwrap();
    assert!(
        ttl > 0,
        "expected stored hash to carry an expiry, got {ttl}"
    );
}
