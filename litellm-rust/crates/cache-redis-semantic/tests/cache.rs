mod support;

use std::time::Duration;

use litellm_cache::{
    BaseCache, Error, JsonCodec, SemanticCacheContext,
    semantic::{SemanticCache, SemanticLookup},
};
use litellm_cache_redis_semantic::{DEFAULT_INDEX_NAME, RedisSemanticCache, RedisSemanticConfig};
use redis_test::{MockCmd, MockRedisConnection};
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use support::FakeEmbedder;

const INDEX: &str = DEFAULT_INDEX_NAME;
const PROMPT: &str = "hello prompt";
const CLOCK: fn() -> f64 = || 1700000000.5;
const VECTOR: [f32; 3] = [0.1, 0.2, 0.3];

type MockCache = RedisSemanticCache<FakeEmbedder, JsonCodec<Value>, MockRedisConnection>;

#[fixture]
fn config() -> RedisSemanticConfig {
    RedisSemanticConfig {
        index_name: INDEX.into(),
        similarity_threshold: 0.9,
    }
}

#[fixture]
fn entry() -> Value {
    json!({"timestamp": 1.0, "response": {"answer": "yes"}})
}

#[fixture]
fn context() -> SemanticCacheContext {
    messages_context(vec![json!({"role": "user", "content": PROMPT})])
}

fn messages_context(messages: Vec<Value>) -> SemanticCacheContext {
    SemanticCacheContext {
        messages: Some(Value::Array(messages)),
        ..Default::default()
    }
}

fn cache(commands: Vec<MockCmd>, embedder: FakeEmbedder) -> MockCache {
    RedisSemanticCache::with_connection(
        MockRedisConnection::new(commands).assert_all_commands_consumed(),
        embedder,
        JsonCodec::new(),
        config(),
    )
    .with_clock(CLOCK)
}

fn encoded(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).unwrap()
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

fn info_missing(index: &str) -> MockCmd {
    MockCmd::new(
        redis::cmd("FT.INFO").arg(index),
        Err::<redis::Value, _>(unknown_index_error()),
    )
}

fn info(index: &str, value: redis::Value) -> MockCmd {
    MockCmd::new(redis::cmd("FT.INFO").arg(index), Ok(value))
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
    info_with_vector(vector_attribute_with(dims, "FLOAT32", "COSINE"))
}

fn unscoped_info(dims: i64) -> redis::Value {
    index_info(vec![
        attribute("prompt", "TEXT", vec![]),
        attribute("response", "TEXT", vec![]),
        attribute("inserted_at", "NUMERIC", vec![]),
        attribute("updated_at", "NUMERIC", vec![]),
        vector_attribute_with(dims, "FLOAT32", "COSINE"),
    ])
}

fn create_index(name: &str, dims: usize) -> MockCmd {
    MockCmd::new(create_index_command(name, dims), Ok("OK"))
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

fn hset(index: &str, prompt: &str, tag: &str, vector: &[f32], value: &Value) -> MockCmd {
    MockCmd::new(
        redis::cmd("HSET")
            .arg(format!("{index}:{}", entry_id(prompt, tag)))
            .arg("entry_id")
            .arg(entry_id(prompt, tag))
            .arg("prompt")
            .arg(prompt)
            .arg("response")
            .arg(encoded(value))
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
}

fn search(
    index: &str,
    tag: &str,
    vector: &[f32],
    reply: redis::RedisResult<redis::Value>,
) -> MockCmd {
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
    MockCmd::new(command, reply)
}

fn hit(tag: &str, distance: &str, response: Vec<u8>) -> redis::Value {
    redis::Value::Array(vec![
        redis::Value::Int(1),
        s("litellm_semantic_cache_index:stored-id"),
        redis::Value::Array(vec![
            s("entry_id"),
            s("stored-id"),
            s("prompt"),
            s(PROMPT),
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
        ]),
    ])
}

fn empty_result() -> redis::Value {
    redis::Value::Array(vec![redis::Value::Int(0)])
}

#[rstest]
#[case::creates_index_and_expires(false, Some(Duration::from_secs(5)), Some(5))]
#[case::existing_index_without_ttl(true, None, None)]
#[case::fractional_ttl_rounds_up(true, Some(Duration::from_millis(1500)), Some(2))]
fn store_writes_the_redisvl_hash(
    #[case] index_exists: bool,
    #[case] ttl: Option<Duration>,
    #[case] expire: Option<u64>,
    entry: Value,
    context: SemanticCacheContext,
) {
    let hash_key = format!("{INDEX}:{}", entry_id(PROMPT, "key1"));
    let mut commands = if index_exists {
        vec![info(INDEX, compatible_info(3))]
    } else {
        vec![info_missing(INDEX), create_index(INDEX, 3)]
    };
    commands.push(hset(INDEX, PROMPT, "key1", &VECTOR, &entry));
    commands.extend(
        expire.map(|seconds| MockCmd::new(redis::cmd("EXPIRE").arg(&hash_key).arg(seconds), Ok(1))),
    );
    let cache = cache(commands, FakeEmbedder::new(&[]));

    cache
        .set_cache("key1", entry, &SemanticCacheContext { ttl, ..context })
        .unwrap();
}

#[rstest]
#[case::below_distance_threshold("key1", "0.05", None, Ok(Some(entry())))]
#[case::above_distance_threshold("key1", "0.5", None, Ok(None))]
#[case::other_cache_key("other", "0.05", None, Ok(None))]
#[case::malformed_response("key1", "0.05", Some(b"not json!".as_slice()), Err(Error::InvalidEntry))]
fn lookup_applies_threshold_scope_and_codec(
    #[case] stored_tag: &str,
    #[case] distance: &str,
    #[case] response: Option<&[u8]>,
    #[case] expected: Result<Option<Value>, Error>,
    entry: Value,
    context: SemanticCacheContext,
) {
    let response = response.map_or_else(|| encoded(&entry), <[u8]>::to_vec);
    let cache = cache(
        vec![
            info(INDEX, compatible_info(3)),
            search(
                INDEX,
                "key1",
                &VECTOR,
                Ok(hit(stored_tag, distance, response)),
            ),
        ],
        FakeEmbedder::new(&[]),
    );

    assert_eq!(cache.get_cache("key1", &context), expected);
}

#[rstest]
#[case::hit(context(), Some(hit("key1", "0.05", encoded(&entry()))), Some(entry()), Some(1.0 - 0.05))]
#[case::beyond_distance_threshold(context(), Some(hit("key1", "0.5", encoded(&entry()))), None, Some(0.0))]
#[case::no_results(context(), Some(empty_result()), None, Some(0.0))]
#[case::other_cache_key(context(), Some(hit("other", "0.05", encoded(&entry()))), None, Some(0.0))]
#[case::no_prompt(SemanticCacheContext::default(), None, None, Some(0.0))]
#[tokio::test]
async fn lookup_reports_python_semantic_similarity(
    #[case] context: SemanticCacheContext,
    #[case] reply: Option<redis::Value>,
    #[case] value: Option<Value>,
    #[case] similarity: Option<f64>,
    #[values(false, true)] use_async: bool,
) {
    let commands = reply.map_or_else(Vec::new, |reply| {
        vec![
            info(INDEX, compatible_info(3)),
            search(INDEX, "key1", &VECTOR, Ok(reply)),
        ]
    });
    let cache = cache(commands, FakeEmbedder::new(&[]));

    let lookup = if use_async {
        cache
            .async_get_cache_with_similarity("key1", &context)
            .await
    } else {
        cache.get_cache_with_similarity("key1", &context)
    };

    assert_eq!(lookup, Ok(SemanticLookup { value, similarity }));
}

#[rstest]
#[tokio::test]
async fn missing_prompt_is_a_noop_that_never_embeds(entry: Value) {
    let embedder = FakeEmbedder::new(&[]);
    let calls = embedder.calls.clone();
    let cache = cache(Vec::new(), embedder);
    let context = SemanticCacheContext::default();

    cache.set_cache("key1", entry.clone(), &context).unwrap();
    assert_eq!(cache.get_cache("key1", &context).unwrap(), None);
    cache
        .async_set_cache("key1", entry, context.clone())
        .await
        .unwrap();
    assert_eq!(cache.async_get_cache("key1", &context).await.unwrap(), None);
    assert!(calls.lock().unwrap().is_empty());
}

#[rstest]
fn scope_overrides_key_as_filter_tag(entry: Value, context: SemanticCacheContext) {
    let cache = cache(
        vec![
            info(INDEX, compatible_info(3)),
            hset(INDEX, PROMPT, "scope-a", &VECTOR, &entry),
            search(
                INDEX,
                "scope\\-a",
                &VECTOR,
                Ok(hit("scope-a", "0.05", encoded(&entry))),
            ),
        ],
        FakeEmbedder::new(&[]),
    );
    let context = SemanticCacheContext {
        scope: Some("scope-a".into()),
        ..context
    };

    cache.set_cache("key1", entry.clone(), &context).unwrap();
    assert_eq!(cache.get_cache("key1", &context).unwrap(), Some(entry));
}

#[rstest]
#[case::unscoped_schema(unscoped_info(3))]
#[case::wrong_distance_metric(info_with_vector(vector_attribute_with(3, "FLOAT32", "L2")))]
#[case::wrong_data_type(info_with_vector(vector_attribute_with(3, "FLOAT64", "COSINE")))]
fn incompatible_schema_falls_back_to_isolated_index(
    #[case] base_info: redis::Value,
    entry: Value,
    context: SemanticCacheContext,
) {
    let isolated = format!("{INDEX}_isolated");
    let cache = cache(
        vec![
            info(INDEX, base_info),
            info_missing(&isolated),
            create_index(&isolated, 3),
            hset(&isolated, PROMPT, "key1", &VECTOR, &entry),
        ],
        FakeEmbedder::new(&[]),
    );

    cache.set_cache("key1", entry, &context).unwrap();
}

#[rstest]
fn create_index_race_rechecks_schema_and_stores(entry: Value, context: SemanticCacheContext) {
    let cache = cache(
        vec![
            info_missing(INDEX),
            MockCmd::new(
                create_index_command(INDEX, 3),
                Err::<&str, _>(redis::RedisError::from((
                    redis::ErrorKind::Extension,
                    "Index already exists",
                ))),
            ),
            info(INDEX, compatible_info(3)),
            hset(INDEX, PROMPT, "key1", &VECTOR, &entry),
        ],
        FakeEmbedder::new(&[]),
    );

    cache.set_cache("key1", entry, &context).unwrap();
}

#[rstest]
#[case::punctuation_and_spaces("a:b, c|d", "a\\:b\\,\\ c\\|d")]
#[case::braces_and_dots("{x}.y", "\\{x\\}\\.y")]
#[case::plain("key1", "key1")]
fn tag_special_characters_are_escaped_in_search_filter(
    #[case] tag: &str,
    #[case] escaped: &str,
    context: SemanticCacheContext,
) {
    let cache = cache(
        vec![
            info(INDEX, compatible_info(3)),
            search(INDEX, escaped, &VECTOR, Ok(empty_result())),
        ],
        FakeEmbedder::new(&[]),
    );

    assert_eq!(cache.get_cache(tag, &context).unwrap(), None);
}

#[rstest]
#[case::content_parts(
    SemanticCacheContext {
        messages: Some(json!([
            {"role": "user", "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]},
            {"role": "assistant", "content": "reply"},
        ])),
        ..Default::default()
    },
    "firstsecondreply"
)]
#[case::responses_string_input(
    SemanticCacheContext { input: Some(json!("  plain input  ")), ..Default::default() },
    "plain input"
)]
#[case::responses_nested_input(
    SemanticCacheContext {
        input: Some(json!([{"content": [{"type": "input_text", "text": "nested"}]}, "tail"])),
        ..Default::default()
    },
    "nested\ntail"
)]
#[case::responses_output_text(
    SemanticCacheContext { input: Some(json!({"output_text": "  result text  "})), ..Default::default() },
    "result text"
)]
#[case::search_results(
    messages_context(vec![json!({
        "role": "user",
        "content": "question",
        "search_results": [{"source": "src", "title": "t", "content": [{"text": "found"}], "citations": {"a": 1}}],
    })]),
    "questionsrctfound{\"a\":1}"
)]
#[case::empty_messages_fall_back_to_input(
    SemanticCacheContext { messages: Some(json!([])), input: Some(json!("fallback")), ..Default::default() },
    "fallback"
)]
fn prompt_extraction_matches_python_message_and_input_shapes(
    #[case] context: SemanticCacheContext,
    #[case] prompt: &str,
) {
    let embedder = FakeEmbedder::new(&[]);
    let calls = embedder.calls.clone();
    let cache = cache(
        vec![
            info(INDEX, compatible_info(3)),
            search(INDEX, "key1", &VECTOR, Ok(empty_result())),
        ],
        embedder,
    );

    cache.get_cache("key1", &context).unwrap();

    assert_eq!(*calls.lock().unwrap(), vec![(prompt.to_owned(), None)]);
}

#[rstest]
#[case(None)]
#[case(Some(Duration::from_secs(9)))]
fn ttl_passes_through_context_only(#[case] ttl: Option<Duration>) {
    let cache = cache(Vec::new(), FakeEmbedder::new(&[]));

    assert_eq!(
        cache.get_ttl(&SemanticCacheContext {
            ttl,
            ..Default::default()
        }),
        ttl
    );
}

#[rstest]
#[tokio::test]
async fn async_paths_embed_with_metadata_then_run_blocking_redis_work(
    entry: Value,
    context: SemanticCacheContext,
) {
    let embedder = FakeEmbedder::new(&[]);
    let calls = embedder.calls.clone();
    let cache = cache(
        vec![
            info(INDEX, compatible_info(3)),
            hset(INDEX, PROMPT, "key1", &VECTOR, &entry),
            search(
                INDEX,
                "key1",
                &VECTOR,
                Ok(hit("key1", "0.05", encoded(&entry))),
            ),
        ],
        embedder,
    );
    let context = SemanticCacheContext {
        metadata: Some(json!({"tenant": "team"})),
        ..context
    };

    cache
        .async_set_cache("key1", entry.clone(), context.clone())
        .await
        .unwrap();
    assert_eq!(
        cache.async_get_cache("key1", &context).await.unwrap(),
        Some(entry)
    );
    assert_eq!(
        *calls.lock().unwrap(),
        vec![(PROMPT.to_owned(), context.metadata.clone()); 2]
    );
}

#[rstest]
fn accessors_report_the_config(config: RedisSemanticConfig) {
    let cache = cache(Vec::new(), FakeEmbedder::new(&[]));

    assert_eq!(cache.index_name(), config.index_name);
    assert!((cache.similarity_threshold() - config.similarity_threshold).abs() < 1e-6);
}

#[rstest]
fn shared_base_index_across_dimensions_replaces_the_isolated_index(entry: Value) {
    // Pins parity with Python's `_isolated` + overwrite=True flow.
    let prompt = "shared prompt";
    let isolated = format!("{INDEX}_isolated");
    let context = || messages_context(vec![json!({"role": "user", "content": prompt})]);

    let vector_a = vec![0.1f32; 8];
    let worker_a = cache(
        vec![
            info_missing(INDEX),
            create_index(INDEX, 8),
            hset(INDEX, prompt, "key1", &vector_a, &entry),
        ],
        FakeEmbedder::new(&[(prompt, &vector_a)]),
    );
    worker_a
        .set_cache("key1", entry.clone(), &context())
        .unwrap();

    let vector_b = vec![0.2f32; 4];
    let worker_b = cache(
        vec![
            info(INDEX, compatible_info(8)),
            info_missing(&isolated),
            create_index(&isolated, 4),
            hset(&isolated, prompt, "key1", &vector_b, &entry),
            search(
                &isolated,
                "key1",
                &vector_b,
                Ok(hit("key1", "0.0", encoded(&entry))),
            ),
            search(
                &isolated,
                "key1",
                &vector_b,
                Err(redis::RedisError::from((
                    redis::ErrorKind::Extension,
                    "Vector dimension mismatch",
                ))),
            ),
        ],
        FakeEmbedder::new(&[(prompt, &vector_b)]),
    );
    worker_b
        .set_cache("key1", entry.clone(), &context())
        .unwrap();
    assert_eq!(
        worker_b.get_cache("key1", &context()).unwrap(),
        Some(entry.clone())
    );

    let vector_c = vec![0.3f32; 16];
    let worker_c = cache(
        vec![
            info(INDEX, compatible_info(8)),
            info(&isolated, compatible_info(4)),
            MockCmd::new(redis::cmd("FT.DROPINDEX").arg(&isolated), Ok("OK")),
            create_index(&isolated, 16),
            hset(&isolated, prompt, "key1", &vector_c, &entry),
        ],
        FakeEmbedder::new(&[(prompt, &vector_c)]),
    );
    worker_c
        .set_cache("key1", entry.clone(), &context())
        .unwrap();

    assert_eq!(
        worker_b.get_cache("key1", &context()).unwrap_err(),
        Error::Unavailable
    );
}

#[fixture]
fn redis_stack_url() -> Option<String> {
    std::env::var("LITELLM_REDIS_STACK_URL").ok()
}

fn live_cache(
    url: &str,
    index_name: &str,
    prompt: &str,
    vector: Vec<f32>,
) -> RedisSemanticCache<FakeEmbedder, JsonCodec<Value>> {
    RedisSemanticCache::new(
        url,
        FakeEmbedder::new(&[(prompt, vector.as_slice())]),
        JsonCodec::<Value>::new(),
        RedisSemanticConfig {
            index_name: index_name.to_owned(),
            similarity_threshold: 0.9,
        },
    )
    .unwrap()
}

#[rstest]
fn live_shared_index_is_replaced_across_dimensions(redis_stack_url: Option<String>, entry: Value) {
    let Some(url) = redis_stack_url else {
        return;
    };
    // Pins parity with Python's `_isolated` + overwrite=True flow.
    let base = format!("rust_semantic_shared_{}", std::process::id());
    let isolated = format!("{base}_isolated");
    let prompt = "shared live prompt";
    let context = || messages_context(vec![json!({"role": "user", "content": prompt})]);

    let worker_a = live_cache(&url, &base, prompt, vec![0.1f32; 8]);
    worker_a
        .set_cache("key1", entry.clone(), &context())
        .unwrap();

    let worker_b = live_cache(&url, &base, prompt, vec![0.2f32; 4]);
    worker_b
        .set_cache("key1", entry.clone(), &context())
        .unwrap();
    assert_eq!(
        worker_b.get_cache("key1", &context()).unwrap(),
        Some(entry.clone())
    );

    let worker_c = live_cache(&url, &base, prompt, vec![0.3f32; 16]);
    worker_c
        .set_cache("key1", entry.clone(), &context())
        .unwrap();

    assert_eq!(
        worker_b.get_cache("key1", &context()).unwrap_err(),
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

#[rstest]
fn live_store_lookup_and_ttl_against_redis_stack(redis_stack_url: Option<String>, entry: Value) {
    let Some(url) = redis_stack_url else {
        return;
    };
    let prompt = "rust semantic cache live prompt";
    let index_name = format!("rust_semantic_test_{}", std::process::id());
    let cache = live_cache(&url, &index_name, prompt, vec![0.1, 0.2, 0.3, 0.4]);
    let context = SemanticCacheContext {
        ttl: Some(Duration::from_secs(120)),
        ..messages_context(vec![json!({"role": "user", "content": prompt})])
    };

    cache
        .set_cache("live-key", entry.clone(), &context)
        .unwrap();
    assert_eq!(cache.get_cache("live-key", &context).unwrap(), Some(entry));
    assert_eq!(cache.get_cache("other-key", &context).unwrap(), None);

    let mut connection = redis::Client::open(url).unwrap().get_connection().unwrap();
    let ttl: i64 = redis::Commands::ttl(
        &mut connection,
        format!("{index_name}:{}", entry_id(prompt, "live-key")),
    )
    .unwrap();
    assert!(
        ttl > 0,
        "expected stored hash to carry an expiry, got {ttl}"
    );
}
