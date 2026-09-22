mod support;

use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

use litellm_cache::{
    BaseCache, BatchCache, BatchEntry, CacheContext, DisconnectCache, Error, ExactCacheContext,
    FlushCache,
};
use litellm_cache_gcs::{GcsCache, GcsConfig, TokenSource, key_prefix};
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use support::FakeBucket;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_bytes, header, method, path, query_param},
};

#[fixture]
async fn server() -> MockServer {
    MockServer::start().await
}

fn context() -> ExactCacheContext {
    ExactCacheContext::default()
}

#[rstest]
#[tokio::test]
async fn set_writes_encoded_object_and_headers(#[future(awt)] server: MockServer) {
    Mock::given(method("POST"))
        .and(path("/upload/storage/v1/b/bucket/o"))
        .and(query_param("uploadType", "media"))
        .and(header("authorization", "Bearer tok"))
        .and(header("content-type", "application/json"))
        .and(body_bytes(br#"{"value":"entry"}"#))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;
    support::cache(&server, Some("cache/"))
        .set_cache("team:a b/c", json!({"value": "entry"}), &context())
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].url.query(),
        Some("uploadType=media&name=cache%2Fteam%3Aa%20b%2Fc")
    );
}

#[rstest]
#[case::hit(
    "hit",
    ResponseTemplate::new(200).set_body_json(json!({"value": "entry"})),
    Ok(Some(json!({"value": "entry"})))
)]
#[case::missing("missing", ResponseTemplate::new(404), Ok(None))]
#[case::server_error("server-error", ResponseTemplate::new(500), Err(Error::Unavailable))]
#[case::invalid(
    "invalid",
    ResponseTemplate::new(200).set_body_string("not json"),
    Err(Error::InvalidEntry)
)]
#[tokio::test]
async fn get_maps_statuses_and_decode_failures(
    #[future(awt)] server: MockServer,
    #[case] key: &str,
    #[case] response: ResponseTemplate,
    #[case] expected: Result<Option<Value>, Error>,
) {
    Mock::given(method("GET"))
        .and(path(format!("/storage/v1/b/bucket/o/{key}")))
        .and(query_param("alt", "media"))
        .respond_with(response)
        .mount(&server)
        .await;
    let cache = support::cache(&server, None);
    assert_eq!(cache.get_cache(key, &context()), expected);
    assert_eq!(cache.async_get_cache(key, &context()).await, expected);
}

#[rstest]
#[case::none(None, "")]
#[case::trailing_slash(Some("a/b/"), "a/b/")]
#[case::no_trailing_slash(Some("a/b"), "a/b/")]
#[case::empty(Some(""), "")]
fn key_prefix_normalizes_paths(#[case] gcs_path: Option<&str>, #[case] expected: &str) {
    assert_eq!(key_prefix(gcs_path), expected);
}

#[rstest]
#[tokio::test]
async fn cache_exposes_its_configuration(#[future(awt)] server: MockServer) {
    let cache = GcsCache::new(
        GcsConfig {
            path_service_account: Some("/secrets/sa.json".into()),
            ..support::config(&server, Some("folder"))
        },
        reqwest::Client::new(),
        litellm_cache::JsonCodec::<Value>::new(),
    );
    assert_eq!(cache.bucket_name(), "bucket");
    assert_eq!(cache.key_prefix(), "folder/");
    assert_eq!(cache.path_service_account(), Some("/secrets/sa.json"));
    assert_eq!(cache.object_name("k"), "folder/k");
}

#[rstest]
#[case::punctuation("a~b-c_d.e/f g%h", "uploadType=media&name=p%2Fa~b-c_d.e%2Ff%20g%25h")]
#[case::utf8("ключ", "uploadType=media&name=p%2F%D0%BA%D0%BB%D1%8E%D1%87")]
#[tokio::test]
async fn object_names_use_python_quote_encoding(
    #[future(awt)] server: MockServer,
    #[case] key: &str,
    #[case] query: &str,
) {
    Mock::given(method("POST"))
        .and(path("/upload/storage/v1/b/bucket/o"))
        .and(query_param("uploadType", "media"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;
    support::cache(&server, Some("p/"))
        .async_set_cache(key, json!({"value": key}), context())
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests[0].url.query(), Some(query));
}

#[rstest]
#[tokio::test]
async fn object_names_are_encoded_in_the_download_path(#[future(awt)] server: MockServer) {
    Mock::given(method("GET"))
        .and(path("/storage/v1/b/bucket/o/p%2Fa%3Ab%20c"))
        .and(query_param("alt", "media"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!(1)))
        .expect(2)
        .mount(&server)
        .await;
    let cache = support::cache(&server, Some("p"));
    assert_eq!(cache.get_cache("a:b c", &context()), Ok(Some(json!(1))));
    assert_eq!(
        cache.async_get_cache("a:b c", &context()).await,
        Ok(Some(json!(1)))
    );
}

#[rstest]
#[tokio::test]
async fn ignores_ttl_and_writes_pipeline_concurrently(#[future(awt)] server: MockServer) {
    for key in ["one", "two", "three"] {
        Mock::given(method("POST"))
            .and(path("/upload/storage/v1/b/bucket/o"))
            .and(query_param("uploadType", "media"))
            .and(query_param("name", key))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;
    }
    let cache = support::cache(&server, None);
    let with_ttl = context().with_ttl(Some(Duration::from_secs(5)));
    assert_eq!(cache.get_ttl(&context()), None);
    assert_eq!(cache.get_ttl(&with_ttl), None);
    cache
        .async_set_cache_pipeline(
            vec![
                ("one".into(), json!({"key": "one"})),
                ("two".into(), json!({"key": "two"})),
                ("three".into(), json!({"key": "three"})),
            ],
            with_ttl,
        )
        .await
        .unwrap();
}

#[rstest]
#[tokio::test]
async fn batch_get_preserves_hits_misses_and_invalid_entries(#[future(awt)] server: MockServer) {
    Mock::given(method("GET"))
        .and(path("/storage/v1/b/bucket/o/hit"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"value": "entry"})))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/storage/v1/b/bucket/o/missing"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/storage/v1/b/bucket/o/invalid"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&server)
        .await;
    let cache = support::cache(&server, None);
    let keys = vec!["hit".to_string(), "missing".into(), "invalid".into()];
    let expected = vec![
        BatchEntry::Hit(json!({"value": "entry"})),
        BatchEntry::Miss,
        BatchEntry::Invalid,
    ];

    assert_eq!(cache.batch_get_cache(&keys, &context()).unwrap(), expected);
    assert_eq!(
        cache.async_batch_get_cache(keys, context()).await.unwrap(),
        expected
    );
}

#[rstest]
#[tokio::test]
async fn flush_and_disconnect_are_noops_like_python(#[future(awt)] server: MockServer) {
    let cache = support::cache(&server, None);
    assert_eq!(cache.flush_cache(), Ok(()));
    assert_eq!(cache.async_flush_cache().await, Ok(()));
    assert_eq!(cache.disconnect().await, Ok(()));
    assert!(server.received_requests().await.unwrap().is_empty());
}

fn round_trip(cache: &support::JsonGcsCache) -> Result<Option<Value>, Error> {
    cache.set_cache("key", json!({"value": "entry"}), &context())?;
    cache.get_cache("key", &context())
}

#[rstest]
fn sync_operations_work_without_an_active_runtime() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let server = runtime.block_on(FakeBucket::serve());
    let cache = support::cache(&server, None);
    assert_eq!(round_trip(&cache), Ok(Some(json!({"value": "entry"}))));
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn sync_operations_work_inside_a_multi_thread_runtime() {
    let server = FakeBucket::serve().await;
    let cache = support::cache(&server, None);
    assert_eq!(round_trip(&cache), Ok(Some(json!({"value": "entry"}))));
}

#[rstest]
#[tokio::test]
async fn sync_operations_work_inside_a_current_thread_runtime() {
    let server = FakeBucket::serve().await;
    let cache = support::cache(&server, None);
    assert_eq!(round_trip(&cache), Ok(Some(json!({"value": "entry"}))));
}

struct FailingTokenSource;

impl TokenSource for FailingTokenSource {
    fn bearer_token(&self) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + '_>> {
        Box::pin(async { Err(Error::Unavailable) })
    }
}

#[rstest]
#[tokio::test]
async fn token_source_failure_skips_http(#[future(awt)] server: MockServer) {
    let cache = support::cache_with_token(&server, None, Arc::new(FailingTokenSource));
    assert_eq!(
        cache.get_cache("key", &context()).unwrap_err(),
        Error::Unavailable
    );
    assert_eq!(
        cache
            .async_set_cache("key", json!(1), context())
            .await
            .unwrap_err(),
        Error::Unavailable
    );
    assert_eq!(server.received_requests().await.unwrap().len(), 0);
}
