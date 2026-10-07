use std::{sync::Arc, time::Duration};

use litellm_cache::{
    BaseCache, BatchCache, BatchEntry, CacheContext, Error, ExactCacheContext, FlushCache,
    JsonCodec,
};
use litellm_cache_gcs::{GcsCache, GcsConfig, StaticTokenSource, TokenSource, key_prefix};
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_bytes, header, method, path, query_param},
};

fn config(server: &MockServer, gcs_path: Option<&str>) -> GcsConfig {
    GcsConfig {
        bucket_name: "bucket".into(),
        gcs_path: gcs_path.map(str::to_string),
        path_service_account: None,
        endpoint: server.uri(),
    }
}

fn cache(server: &MockServer, gcs_path: Option<&str>) -> GcsCache<JsonCodec<serde_json::Value>> {
    GcsCache::with_token_source(
        config(server, gcs_path),
        JsonCodec::new(),
        Arc::new(StaticTokenSource("tok".into())),
    )
    .unwrap()
}

#[tokio::test]
async fn set_writes_encoded_object_and_headers() {
    let server = MockServer::start().await;
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
    cache(&server, Some("cache/"))
        .set_cache(
            "team:a b/c",
            json!({"value": "entry"}),
            &ExactCacheContext::default(),
        )
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].url.query(),
        Some("uploadType=media&name=cache%2Fteam%3Aa%20b%2Fc")
    );
}

#[tokio::test]
async fn get_maps_statuses_and_decode_failures() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/storage/v1/b/bucket/o/hit"))
        .and(query_param("alt", "media"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"value": "entry"})))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/storage/v1/b/bucket/o/missing"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/storage/v1/b/bucket/o/server-error"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/storage/v1/b/bucket/o/invalid"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&server)
        .await;

    let cache = cache(&server, None);
    assert_eq!(
        cache
            .get_cache("hit", &ExactCacheContext::default())
            .unwrap(),
        Some(json!({"value": "entry"}))
    );
    assert_eq!(
        cache
            .get_cache("missing", &ExactCacheContext::default())
            .unwrap(),
        None
    );
    assert_eq!(
        cache
            .get_cache("server-error", &ExactCacheContext::default())
            .unwrap_err(),
        Error::Unavailable
    );
    assert_eq!(
        cache
            .get_cache("invalid", &ExactCacheContext::default())
            .unwrap_err(),
        Error::InvalidEntry
    );
}

#[test]
fn key_prefix_normalizes_paths() {
    assert_eq!(key_prefix(None), "");
    assert_eq!(key_prefix(Some("a/b/")), "a/b/");
    assert_eq!(key_prefix(Some("a/b")), "a/b/");
    assert_eq!(key_prefix(Some("")), "");
}

#[tokio::test]
async fn object_names_use_python_quote_encoding() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/upload/storage/v1/b/bucket/o"))
        .and(query_param("uploadType", "media"))
        .respond_with(ResponseTemplate::new(200))
        .expect(2)
        .mount(&server)
        .await;
    let cache = cache(&server, Some("p/"));
    cache
        .set_cache(
            "a~b-c_d.e/f g%h",
            json!({"value": "punctuation"}),
            &ExactCacheContext::default(),
        )
        .unwrap();
    cache
        .set_cache(
            "ключ",
            json!({"value": "utf8"}),
            &ExactCacheContext::default(),
        )
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let queries: Vec<_> = requests
        .iter()
        .filter_map(|request| request.url.query())
        .collect();
    assert!(queries.contains(&"uploadType=media&name=p%2Fa~b-c_d.e%2Ff%20g%25h"));
    assert!(queries.contains(&"uploadType=media&name=p%2F%D0%BA%D0%BB%D1%8E%D1%87"));
}

#[tokio::test]
async fn ignores_ttl_and_writes_pipeline_concurrently() {
    let server = MockServer::start().await;
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
    let cache = cache(&server, None);
    assert_eq!(cache.get_ttl(&ExactCacheContext::default()), None);
    assert_eq!(
        cache.get_ttl(&ExactCacheContext::default().with_ttl(Some(Duration::from_secs(5)))),
        None
    );
    cache
        .async_set_cache_pipeline(
            vec![
                ("one".into(), json!({"key": "one"})),
                ("two".into(), json!({"key": "two"})),
                ("three".into(), json!({"key": "three"})),
            ],
            ExactCacheContext::default().with_ttl(Some(Duration::from_secs(5))),
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn async_batch_get_preserves_hits_misses_and_invalid_entries() {
    let server = MockServer::start().await;
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

    assert_eq!(
        cache(&server, None)
            .async_batch_get_cache(
                vec!["hit".into(), "missing".into(), "invalid".into()],
                ExactCacheContext::default(),
            )
            .await
            .unwrap(),
        vec![
            BatchEntry::Hit(json!({"value": "entry"})),
            BatchEntry::Miss,
            BatchEntry::Invalid,
        ]
    );
}

#[tokio::test]
async fn lifecycle_operations_are_noops_and_connection_test_is_unsupported() {
    let server = MockServer::start().await;
    let cache = cache(&server, None);
    assert_eq!(cache.flush_cache(), Ok(()));
    assert_eq!(cache.disconnect().await, Ok(()));
    assert_eq!(
        cache.test_connection().await,
        Err(Error::UnsupportedOperation)
    );
}

#[test]
fn sync_operations_work_without_an_active_runtime() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let server = runtime.block_on(MockServer::start());
    runtime.block_on(
        Mock::given(method("POST"))
            .and(path("/upload/storage/v1/b/bucket/o"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server),
    );
    runtime.block_on(
        Mock::given(method("GET"))
            .and(path("/storage/v1/b/bucket/o/key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"value": "entry"})))
            .mount(&server),
    );
    let cache = cache(&server, None);
    cache
        .set_cache(
            "key",
            json!({"value": "entry"}),
            &ExactCacheContext::default(),
        )
        .unwrap();
    assert_eq!(
        cache
            .get_cache("key", &ExactCacheContext::default())
            .unwrap(),
        Some(json!({"value": "entry"}))
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn sync_operations_work_inside_a_multi_thread_runtime() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/upload/storage/v1/b/bucket/o"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/storage/v1/b/bucket/o/key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"value": "entry"})))
        .mount(&server)
        .await;
    let cache = cache(&server, None);
    cache
        .set_cache(
            "key",
            json!({"value": "entry"}),
            &ExactCacheContext::default(),
        )
        .unwrap();
    assert_eq!(
        cache
            .get_cache("key", &ExactCacheContext::default())
            .unwrap(),
        Some(json!({"value": "entry"}))
    );
}

struct FailingTokenSource;

impl TokenSource for FailingTokenSource {
    fn bearer_token(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, Error>> + Send + '_>>
    {
        Box::pin(async { Err(Error::Unavailable) })
    }
}

#[tokio::test]
async fn token_source_failure_skips_http() {
    let server = MockServer::start().await;
    let cache = GcsCache::with_token_source(
        config(&server, None),
        JsonCodec::<serde_json::Value>::new(),
        Arc::new(FailingTokenSource),
    )
    .unwrap();
    assert_eq!(
        cache
            .get_cache("key", &ExactCacheContext::default())
            .unwrap_err(),
        Error::Unavailable
    );
    assert_eq!(server.received_requests().await.unwrap().len(), 0);
}
