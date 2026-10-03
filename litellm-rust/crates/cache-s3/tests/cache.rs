mod support;

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aws_smithy_types::{DateTime, date_time::Format};
use litellm_cache::{
    BaseCache, BatchCache, BatchEntry, DisconnectCache, Error, ExactCacheContext, FlushCache,
};
use litellm_cache_s3::S3CacheConfig;
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use support::FakeBucket;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    http::HeaderMap,
    matchers::{method, path},
};

#[fixture]
async fn server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200).insert_header("etag", "\"etag\""))
        .mount(&server)
        .await;
    server
}

fn http_date_from(headers: &HeaderMap, name: &str) -> Option<SystemTime> {
    headers
        .get(name)
        .and_then(|value| DateTime::from_str(value.to_str().ok()?, Format::HttpDate).ok())
        .map(|date| UNIX_EPOCH + Duration::new(date.secs() as u64, date.subsec_nanos()))
}

fn ttl(seconds: u64) -> ExactCacheContext {
    ExactCacheContext {
        ttl: Some(Duration::from_secs(seconds)),
    }
}

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_writes_python_metadata_with_and_without_ttl(#[future(awt)] server: MockServer) {
    let cache = support::cache(&server.uri());
    cache
        .set_cache("alpha:beta", json!({"answer": 1}), &ttl(90))
        .unwrap();
    cache
        .set_cache("plain", json!({"answer": 2}), &ExactCacheContext::default())
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    let ttl_request = requests
        .iter()
        .find(|request| request.url.path() == "/cache-bucket/team/alpha/beta")
        .expect("ttl write should hit the converted S3 key");
    assert_eq!(
        ttl_request.headers["cache-control"].to_str().unwrap(),
        "immutable, max-age=90, s-maxage=90"
    );
    assert_eq!(
        ttl_request.headers["content-type"].to_str().unwrap(),
        "application/json"
    );
    assert_eq!(
        ttl_request.headers["content-language"].to_str().unwrap(),
        "en"
    );
    assert_eq!(
        ttl_request.headers["content-disposition"].to_str().unwrap(),
        "inline; filename=\"team/alpha/beta.json\""
    );
    let expires = http_date_from(&ttl_request.headers, "expires").expect("ttl write sets Expires");
    let remaining = expires.duration_since(SystemTime::now()).unwrap();
    assert!(remaining > Duration::from_secs(60) && remaining <= Duration::from_secs(91));
    assert_eq!(
        serde_json::from_slice::<Value>(&ttl_request.body).unwrap(),
        json!({"answer": 1})
    );

    let plain = requests
        .iter()
        .find(|request| request.url.path() == "/cache-bucket/team/plain")
        .expect("no-ttl write should hit the converted S3 key");
    assert_eq!(
        plain.headers["cache-control"].to_str().unwrap(),
        "immutable, max-age=31536000, s-maxage=31536000"
    );
    assert!(plain.headers.get("expires").is_none());
    assert_eq!(
        plain.headers["content-disposition"].to_str().unwrap(),
        "inline; filename=\"team/plain.json\""
    );
}

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn async_set_signs_with_the_configured_keys(#[future(awt)] server: MockServer) {
    support::cache(&server.uri())
        .async_set_cache("key", json!({"answer": 1}), ttl(3600))
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    let request = &requests[0];
    assert_eq!(request.url.path(), "/cache-bucket/team/key");
    assert!(
        request.headers["authorization"]
            .to_str()
            .unwrap()
            .contains("Credential=key/")
    );
    assert!(request.headers.get("x-amz-security-token").is_none());
    assert_eq!(
        request.headers["cache-control"].to_str().unwrap(),
        "immutable, max-age=3600, s-maxage=3600"
    );
}

#[rstest]
#[case::hit("hit", ResponseTemplate::new(200).set_body_json(json!({"answer": 3})), Ok(Some(json!({"answer": 3}))))]
#[case::no_such_key(
    "missing",
    ResponseTemplate::new(404).set_body_string("<Error><Code>NoSuchKey</Code></Error>"),
    Ok(None)
)]
#[case::access_denied(
    "denied",
    ResponseTemplate::new(403).set_body_string("<Error><Code>AccessDenied</Code></Error>"),
    Ok(None)
)]
#[case::expired(
    "expired",
    ResponseTemplate::new(200)
        .insert_header("expires", "Thu, 01 Jan 1970 00:00:00 GMT")
        .set_body_json(json!({"answer": 4})),
    Ok(None)
)]
#[case::not_yet_expired(
    "fresh",
    ResponseTemplate::new(200)
        .insert_header("expires", "Fri, 01 Jan 2100 00:00:00 GMT")
        .set_body_json(json!({"answer": 5})),
    Ok(Some(json!({"answer": 5})))
)]
#[case::malformed(
    "malformed",
    ResponseTemplate::new(200).set_body_string("not a cache entry"),
    Err(Error::InvalidEntry)
)]
#[case::server_error("broken", ResponseTemplate::new(500), Err(Error::Unavailable))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_maps_s3_responses(
    #[future(awt)] server: MockServer,
    #[case] key: &str,
    #[case] response: ResponseTemplate,
    #[case] expected: Result<Option<Value>, Error>,
) {
    Mock::given(method("GET"))
        .and(path(format!("/cache-bucket/team/{key}")))
        .respond_with(response)
        .mount(&server)
        .await;
    let cache = support::cache(&server.uri());
    let context = ExactCacheContext::default();

    assert_eq!(cache.get_cache(key, &context), expected);
    assert_eq!(cache.async_get_cache(key, &context).await, expected);
}

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batch_get_preserves_order_with_hits_misses_and_invalid(#[future(awt)] server: MockServer) {
    for (key, status, body) in [
        ("first", 200, "{\"answer\": 1}"),
        ("invalid", 200, "garbage"),
    ] {
        Mock::given(method("GET"))
            .and(path(format!("/cache-bucket/team/{key}")))
            .respond_with(ResponseTemplate::new(status).set_body_string(body))
            .mount(&server)
            .await;
    }
    Mock::given(method("GET"))
        .and(path("/cache-bucket/team/miss"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;
    let cache = support::cache(&server.uri());
    let context = ExactCacheContext::default();
    let keys = vec![
        "first".to_string(),
        "miss".to_string(),
        "invalid".to_string(),
    ];
    let expected = vec![
        BatchEntry::Hit(json!({"answer": 1})),
        BatchEntry::Miss,
        BatchEntry::Invalid,
    ];

    assert_eq!(cache.batch_get_cache(&keys, &context).unwrap(), expected);
    assert_eq!(
        cache.async_batch_get_cache(keys, context).await.unwrap(),
        expected
    );
}

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pipeline_writes_every_entry_with_the_shared_ttl() {
    let server = FakeBucket::serve().await;
    let cache = support::cache(&server.uri());
    cache
        .async_set_cache_pipeline(
            vec![
                ("one".into(), json!({"n": 1})),
                ("two".into(), json!({"n": 2})),
            ],
            ttl(30),
        )
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests.iter().all(|request| {
        request.headers["cache-control"].to_str().unwrap() == "immutable, max-age=30, s-maxage=30"
    }));
    assert_eq!(
        cache
            .get_cache("two", &ExactCacheContext::default())
            .unwrap(),
        Some(json!({"n": 2}))
    );
}

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flush_and_disconnect_are_noops_like_python(#[future(awt)] server: MockServer) {
    let cache = support::cache(&server.uri());

    cache.flush_cache().unwrap();
    cache.async_flush_cache().await.unwrap();
    cache.disconnect().await.unwrap();
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[rstest]
#[case::without_ttl(ExactCacheContext::default(), None)]
#[case::with_ttl(ttl(45), Some(Duration::from_secs(45)))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_ttl_reports_the_request_ttl(
    #[case] context: ExactCacheContext,
    #[case] expected: Option<Duration>,
) {
    assert_eq!(
        support::cache("http://localhost").get_ttl(&context),
        expected
    );
}

#[rstest]
#[case::prefixed("team/", "a:b:c", "team/a/b/c")]
#[case::prefixed_plain("team/", "plain", "team/plain")]
#[case::unprefixed("", "a:b", "a/b")]
fn key_conversion_prefixes_and_splits_colons(
    #[case] key_prefix: &str,
    #[case] key: &str,
    #[case] expected: &str,
) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let cache = support::cache_with(
        S3CacheConfig {
            key_prefix: key_prefix.to_string(),
            ..support::config("http://localhost")
        },
        runtime.handle().clone(),
    );

    assert_eq!(cache.bucket(), "cache-bucket");
    assert_eq!(cache.key_prefix(), key_prefix);
    assert_eq!(cache.region(), "us-east-1");
    assert_eq!(cache.endpoint(), Some("http://localhost"));
    assert_eq!(cache.to_s3_key(key), expected);
}

#[rstest]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sync_methods_block_outside_the_runtime() {
    let server = FakeBucket::serve().await;
    let uri = server.uri();
    let handle = tokio::runtime::Handle::current();
    let cached = tokio::task::spawn_blocking(move || {
        let cache = support::cache_with(support::config(&uri), handle);
        let context = ExactCacheContext::default();
        cache
            .set_cache("key", json!({"answer": 9}), &context)
            .unwrap();
        cache.get_cache("key", &context).unwrap()
    })
    .await
    .unwrap();

    assert_eq!(cached, Some(json!({"answer": 9})));
}
