use std::time::{Duration, SystemTime, UNIX_EPOCH};

use litellm_auth_aws::AwsAuthConfig;
use litellm_cache::{
    BaseCache, BatchCache, BatchEntry, Error, ExactCacheContext, FlushCache, JsonCodec,
};
use litellm_cache_s3::{S3Cache, S3CacheConfig, S3Endpoint};
use serde_json::{Value, json};
use tokio::runtime::Handle;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

fn config(endpoint: String) -> S3CacheConfig {
    S3CacheConfig {
        bucket: "cache-bucket".to_string(),
        key_prefix: "team/".to_string(),
        region: "us-east-1".to_string(),
        endpoint: Some(S3Endpoint { url: endpoint }),
        auth: AwsAuthConfig {
            access_key_id: Some("key".to_string()),
            secret_access_key: Some("secret".to_string()),
            region_name: Some("us-east-1".to_string()),
            ..Default::default()
        },
    }
}

fn cache(endpoint: &str) -> S3Cache<JsonCodec<Value>> {
    S3Cache::new(
        config(endpoint.to_string()),
        JsonCodec::<Value>::new(),
        Handle::current(),
    )
}

async fn mock_server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(200).insert_header("etag", "\"etag\""))
        .mount(&server)
        .await;
    server
}

fn http_date_from(headers: &wiremock::http::HeaderMap, name: &str) -> Option<SystemTime> {
    use aws_smithy_types::{DateTime, date_time::Format};
    headers
        .get(name)
        .and_then(|value| DateTime::from_str(value.to_str().ok()?, Format::HttpDate).ok())
        .map(|date| UNIX_EPOCH + Duration::new(date.secs() as u64, date.subsec_nanos()))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_writes_python_metadata_with_and_without_ttl() {
    let server = mock_server().await;
    let cache = cache(&server.uri());
    let context = ExactCacheContext {
        ttl: Some(Duration::from_secs(90)),
    };
    cache
        .set_cache("alpha:beta", json!({"answer": 1}), &context)
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_hit_miss_expired_and_invalid_entries() {
    let server = mock_server().await;
    Mock::given(method("GET"))
        .and(path("/cache-bucket/team/hit"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"answer": 3})))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/cache-bucket/team/missing"))
        .respond_with(
            ResponseTemplate::new(404).set_body_string("<Error><Code>NoSuchKey</Code></Error>"),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/cache-bucket/team/denied"))
        .respond_with(
            ResponseTemplate::new(403).set_body_string("<Error><Code>AccessDenied</Code></Error>"),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/cache-bucket/team/expired"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("expires", "Thu, 01 Jan 1970 00:00:00 GMT")
                .set_body_json(json!({"answer": 4})),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/cache-bucket/team/malformed"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not a cache entry"))
        .mount(&server)
        .await;
    let cache = cache(&server.uri());
    let context = ExactCacheContext::default();

    assert_eq!(
        cache.get_cache("hit", &context).unwrap(),
        Some(json!({"answer": 3}))
    );
    assert_eq!(cache.get_cache("missing", &context).unwrap(), None);
    assert_eq!(cache.get_cache("denied", &context).unwrap(), None);
    assert_eq!(cache.get_cache("expired", &context).unwrap(), None);
    assert_eq!(
        cache.get_cache("malformed", &context),
        Err(Error::InvalidEntry)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batch_get_preserves_order_with_hits_misses_and_invalid() {
    let server = mock_server().await;
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
    let cache = cache(&server.uri());
    let context = ExactCacheContext::default();
    let keys = vec![
        "first".to_string(),
        "miss".to_string(),
        "invalid".to_string(),
    ];

    let entries = cache.batch_get_cache(&keys, &context).unwrap();

    assert_eq!(
        entries,
        vec![
            BatchEntry::Hit(json!({"answer": 1})),
            BatchEntry::Miss,
            BatchEntry::Invalid,
        ]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unsupported_and_noop_capabilities_match_python() {
    let server = mock_server().await;
    let cache = cache(&server.uri());

    assert_eq!(
        cache.test_connection().await,
        Err(Error::UnsupportedOperation)
    );
    cache.flush_cache().unwrap();
    cache.disconnect().await.unwrap();
    assert_eq!(cache.get_ttl(&ExactCacheContext::default()), None);
    assert_eq!(
        cache.get_ttl(&ExactCacheContext {
            ttl: Some(Duration::from_secs(45)),
        }),
        Some(Duration::from_secs(45))
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[test]
fn key_conversion_prefixes_and_splits_colons() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let _guard = runtime.enter();
    let cache = S3Cache::new(
        S3CacheConfig {
            key_prefix: "team/".to_string(),
            ..config("http://localhost".to_string())
        },
        JsonCodec::<Value>::new(),
        runtime.handle().clone(),
    );

    assert_eq!(cache.bucket(), "cache-bucket");
    assert_eq!(cache.key_prefix(), "team/");
    assert_eq!(cache.to_s3_key("a:b:c"), "team/a/b/c");
    assert_eq!(cache.to_s3_key("plain"), "team/plain");

    let unprefixed = S3Cache::new(
        S3CacheConfig {
            key_prefix: String::new(),
            ..config("http://localhost".to_string())
        },
        JsonCodec::<Value>::new(),
        runtime.handle().clone(),
    );
    assert_eq!(unprefixed.to_s3_key("a:b"), "a/b");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sync_methods_block_inside_and_outside_the_runtime() {
    let server = mock_server().await;
    Mock::given(method("GET"))
        .and(path("/cache-bucket/team/key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"answer": 9})))
        .mount(&server)
        .await;
    let uri = server.uri();
    let cache = tokio::task::spawn_blocking(move || {
        let cache = cache(&uri);
        let context = ExactCacheContext::default();
        cache
            .set_cache("key", json!({"answer": 9}), &context)
            .unwrap();
        cache.get_cache("key", &context).unwrap()
    })
    .await
    .unwrap();

    assert_eq!(cache, Some(json!({"answer": 9})));
}
