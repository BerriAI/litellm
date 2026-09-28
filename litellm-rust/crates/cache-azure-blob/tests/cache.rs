mod support;

use std::{sync::Arc, time::Duration};

use azure_core::http::Method;
use litellm_cache::{
    BaseCache, BatchCache, BatchEntry, DisconnectCache, Error, ExactCacheContext, FlushCache,
};
use litellm_cache_azure_blob::AzureBlobCache;
use litellm_cache_response::{
    CacheEntry, CacheKeyField, CacheKeyInput, ResponseCache, ResponseCacheCodec,
    ResponseCacheRequest, cache_key,
};
use rstest::{fixture, rstest};
use serde_json::json;
use support::{ACCOUNT_URL, CONTAINER, FakeBlobService, RecordedRequest};
use tokio::runtime::Runtime;

type Fixture = support::Fixture<ResponseCacheCodec>;

#[fixture]
fn fixture() -> Fixture {
    Fixture::new(FakeBlobService::default(), ResponseCacheCodec)
}

fn response_cache(fixture: &Fixture) -> ResponseCache<AzureBlobCache<ResponseCacheCodec>> {
    ResponseCache::new(fixture.cache.clone())
}

fn request(model: &str) -> ResponseCacheRequest {
    ResponseCacheRequest::new(CacheKeyInput {
        fields: vec![CacheKeyField {
            name: "model".into(),
            value: Some(model.into()),
            api_parameter: true,
            internal_parameter: false,
        }],
        preset: None,
        namespace: None,
        include_provider_parameters: false,
    })
}

fn now() -> Duration {
    Duration::from_secs(1_700_000_000)
}

fn entry(value: serde_json::Value) -> CacheEntry {
    CacheEntry {
        timestamp: Some(1_700_000_000.5),
        response: value,
    }
}

fn no_ttl() -> ExactCacheContext {
    ExactCacheContext::default()
}

fn with_ttl(seconds: u64) -> ExactCacheContext {
    ExactCacheContext {
        ttl: Some(Duration::from_secs(seconds)),
    }
}

fn connect_to(account_url: &str) -> (FakeBlobService, AzureBlobCache<ResponseCacheCodec>) {
    let runtime = Runtime::new().unwrap();
    let service = FakeBlobService::default();
    let cache = runtime
        .block_on(support::connect(
            &service,
            account_url,
            ResponseCacheCodec,
            runtime.handle().clone(),
        ))
        .unwrap();
    (service, cache)
}

#[rstest]
fn connect_creates_the_container_once(fixture: Fixture) {
    assert!(fixture.service.container_exists());
    assert_eq!(
        fixture.service.requests(),
        vec![RecordedRequest {
            method: Method::Put,
            path: format!("/{CONTAINER}"),
            query: "restype=container".into(),
            if_none_match: None,
        }]
    );
    assert_eq!(fixture.cache.account_url(), ACCOUNT_URL);
    assert_eq!(fixture.cache.container_name(), CONTAINER);
}

#[rstest]
fn connect_accepts_an_existing_container() {
    let fixture = Fixture::new(
        FakeBlobService::with_existing_container(),
        ResponseCacheCodec,
    );
    assert!(fixture.service.container_exists());
    assert_eq!(fixture.service.requests().len(), 1);
}

#[rstest]
fn connect_accepts_account_urls_with_trailing_slash() {
    let (service, cache) = connect_to("https://example.blob.core.windows.net/");
    assert_eq!(service.requests()[0].path, format!("/{CONTAINER}"));
    assert_eq!(cache.account_url(), "https://example.blob.core.windows.net");
}

#[rstest]
fn connect_keeps_account_url_query_parameters_on_the_container_path() {
    let (service, _) = connect_to("https://example.blob.core.windows.net/?sv=2024-01-01&sig=abc");
    let create = &service.requests()[0];
    assert_eq!(create.path, format!("/{CONTAINER}"));
    assert!(create.query.contains("sig=abc"));
}

#[rstest]
fn connect_surfaces_service_failures() {
    let runtime = Runtime::new().unwrap();
    let service = FakeBlobService::default();
    service.set_failing(true);
    let result = runtime.block_on(support::connect(
        &service,
        ACCOUNT_URL,
        ResponseCacheCodec,
        runtime.handle().clone(),
    ));
    assert!(matches!(result, Err(Error::Unavailable)));
}

#[rstest]
fn sync_set_and_get_round_trip_python_json_shape(fixture: Fixture) {
    let value = entry(json!({"choices": [{"message": {"content": "héllo 🌍"}}]}));
    fixture
        .cache
        .set_cache("key-1", value.clone(), &no_ttl())
        .unwrap();

    assert_eq!(
        fixture.stored_json("key-1"),
        json!({
            "timestamp": 1_700_000_000.5,
            "response": {"choices": [{"message": {"content": "héllo 🌍"}}]}
        })
    );
    assert_eq!(
        fixture.cache.get_cache("key-1", &no_ttl()).unwrap(),
        Some(value)
    );
}

#[rstest]
#[case::blob_already_exists(false)]
#[case::precondition_conflict(true)]
fn sync_set_does_not_overwrite_an_existing_blob(fixture: Fixture, #[case] precondition: bool) {
    fixture.service.set_precondition_conflicts(precondition);
    fixture
        .cache
        .set_cache("key", entry(json!({"v": "first"})), &no_ttl())
        .unwrap();
    fixture
        .cache
        .set_cache("key", entry(json!({"v": "second"})), &no_ttl())
        .unwrap();

    assert_eq!(
        fixture.stored_json("key")["response"],
        json!({"v": "first"})
    );
    let uploads: Vec<_> = fixture
        .service
        .requests()
        .into_iter()
        .filter(|request| request.method == Method::Put && request.path.ends_with("/key"))
        .collect();
    assert_eq!(uploads.len(), 2);
    assert!(
        uploads
            .iter()
            .all(|request| request.if_none_match.as_deref() == Some("*"))
    );
}

#[rstest]
fn async_set_overwrites_an_existing_blob(fixture: Fixture) {
    fixture.runtime.block_on(async {
        fixture
            .cache
            .async_set_cache("key", entry(json!({"v": "first"})), no_ttl())
            .await
            .unwrap();
        fixture
            .cache
            .async_set_cache("key", entry(json!({"v": "second"})), no_ttl())
            .await
            .unwrap();
        assert_eq!(
            fixture
                .cache
                .async_get_cache("key", &no_ttl())
                .await
                .unwrap(),
            Some(entry(json!({"v": "second"})))
        );
    });
    assert_eq!(
        fixture.stored_json("key")["response"],
        json!({"v": "second"})
    );
    assert!(
        fixture
            .service
            .requests()
            .iter()
            .filter(|request| request.method == Method::Put && request.path.ends_with("/key"))
            .all(|request| request.if_none_match.is_none())
    );
}

#[rstest]
fn missing_blobs_are_misses(fixture: Fixture) {
    assert_eq!(fixture.cache.get_cache("absent", &no_ttl()).unwrap(), None);
    assert_eq!(
        fixture
            .runtime
            .block_on(fixture.cache.async_get_cache("absent", &no_ttl()))
            .unwrap(),
        None
    );
}

#[rstest]
fn ttl_is_ignored_and_entries_never_expire(fixture: Fixture) {
    assert_eq!(fixture.cache.get_ttl(&with_ttl(1)), None);
    assert_eq!(fixture.cache.get_ttl(&no_ttl()), None);

    fixture
        .cache
        .set_cache("key", entry(json!("value")), &with_ttl(1))
        .unwrap();
    std::thread::sleep(Duration::from_millis(1100));
    assert_eq!(
        fixture.cache.get_cache("key", &with_ttl(1)).unwrap(),
        Some(entry(json!("value")))
    );
    assert!(
        fixture
            .service
            .requests()
            .iter()
            .all(|request| !request.query.contains("expiry"))
    );
}

#[rstest]
#[case::broken_json("broken-json", b"{not json".as_slice())]
#[case::broken_utf8("broken-utf8", &[0xff, 0xfe, 0x22])]
#[case::wrong_shape("wrong-shape", br#"{"timestamp": "yesterday"}"#.as_slice())]
fn malformed_blobs_are_invalid_entries(fixture: Fixture, #[case] key: &str, #[case] bytes: &[u8]) {
    fixture.service.seed_blob(key, bytes);
    assert!(matches!(
        fixture.cache.get_cache(key, &no_ttl()),
        Err(Error::InvalidEntry)
    ));
}

#[rstest]
fn malformed_blobs_are_response_cache_misses(fixture: Fixture) {
    let response_cache = response_cache(&fixture);
    let broken = request("broken");
    fixture
        .service
        .seed_blob(&cache_key(&broken.key), b"{not json");
    assert_eq!(response_cache.lookup(&broken, now()).unwrap(), None);
    assert_eq!(
        fixture
            .runtime
            .block_on(response_cache.async_lookup(&broken, now()))
            .unwrap(),
        None
    );
}

#[rstest]
fn batch_get_preserves_order_and_marks_misses_and_invalid_entries(fixture: Fixture) {
    fixture
        .cache
        .set_cache("a", entry(json!("A")), &no_ttl())
        .unwrap();
    fixture
        .cache
        .set_cache("c", entry(json!("C")), &no_ttl())
        .unwrap();
    fixture.service.seed_blob("bad", b"nope");
    let keys = ["c", "missing", "a", "bad"].map(String::from);

    let sync = fixture.cache.batch_get_cache(&keys, &no_ttl()).unwrap();
    assert_eq!(
        sync,
        vec![
            BatchEntry::Hit(entry(json!("C"))),
            BatchEntry::Miss,
            BatchEntry::Hit(entry(json!("A"))),
            BatchEntry::Invalid,
        ]
    );

    let asynchronous = fixture
        .runtime
        .block_on(fixture.cache.async_batch_get_cache(keys.to_vec(), no_ttl()))
        .unwrap();
    assert_eq!(asynchronous, sync);

    let response_cache = response_cache(&fixture);
    let requests = [request("hit"), request("missing"), request("bad")];
    response_cache
        .store(&requests[0], json!("HIT"), now())
        .unwrap();
    fixture
        .service
        .seed_blob(&cache_key(&requests[2].key), b"nope");
    let hits = response_cache.lookup_batch(&requests, now()).unwrap();
    assert_eq!(hits.values, vec![Some(json!("HIT")), None, None]);
    assert_eq!(hits.missing_indices, vec![1, 2]);
    let async_hits = fixture
        .runtime
        .block_on(response_cache.async_lookup_batch(&requests, now()))
        .unwrap();
    assert_eq!(async_hits.values, hits.values);
}

#[rstest]
fn async_pipeline_writes_every_entry_with_overwrite(fixture: Fixture) {
    fixture.service.seed_blob("k2", b"stale");
    fixture
        .runtime
        .block_on(fixture.cache.async_set_cache_pipeline(
            vec![
                ("k1".into(), entry(json!({"n": 1}))),
                ("k2".into(), entry(json!({"n": 2}))),
                ("k3".into(), entry(json!({"n": 3}))),
            ],
            with_ttl(30),
        ))
        .unwrap();
    assert_eq!(fixture.service.blob_names(), ["k1", "k2", "k3"]);
    assert_eq!(fixture.stored_json("k2")["response"], json!({"n": 2}));
}

#[rstest]
fn flush_deletes_every_blob_in_the_container(fixture: Fixture) {
    for key in ["x", "y", "z"] {
        fixture
            .cache
            .set_cache(key, entry(json!(key)), &no_ttl())
            .unwrap();
    }
    fixture.cache.flush_cache().unwrap();
    assert!(fixture.service.blob_names().is_empty());
    assert!(fixture.service.container_exists());

    fixture
        .cache
        .set_cache("again", entry(json!(1)), &no_ttl())
        .unwrap();
    fixture
        .runtime
        .block_on(fixture.cache.async_flush_cache())
        .unwrap();
    assert!(fixture.service.blob_names().is_empty());
}

#[rstest]
fn service_failures_map_to_unavailable(fixture: Fixture) {
    fixture.service.set_failing(true);
    assert!(matches!(
        fixture.cache.get_cache("key", &no_ttl()),
        Err(Error::Unavailable)
    ));
    assert!(matches!(
        fixture.cache.set_cache("key", entry(json!(1)), &no_ttl()),
        Err(Error::Unavailable)
    ));
    assert!(matches!(
        fixture.cache.flush_cache(),
        Err(Error::Unavailable)
    ));
    assert!(matches!(
        fixture.runtime.block_on(
            fixture
                .cache
                .async_set_cache_pipeline(vec![("k".into(), entry(json!(1)))], no_ttl())
        ),
        Err(Error::Unavailable)
    ));
}

#[rstest]
fn disconnect_is_idempotent_and_keeps_data(fixture: Fixture) {
    fixture
        .cache
        .set_cache("key", entry(json!(1)), &no_ttl())
        .unwrap();
    fixture.runtime.block_on(async {
        fixture.cache.disconnect().await.unwrap();
        fixture.cache.disconnect().await.unwrap();
    });
    assert_eq!(
        fixture.cache.get_cache("key", &no_ttl()).unwrap(),
        Some(entry(json!(1)))
    );
}

#[rstest]
fn response_cache_stores_and_reads_through_the_backend(fixture: Fixture) {
    let response_cache = response_cache(&fixture);
    let mut request = request("gpt");
    request.context = with_ttl(60);
    let response = json!({"id": "chatcmpl-1"});
    response_cache
        .store(&request, response.clone(), now())
        .unwrap();
    assert_eq!(
        fixture.stored_json(&cache_key(&request.key)),
        json!({"timestamp": 1_700_000_000.0, "response": {"id": "chatcmpl-1"}})
    );
    assert_eq!(
        response_cache
            .lookup(&request, now() + Duration::from_secs(3600))
            .unwrap(),
        Some(response.clone())
    );
    assert_eq!(
        fixture
            .runtime
            .block_on(response_cache.async_lookup(&request, now() + Duration::from_secs(3600)))
            .unwrap(),
        Some(response.clone())
    );
    fixture.runtime.block_on(async {
        response_cache
            .async_store(&request, json!("replaced"), now())
            .await
            .unwrap();
        assert_eq!(
            response_cache.async_lookup(&request, now()).await.unwrap(),
            Some(json!("replaced"))
        );
        response_cache.async_flush().await.unwrap();
        assert_eq!(
            response_cache.async_lookup(&request, now()).await.unwrap(),
            None
        );
    });
}

#[rstest]
fn non_object_responses_are_written_serialized_like_python(fixture: Fixture) {
    fixture
        .cache
        .set_cache("s", entry(json!("plain")), &no_ttl())
        .unwrap();
    assert_eq!(
        fixture.stored_json("s"),
        json!({"timestamp": 1_700_000_000.5, "response": "\"plain\""})
    );
    assert_eq!(
        fixture.cache.get_cache("s", &no_ttl()).unwrap(),
        Some(entry(json!("plain")))
    );
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn sync_methods_block_inside_a_multi_thread_runtime() {
    let service = FakeBlobService::default();
    let cache = support::connect(
        &service,
        ACCOUNT_URL,
        ResponseCacheCodec,
        tokio::runtime::Handle::current(),
    )
    .await
    .map(Arc::new)
    .unwrap();
    cache.set_cache("key", entry(json!(1)), &no_ttl()).unwrap();
    assert_eq!(
        cache.get_cache("key", &no_ttl()).unwrap(),
        Some(entry(json!(1)))
    );
}
