use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use azure_core::http::{
    AsyncRawResponse, Body, ClientOptions, HttpClient, Method, Request, StatusCode, Transport,
    headers::{HeaderName, Headers},
};
use litellm_cache::{
    BaseCache, BatchCache, BatchEntry, CacheConnectionStatus, Error, ExactCacheContext, FlushCache,
};
use litellm_cache_response::{
    CacheEntry, CacheKeyField, CacheKeyInput, ResponseCache, ResponseCacheCodec,
    ResponseCacheRequest, cache_key,
};
use serde_json::json;
use tokio::runtime::Runtime;

use super::AzureBlobCache;

const ACCOUNT_URL: &str = "https://example.blob.core.windows.net";
const CONTAINER: &str = "litellm-cache";
const IF_NONE_MATCH: HeaderName = HeaderName::from_static("if-none-match");
const ERROR_CODE: HeaderName = HeaderName::from_static("x-ms-error-code");

#[derive(Clone, Debug, PartialEq, Eq)]
struct RecordedRequest {
    method: Method,
    path: String,
    query: String,
    if_none_match: Option<String>,
}

#[derive(Default)]
struct FakeState {
    container_exists: bool,
    blobs: BTreeMap<String, Vec<u8>>,
    requests: Vec<RecordedRequest>,
    failing: bool,
    precondition_conflicts: bool,
}

#[derive(Clone, Default)]
struct FakeBlobService {
    state: Arc<Mutex<FakeState>>,
}

impl std::fmt::Debug for FakeBlobService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FakeBlobService")
    }
}

impl FakeBlobService {
    fn with_existing_container() -> Self {
        let service = Self::default();
        service.state.lock().unwrap().container_exists = true;
        service
    }

    fn blob(&self, name: &str) -> Option<Vec<u8>> {
        self.state.lock().unwrap().blobs.get(name).cloned()
    }

    fn blob_names(&self) -> Vec<String> {
        self.state.lock().unwrap().blobs.keys().cloned().collect()
    }

    fn seed_blob(&self, name: &str, bytes: &[u8]) {
        self.state
            .lock()
            .unwrap()
            .blobs
            .insert(name.to_string(), bytes.to_vec());
    }

    fn set_failing(&self, failing: bool) {
        self.state.lock().unwrap().failing = failing;
    }

    fn set_precondition_conflicts(&self, enabled: bool) {
        self.state.lock().unwrap().precondition_conflicts = enabled;
    }

    fn requests(&self) -> Vec<RecordedRequest> {
        self.state.lock().unwrap().requests.clone()
    }

    fn container_exists(&self) -> bool {
        self.state.lock().unwrap().container_exists
    }

    fn respond(status: StatusCode, error_code: Option<&str>, body: Vec<u8>) -> AsyncRawResponse {
        let mut headers = Headers::new();
        if let Some(code) = error_code {
            headers.insert(ERROR_CODE, code.to_string());
        }
        AsyncRawResponse::from_bytes(status, headers, body)
    }

    fn list_body(state: &FakeState) -> Vec<u8> {
        let mut xml = String::from(
            r#"<?xml version="1.0" encoding="utf-8"?><EnumerationResults ServiceEndpoint="https://example.blob.core.windows.net/" ContainerName="litellm-cache"><Blobs>"#,
        );
        for name in state.blobs.keys() {
            xml.push_str(&format!(
                "<Blob><Name>{name}</Name><Properties><BlobType>BlockBlob</BlobType></Properties></Blob>"
            ));
        }
        xml.push_str("</Blobs><NextMarker /></EnumerationResults>");
        xml.into_bytes()
    }
}

#[async_trait::async_trait]
impl HttpClient for FakeBlobService {
    async fn execute_request(&self, request: &Request) -> azure_core::Result<AsyncRawResponse> {
        let mut state = self.state.lock().unwrap();
        let path = request.url().path().to_string();
        let query = request.url().query().unwrap_or_default().to_string();
        let if_none_match = request
            .headers()
            .get_optional_str(&IF_NONE_MATCH)
            .map(str::to_owned);
        state.requests.push(RecordedRequest {
            method: request.method(),
            path: path.clone(),
            query: query.clone(),
            if_none_match: if_none_match.clone(),
        });
        if state.failing {
            return Ok(Self::respond(
                StatusCode::Forbidden,
                Some("AuthorizationFailure"),
                Vec::new(),
            ));
        }
        let container_path = format!("/{CONTAINER}");
        let blob_name = path
            .strip_prefix(&format!("{container_path}/"))
            .map(str::to_owned);
        let is_container = path == container_path && query.contains("restype=container");
        let response = match (request.method(), is_container, blob_name) {
            (Method::Put, true, None) if state.container_exists => Self::respond(
                StatusCode::Conflict,
                Some("ContainerAlreadyExists"),
                Vec::new(),
            ),
            (Method::Put, true, None) => {
                state.container_exists = true;
                Self::respond(StatusCode::Created, None, Vec::new())
            }
            (Method::Get, true, None) if query.contains("comp=list") => {
                Self::respond(StatusCode::Ok, None, Self::list_body(&state))
            }
            (Method::Get, true, None) if state.container_exists => {
                Self::respond(StatusCode::Ok, None, Vec::new())
            }
            (Method::Get, true, None) => {
                Self::respond(StatusCode::NotFound, Some("ContainerNotFound"), Vec::new())
            }
            (Method::Put, false, Some(name)) => {
                if if_none_match.as_deref() == Some("*") && state.blobs.contains_key(&name) {
                    if state.precondition_conflicts {
                        Self::respond(
                            StatusCode::PreconditionFailed,
                            Some("ConditionNotMet"),
                            Vec::new(),
                        )
                    } else {
                        Self::respond(StatusCode::Conflict, Some("BlobAlreadyExists"), Vec::new())
                    }
                } else {
                    let bytes = match request.body() {
                        Body::Bytes(bytes) => bytes.to_vec(),
                        Body::SeekableStream(_) => panic!("unexpected streaming upload"),
                    };
                    state.blobs.insert(name, bytes);
                    Self::respond(StatusCode::Created, None, Vec::new())
                }
            }
            (Method::Get, false, Some(name)) => match state.blobs.get(&name) {
                Some(bytes) => Self::respond(StatusCode::Ok, None, bytes.clone()),
                None => Self::respond(StatusCode::NotFound, Some("BlobNotFound"), Vec::new()),
            },
            (Method::Delete, false, Some(name)) => match state.blobs.remove(&name) {
                Some(_) => Self::respond(StatusCode::Accepted, None, Vec::new()),
                None => Self::respond(StatusCode::NotFound, Some("BlobNotFound"), Vec::new()),
            },
            (method, _, _) => panic!("unexpected request {method:?} {path}?{query}"),
        };
        Ok(response)
    }
}

struct Fixture {
    runtime: Runtime,
    service: FakeBlobService,
    cache: Arc<AzureBlobCache<ResponseCacheCodec>>,
}

impl Fixture {
    fn new(service: FakeBlobService) -> Self {
        let runtime = Runtime::new().unwrap();
        let cache = runtime
            .block_on(Self::connect(&service, runtime.handle().clone()))
            .unwrap();
        Self {
            runtime,
            service,
            cache: Arc::new(cache),
        }
    }

    async fn connect(
        service: &FakeBlobService,
        handle: tokio::runtime::Handle,
    ) -> Result<AzureBlobCache<ResponseCacheCodec>, Error> {
        AzureBlobCache::connect_with_options(
            ACCOUNT_URL,
            CONTAINER,
            None,
            ClientOptions {
                transport: Some(Transport::new(Arc::new(service.clone()))),
                ..ClientOptions::default()
            },
            ResponseCacheCodec,
            handle,
        )
        .await
    }

    fn response_cache(&self) -> ResponseCache<AzureBlobCache<ResponseCacheCodec>> {
        ResponseCache::new(self.cache.clone())
    }

    fn stored_json(&self, key: &str) -> serde_json::Value {
        serde_json::from_slice(&self.service.blob(key).expect("blob should exist")).unwrap()
    }
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

#[test]
fn connect_creates_the_container_once() {
    let fixture = Fixture::new(FakeBlobService::default());
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

#[test]
fn connect_accepts_an_existing_container() {
    let fixture = Fixture::new(FakeBlobService::with_existing_container());
    assert!(fixture.service.container_exists());
    assert_eq!(fixture.service.requests().len(), 1);
}

#[test]
fn connect_accepts_account_urls_with_trailing_slash() {
    let runtime = Runtime::new().unwrap();
    let service = FakeBlobService::default();
    let cache = runtime
        .block_on(AzureBlobCache::connect_with_options(
            "https://example.blob.core.windows.net/",
            CONTAINER,
            None,
            ClientOptions {
                transport: Some(Transport::new(Arc::new(service.clone()))),
                ..ClientOptions::default()
            },
            ResponseCacheCodec,
            runtime.handle().clone(),
        ))
        .unwrap();
    assert_eq!(service.requests()[0].path, format!("/{CONTAINER}"));
    assert_eq!(cache.account_url(), "https://example.blob.core.windows.net");
}

#[test]
fn connect_keeps_account_url_query_parameters_on_the_container_path() {
    let runtime = Runtime::new().unwrap();
    let service = FakeBlobService::default();
    runtime
        .block_on(AzureBlobCache::connect_with_options(
            "https://example.blob.core.windows.net/?sv=2024-01-01&sig=abc",
            CONTAINER,
            None,
            ClientOptions {
                transport: Some(Transport::new(Arc::new(service.clone()))),
                ..ClientOptions::default()
            },
            ResponseCacheCodec,
            runtime.handle().clone(),
        ))
        .unwrap();
    let create = &service.requests()[0];
    assert_eq!(create.path, format!("/{CONTAINER}"));
    assert!(create.query.contains("sig=abc"));
}

#[test]
fn connect_surfaces_service_failures() {
    let runtime = Runtime::new().unwrap();
    let service = FakeBlobService::default();
    service.set_failing(true);
    let result = runtime.block_on(Fixture::connect(&service, runtime.handle().clone()));
    assert!(matches!(result, Err(Error::Unavailable)));
}

#[test]
fn sync_set_and_get_round_trip_python_json_shape() {
    let fixture = Fixture::new(FakeBlobService::default());
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

#[test]
fn sync_set_does_not_overwrite_an_existing_blob() {
    let fixture = Fixture::new(FakeBlobService::default());
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

#[test]
fn sync_set_treats_a_precondition_conflict_as_an_existing_blob() {
    let fixture = Fixture::new(FakeBlobService::default());
    fixture.service.set_precondition_conflicts(true);
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
}

#[test]
fn async_set_overwrites_an_existing_blob() {
    let fixture = Fixture::new(FakeBlobService::default());
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

#[test]
fn missing_blobs_are_misses() {
    let fixture = Fixture::new(FakeBlobService::default());
    assert_eq!(fixture.cache.get_cache("absent", &no_ttl()).unwrap(), None);
    assert_eq!(
        fixture
            .runtime
            .block_on(fixture.cache.async_get_cache("absent", &no_ttl()))
            .unwrap(),
        None
    );
}

#[test]
fn ttl_is_ignored_and_entries_never_expire() {
    let fixture = Fixture::new(FakeBlobService::default());
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

#[test]
fn malformed_blobs_are_invalid_entries_and_response_cache_misses() {
    let fixture = Fixture::new(FakeBlobService::default());
    fixture.service.seed_blob("broken-json", b"{not json");
    fixture
        .service
        .seed_blob("broken-utf8", &[0xff, 0xfe, 0x22]);
    fixture
        .service
        .seed_blob("wrong-shape", br#"{"timestamp": "yesterday"}"#);

    for key in ["broken-json", "broken-utf8", "wrong-shape"] {
        assert!(matches!(
            fixture.cache.get_cache(key, &no_ttl()),
            Err(Error::InvalidEntry)
        ));
    }

    let response_cache = fixture.response_cache();
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

#[test]
fn batch_get_preserves_order_and_marks_misses_and_invalid_entries() {
    let fixture = Fixture::new(FakeBlobService::default());
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

    let response_cache = fixture.response_cache();
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

#[test]
fn async_pipeline_writes_every_entry_with_overwrite() {
    let fixture = Fixture::new(FakeBlobService::default());
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

#[test]
fn flush_deletes_every_blob_in_the_container() {
    let fixture = Fixture::new(FakeBlobService::default());
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

#[test]
fn service_failures_map_to_unavailable() {
    let fixture = Fixture::new(FakeBlobService::default());
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

#[test]
fn test_connection_reports_container_reachability() {
    let fixture = Fixture::new(FakeBlobService::default());
    let ok = fixture
        .runtime
        .block_on(fixture.cache.test_connection())
        .unwrap();
    assert_eq!(ok.status, CacheConnectionStatus::Success);
    assert!(ok.error.is_none());

    fixture.service.set_failing(true);
    let failed = fixture
        .runtime
        .block_on(fixture.cache.test_connection())
        .unwrap();
    assert_eq!(failed.status, CacheConnectionStatus::Failed);
    assert!(failed.error.is_some());
}

#[test]
fn disconnect_is_idempotent_and_keeps_data() {
    let fixture = Fixture::new(FakeBlobService::default());
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

#[test]
fn response_cache_stores_and_reads_through_the_backend() {
    let fixture = Fixture::new(FakeBlobService::default());
    let response_cache = fixture.response_cache();
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

#[test]
fn non_object_responses_are_written_serialized_like_python() {
    let fixture = Fixture::new(FakeBlobService::default());
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
