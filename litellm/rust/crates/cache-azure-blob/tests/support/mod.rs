#![allow(dead_code)]

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use azure_core::http::{
    AsyncRawResponse, Body, ClientOptions, HttpClient, Method, Request, StatusCode, Transport,
    headers::{HeaderName, Headers},
};
use litellm_cache::{CacheCodec, Error};
use litellm_cache_azure_blob::AzureBlobCache;
use tokio::runtime::{Handle, Runtime};

pub const ACCOUNT_URL: &str = "https://example.blob.core.windows.net";
pub const CONTAINER: &str = "litellm-cache";
const IF_NONE_MATCH: HeaderName = HeaderName::from_static("if-none-match");
const ERROR_CODE: HeaderName = HeaderName::from_static("x-ms-error-code");

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedRequest {
    pub method: Method,
    pub path: String,
    pub query: String,
    pub if_none_match: Option<String>,
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
pub struct FakeBlobService {
    state: Arc<Mutex<FakeState>>,
}

impl std::fmt::Debug for FakeBlobService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FakeBlobService")
    }
}

impl FakeBlobService {
    pub fn with_existing_container() -> Self {
        let service = Self::default();
        service.state.lock().unwrap().container_exists = true;
        service
    }

    pub fn blob(&self, name: &str) -> Option<Vec<u8>> {
        self.state.lock().unwrap().blobs.get(name).cloned()
    }

    pub fn blob_names(&self) -> Vec<String> {
        self.state.lock().unwrap().blobs.keys().cloned().collect()
    }

    pub fn seed_blob(&self, name: &str, bytes: &[u8]) {
        self.state
            .lock()
            .unwrap()
            .blobs
            .insert(name.to_string(), bytes.to_vec());
    }

    pub fn set_failing(&self, failing: bool) {
        self.state.lock().unwrap().failing = failing;
    }

    pub fn set_precondition_conflicts(&self, enabled: bool) {
        self.state.lock().unwrap().precondition_conflicts = enabled;
    }

    pub fn requests(&self) -> Vec<RecordedRequest> {
        self.state.lock().unwrap().requests.clone()
    }

    pub fn container_exists(&self) -> bool {
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

pub async fn connect<C: CacheCodec>(
    service: &FakeBlobService,
    account_url: &str,
    codec: C,
    handle: Handle,
) -> Result<AzureBlobCache<C>, Error> {
    AzureBlobCache::connect_with_options(
        account_url,
        CONTAINER,
        None,
        ClientOptions {
            transport: Some(Transport::new(Arc::new(service.clone()))),
            ..ClientOptions::default()
        },
        codec,
        handle,
    )
    .await
}

/// A cache on its own fake service and runtime, so sync methods run outside any runtime.
pub struct Fixture<C> {
    pub runtime: Runtime,
    pub service: FakeBlobService,
    pub cache: Arc<AzureBlobCache<C>>,
}

impl<C: CacheCodec> Fixture<C> {
    pub fn new(service: FakeBlobService, codec: C) -> Self {
        let runtime = Runtime::new().unwrap();
        let cache = runtime
            .block_on(connect(
                &service,
                ACCOUNT_URL,
                codec,
                runtime.handle().clone(),
            ))
            .unwrap();
        Self {
            runtime,
            service,
            cache: Arc::new(cache),
        }
    }

    pub fn stored_json(&self, key: &str) -> serde_json::Value {
        serde_json::from_slice(&self.service.blob(key).expect("blob should exist")).unwrap()
    }
}
