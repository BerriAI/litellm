#![allow(dead_code)]

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use litellm_cache::JsonCodec;
use litellm_cache_gcs::{GcsCache, GcsConfig, StaticTokenSource, TokenSource};
use percent_encoding::percent_decode_str;
use serde_json::Value;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate, http::Method, matchers::any};

pub type JsonGcsCache = GcsCache<JsonCodec<Value>>;

pub fn config(server: &MockServer, gcs_path: Option<&str>) -> GcsConfig {
    GcsConfig {
        bucket_name: "bucket".into(),
        gcs_path: gcs_path.map(str::to_string),
        path_service_account: None,
        endpoint: server.uri(),
    }
}

pub fn cache_with_token(
    server: &MockServer,
    gcs_path: Option<&str>,
    token: Arc<dyn TokenSource>,
) -> JsonGcsCache {
    GcsCache::with_token_source(
        config(server, gcs_path),
        reqwest::Client::new(),
        JsonCodec::new(),
        token,
    )
}

pub fn cache(server: &MockServer, gcs_path: Option<&str>) -> JsonGcsCache {
    cache_with_token(server, gcs_path, Arc::new(StaticTokenSource("tok".into())))
}

/// An in-memory bucket speaking the JSON API's media upload and `alt=media` download.
#[derive(Clone, Default)]
pub struct FakeBucket {
    objects: Arc<Mutex<HashMap<String, Vec<u8>>>>,
}

impl FakeBucket {
    pub async fn serve() -> MockServer {
        let server = MockServer::start().await;
        Mock::given(any())
            .respond_with(Self::default())
            .mount(&server)
            .await;
        server
    }
}

impl Respond for FakeBucket {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let mut objects = self.objects.lock().unwrap();
        match request.method {
            Method::POST => {
                let name = request
                    .url
                    .query_pairs()
                    .find_map(|(key, value)| (key == "name").then(|| value.into_owned()))
                    .expect("uploads carry the object name");
                objects.insert(name, request.body.clone());
                ResponseTemplate::new(200)
            }
            Method::GET => {
                let encoded = request
                    .url
                    .path()
                    .strip_prefix("/storage/v1/b/bucket/o/")
                    .expect("downloads address an object");
                let name = percent_decode_str(encoded).decode_utf8().unwrap();
                match objects.get(name.as_ref()) {
                    Some(body) => ResponseTemplate::new(200).set_body_bytes(body.clone()),
                    None => ResponseTemplate::new(404),
                }
            }
            _ => ResponseTemplate::new(405),
        }
    }
}
