#![allow(dead_code)]

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use litellm_auth_aws::AwsAuthConfig;
use litellm_cache::JsonCodec;
use litellm_cache_s3::{S3Cache, S3CacheConfig, S3Endpoint};
use serde_json::Value;
use tokio::runtime::Handle;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate, http::Method, matchers::any};

pub type JsonS3Cache = S3Cache<JsonCodec<Value>>;

pub fn config(endpoint: &str) -> S3CacheConfig {
    S3CacheConfig {
        bucket: "cache-bucket".to_string(),
        key_prefix: "team/".to_string(),
        region: "us-east-1".to_string(),
        endpoint: Some(S3Endpoint {
            url: endpoint.to_string(),
        }),
        auth: AwsAuthConfig {
            access_key_id: Some("key".to_string()),
            secret_access_key: Some("secret".to_string()),
            region_name: Some("us-east-1".to_string()),
            ..Default::default()
        },
    }
}

pub fn cache_with(config: S3CacheConfig, runtime: Handle) -> JsonS3Cache {
    S3Cache::new(
        config,
        litellm_http::Client::plain_for_test(),
        JsonCodec::new(),
        runtime,
    )
}

pub fn cache(endpoint: &str) -> JsonS3Cache {
    cache_with(config(endpoint), Handle::current())
}

/// An in-memory bucket: PUT stores the body under the request path, GET serves it or answers
/// `NoSuchKey`.
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
        let path = request.url.path().to_string();
        let mut objects = self.objects.lock().unwrap();
        match request.method {
            Method::PUT => {
                objects.insert(path, request.body.clone());
                ResponseTemplate::new(200).insert_header("etag", "\"etag\"")
            }
            Method::GET => match objects.get(&path) {
                Some(body) => ResponseTemplate::new(200).set_body_bytes(body.clone()),
                None => ResponseTemplate::new(404)
                    .set_body_string("<Error><Code>NoSuchKey</Code></Error>"),
            },
            _ => ResponseTemplate::new(405),
        }
    }
}
