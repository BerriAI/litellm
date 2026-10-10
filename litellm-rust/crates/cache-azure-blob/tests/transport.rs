use std::sync::Arc;

use azure_core::http::{ClientOptions, Transport};
use litellm_cache::{BaseCache, ExactCacheContext, JsonCodec};
use litellm_cache_azure_blob::{AzureBlobCache, ReqwestTransport};
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use tokio::runtime::Handle;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header, method, path, query_param},
};

#[fixture]
async fn server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path("/litellm-cache"))
        .and(query_param("restype", "container"))
        .respond_with(ResponseTemplate::new(201))
        .expect(1)
        .mount(&server)
        .await;
    server
}

async fn connect(server: &MockServer) -> AzureBlobCache<JsonCodec<Value>> {
    AzureBlobCache::connect_with_options(
        &server.uri(),
        "litellm-cache",
        None,
        ClientOptions {
            transport: Some(Transport::new(Arc::new(ReqwestTransport(
                litellm_http::Client::plain_for_test(),
            )))),
            ..ClientOptions::default()
        },
        JsonCodec::new(),
        Handle::current(),
    )
    .await
    .unwrap()
}

#[rstest]
#[tokio::test(flavor = "multi_thread")]
async fn uploads_go_through_the_host_client(#[future(awt)] server: MockServer) {
    Mock::given(method("PUT"))
        .and(path("/litellm-cache/key"))
        .and(header("if-none-match", "*"))
        .and(body_json(json!({"answer": 1})))
        .respond_with(ResponseTemplate::new(201))
        .expect(1)
        .mount(&server)
        .await;
    connect(&server)
        .await
        .set_cache("key", json!({"answer": 1}), &ExactCacheContext::default())
        .unwrap();
}

#[rstest]
#[case::hit(
    ResponseTemplate::new(200).set_body_json(json!({"answer": 2})),
    Some(json!({"answer": 2}))
)]
#[case::blob_not_found(
    ResponseTemplate::new(404).insert_header("x-ms-error-code", "BlobNotFound"),
    None
)]
#[tokio::test(flavor = "multi_thread")]
async fn downloads_map_the_host_client_response(
    #[future(awt)] server: MockServer,
    #[case] response: ResponseTemplate,
    #[case] expected: Option<Value>,
) {
    Mock::given(method("GET"))
        .and(path("/litellm-cache/key"))
        .respond_with(response)
        .mount(&server)
        .await;
    assert_eq!(
        connect(&server)
            .await
            .async_get_cache("key", &ExactCacheContext::default())
            .await
            .unwrap(),
        expected
    );
}
