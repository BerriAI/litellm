use std::time::Duration;

use aws_credential_types::Credentials;
use aws_sdk_s3::config::SharedCredentialsProvider;
use litellm_model_catalog::BUNDLED_REGISTRY;
use litellm_model_catalog_refresh::{
    Installed, Origin, RetryPolicy, S3Object, S3Source, Source, Step, Transports, fetch_with_retry,
    git_blob_id,
};
use litellm_s3::S3ClientConfig;
use rstest::rstest;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn installed(origin: Origin, etag: Option<&str>) -> Installed {
    Installed {
        origin,
        etag: etag.map(str::to_owned),
        revision: git_blob_id(BUNDLED_REGISTRY),
        model_count: 0,
    }
}

async fn no_sleep(_: Duration) {}

fn s3_source(server: &MockServer, object: S3Object) -> S3Source {
    S3Source::new(
        object,
        litellm_http::Client::for_test(),
        SharedCredentialsProvider::new(Credentials::new("key", "secret", None, None, "test")),
        S3ClientConfig {
            region: "us-east-1".into(),
            endpoint: Some(server.uri()),
        },
    )
}

#[rstest]
#[tokio::test]
async fn http_retries_a_503_then_adopts_the_map_with_its_etag() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/map.json"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/map.json"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("ETag", "W/\"v2\"")
                .set_body_bytes(BUNDLED_REGISTRY),
        )
        .mount(&server)
        .await;
    let source = Source::parse(&format!("{}/map.json", server.uri())).unwrap();
    let http = litellm_http::Client::for_test();

    let step = fetch_with_retry(
        &source,
        &installed(Origin::Bundled, None),
        RetryPolicy::default(),
        Some(0),
        &Transports {
            http: &http,
            s3: None,
        },
        no_sleep,
    )
    .await;

    let Step::Adopt { installed, .. } = step else {
        panic!("expected adopt, got {step:?}");
    };
    assert_eq!(installed.etag.as_deref(), Some("W/\"v2\""));
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[rstest]
#[tokio::test]
async fn s3_retries_slow_down_and_sends_if_none_match() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/bucket/prices/map.json"))
        .and(header("if-none-match", "\"v1\""))
        .respond_with(ResponseTemplate::new(503).set_body_string(
            "<Error><Code>SlowDown</Code><Message>Please reduce your request rate.</Message></Error>",
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/bucket/prices/map.json"))
        .and(header("if-none-match", "\"v1\""))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("ETag", "\"v2\"")
                .set_body_bytes(BUNDLED_REGISTRY),
        )
        .mount(&server)
        .await;
    let object = S3Object {
        bucket: "bucket".into(),
        key: "prices/map.json".into(),
    };
    let s3 = s3_source(&server, object.clone());
    let source = Source::S3(object);
    let http = litellm_http::Client::for_test();

    let step = fetch_with_retry(
        &source,
        &installed(
            Origin::Remote {
                source: source.clone(),
            },
            Some("\"v1\""),
        ),
        RetryPolicy::default(),
        Some(0),
        &Transports {
            http: &http,
            s3: Some(&s3),
        },
        no_sleep,
    )
    .await;

    let Step::Adopt { installed, .. } = step else {
        panic!("expected adopt, got {step:?}");
    };
    assert_eq!(installed.etag.as_deref(), Some("\"v2\""));
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
}

#[rstest]
#[tokio::test]
async fn s3_not_found_is_final() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(404).set_body_string("<Error><Code>NoSuchKey</Code></Error>"),
        )
        .mount(&server)
        .await;
    let object = S3Object {
        bucket: "bucket".into(),
        key: "missing.json".into(),
    };
    let s3 = s3_source(&server, object.clone());
    let http = litellm_http::Client::for_test();

    let step = fetch_with_retry(
        &Source::S3(object),
        &installed(Origin::Bundled, None),
        RetryPolicy::default(),
        Some(0),
        &Transports {
            http: &http,
            s3: Some(&s3),
        },
        no_sleep,
    )
    .await;

    let Step::Done { error, .. } = step else {
        panic!("expected done, got {step:?}");
    };
    assert_eq!(error.status(), Some(404));
    assert!(!error.transient());
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[rstest]
#[tokio::test]
async fn a_file_source_is_read_without_a_transport() {
    let directory = std::env::temp_dir().join(format!("catalog-refresh-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let file = directory.join("map.json");
    std::fs::write(&file, BUNDLED_REGISTRY).unwrap();
    let http = litellm_http::Client::for_test();

    let step = fetch_with_retry(
        &Source::File(file.clone()),
        &installed(Origin::Bundled, None),
        RetryPolicy::default(),
        Some(0),
        &Transports {
            http: &http,
            s3: None,
        },
        no_sleep,
    )
    .await;

    std::fs::remove_dir_all(&directory).unwrap();
    assert!(matches!(step, Step::Adopt { .. }), "{step:?}");
}
