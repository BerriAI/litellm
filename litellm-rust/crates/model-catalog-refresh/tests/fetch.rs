use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_credential_types::Credentials;
use aws_credential_types::provider::error::CredentialsError;
use aws_credential_types::provider::{ProvideCredentials, future};
use litellm_http::Client;
use litellm_model_catalog_refresh::{Failed, Fetched, RetryPolicy, Start, Target, fetch};
use rstest::rstest;

const POLICY: RetryPolicy = RetryPolicy {
    max_attempts: 3,
    max_wait: Duration::from_secs(30),
    timeout: Duration::from_secs(5),
    backoff_base_seconds: 2.0,
    jitter_min_seconds: 0.0,
    jitter_max_seconds: 1.0,
};

const OK: &str =
    "HTTP/1.1 200 OK\r\nETag: \"v1\"\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";
const UNAVAILABLE: &str = "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 0\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
const NOT_FOUND: &str = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

#[derive(Clone, Copy)]
enum Answer {
    Raw(&'static str),
    Stall,
}

/// A scripted upstream: the n-th request gets the n-th answer, repeating the last, and every request head is kept
struct Upstream {
    base: String,
    heads: Arc<Mutex<Vec<String>>>,
}

impl Upstream {
    fn serve(answers: Vec<Answer>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let heads = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&heads);
        std::thread::spawn(move || {
            for (index, stream) in listener.incoming().enumerate() {
                let answer = answers[index.min(answers.len() - 1)];
                let recorded = Arc::clone(&recorded);
                std::thread::spawn(move || {
                    let mut stream = stream.unwrap();
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut head = String::new();
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).unwrap() <= 2 {
                            break;
                        }
                        head.push_str(&line);
                    }
                    recorded.lock().unwrap().push(head);
                    match answer {
                        Answer::Raw(raw) => stream.write_all(raw.as_bytes()).unwrap(),
                        Answer::Stall => std::thread::sleep(Duration::from_secs(5)),
                    }
                });
            }
        });
        Self { base, heads }
    }

    fn url(&self) -> String {
        format!("{}/cost.json", self.base)
    }

    fn heads(&self) -> Vec<String> {
        self.heads.lock().unwrap().clone()
    }
}

fn header<'a>(head: &'a str, name: &str) -> Option<&'a str> {
    head.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.eq_ignore_ascii_case(name).then(|| value.trim())
    })
}

fn target(url: &str) -> Target {
    Target {
        url: url.to_owned(),
        aws_region: "us-west-2".into(),
        aws_endpoint_url: None,
    }
}

#[derive(Clone, Debug)]
struct NoCredentials;

impl ProvideCredentials for NoCredentials {
    fn provide_credentials<'a>(&'a self) -> future::ProvideCredentials<'a>
    where
        Self: 'a,
    {
        future::ProvideCredentials::ready(Err(CredentialsError::not_loaded("no credentials")))
    }
}

fn keys() -> Credentials {
    Credentials::new(
        "AKIDMODELCOST",
        "model-cost-secret",
        None,
        None,
        "model-cost-test",
    )
}

async fn fetched(target: &Target, policy: RetryPolicy, start: Start) -> Result<Fetched, Failed> {
    fetch(&Client::for_test(), &NoCredentials, target, policy, start).await
}

#[rstest]
#[case::one_etag(
    "HTTP/1.1 200 OK\r\nETag: \"v1\"\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
    Some("\"v1\"")
)]
#[case::every_etag_joined(
    "HTTP/1.1 200 OK\r\nETag: \"a\"\r\nETag: \"b\"\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
    Some("\"a\", \"b\"")
)]
#[case::no_etag(
    "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
    None
)]
#[tokio::test]
async fn a_200_returns_the_body_and_its_etags(
    #[case] response: &'static str,
    #[case] etag: Option<&str>,
) {
    let upstream = Upstream::serve(vec![Answer::Raw(response)]);
    assert_eq!(
        fetched(&target(&upstream.url()), POLICY, Start::Fresh).await,
        Ok(Fetched {
            body: b"{}".to_vec(),
            etag: etag.map(str::to_owned),
        })
    );
    assert_eq!(upstream.heads().len(), 1);
}

#[rstest]
#[tokio::test]
async fn a_redirect_is_not_followed_and_its_body_is_the_document() {
    let upstream = Upstream::serve(vec![
        Answer::Raw(
            "HTTP/1.1 301 Moved Permanently\r\nLocation: /elsewhere.json\r\nContent-Length: 7\r\nConnection: close\r\n\r\n{\"a\":1}",
        ),
        Answer::Raw(OK),
    ]);
    assert_eq!(
        fetched(&target(&upstream.url()), POLICY, Start::Fresh).await,
        Ok(Fetched {
            body: b"{\"a\":1}".to_vec(),
            etag: None,
        })
    );
    assert_eq!(upstream.heads().len(), 1);
}

#[rstest]
#[tokio::test]
async fn a_retryable_status_is_retried_until_it_succeeds() {
    let upstream = Upstream::serve(vec![Answer::Raw(UNAVAILABLE), Answer::Raw(OK)]);
    assert!(
        fetched(&target(&upstream.url()), POLICY, Start::Fresh)
            .await
            .is_ok()
    );
    assert_eq!(upstream.heads().len(), 2);
}

#[rstest]
#[tokio::test]
async fn retryable_failures_stop_at_the_attempt_limit() {
    let upstream = Upstream::serve(vec![Answer::Raw(UNAVAILABLE)]);
    let url = upstream.url();
    assert_eq!(
        fetched(&target(&url), POLICY, Start::Fresh).await,
        Err(Failed {
            reason: format!("HTTP 503 from {url} (after 3 attempts)"),
            retryable: true,
            retry_after: Some(0.0),
        })
    );
    assert_eq!(upstream.heads().len(), 3);
}

#[rstest]
#[tokio::test]
async fn a_404_fails_on_the_first_attempt() {
    let upstream = Upstream::serve(vec![Answer::Raw(NOT_FOUND)]);
    let url = upstream.url();
    assert_eq!(
        fetched(&target(&url), POLICY, Start::Fresh).await,
        Err(Failed {
            reason: format!("HTTP 404 from {url}"),
            retryable: false,
            retry_after: None,
        })
    );
    assert_eq!(upstream.heads().len(), 1);
}

#[rstest]
#[tokio::test]
async fn a_stalled_response_times_out_and_is_retried() {
    let upstream = Upstream::serve(vec![Answer::Stall]);
    let url = upstream.url();
    let policy = RetryPolicy {
        max_attempts: 2,
        timeout: Duration::from_millis(100),
        backoff_base_seconds: 0.0,
        jitter_max_seconds: 0.0,
        ..POLICY
    };
    let failed = fetched(&target(&url), policy, Start::Fresh)
        .await
        .unwrap_err();
    assert!(
        failed
            .reason
            .starts_with(&format!("error fetching {url}: "))
            && failed.reason.ends_with(" (after 2 attempts)"),
        "{}",
        failed.reason
    );
    assert!(failed.retryable);
    assert_eq!(upstream.heads().len(), 2);
}

#[rstest]
#[tokio::test]
async fn a_resumed_fetch_makes_only_the_attempts_left() {
    let upstream = Upstream::serve(vec![Answer::Raw(UNAVAILABLE)]);
    let url = upstream.url();
    let first = Failed {
        reason: "first".into(),
        retryable: true,
        retry_after: Some(0.0),
    };
    let failed = fetched(&target(&url), POLICY, Start::After(first))
        .await
        .unwrap_err();
    assert_eq!(
        failed.reason,
        format!("HTTP 503 from {url} (after 3 attempts)")
    );
    assert_eq!(upstream.heads().len(), 2);
}

#[rstest]
#[tokio::test]
async fn a_resumed_fetch_with_no_attempts_left_returns_the_failure_it_resumed() {
    let upstream = Upstream::serve(vec![Answer::Raw(OK)]);
    let first = Failed {
        reason: "first".into(),
        retryable: true,
        retry_after: None,
    };
    let policy = RetryPolicy {
        max_attempts: 1,
        ..POLICY
    };
    assert_eq!(
        fetched(
            &target(&upstream.url()),
            policy,
            Start::After(first.clone())
        )
        .await,
        Err(first)
    );
    assert!(upstream.heads().is_empty());
}

#[rstest]
#[case::unsupported_scheme("ftp://example.com/cost.json")]
#[case::not_a_url("not a url")]
#[case::s3_without_a_key("s3://prices")]
#[tokio::test]
async fn an_unfetchable_url_fails_without_a_request(#[case] url: &str) {
    let failed = fetched(&target(url), POLICY, Start::Fresh)
        .await
        .unwrap_err();
    assert!(
        failed
            .reason
            .starts_with(&format!("error fetching {url}: ")),
        "{}",
        failed.reason
    );
    assert!(!failed.retryable);
}

fn s3_target(url: &str, upstream: &Upstream) -> Target {
    Target {
        aws_endpoint_url: Some(upstream.base.clone()),
        ..target(url)
    }
}

async fn fetched_from_s3(
    target: &Target,
    credentials: &(impl ProvideCredentials + Clone + 'static),
) -> Result<Fetched, Failed> {
    fetch(
        &Client::for_test(),
        credentials,
        target,
        POLICY,
        Start::Fresh,
    )
    .await
}

#[rstest]
#[tokio::test]
async fn an_s3_fetch_sends_a_signed_get_object_to_the_endpoint() {
    let upstream = Upstream::serve(vec![Answer::Raw(OK)]);
    assert_eq!(
        fetched_from_s3(&s3_target("s3://prices/maps/cost.json", &upstream), &keys()).await,
        Ok(Fetched {
            body: b"{}".to_vec(),
            etag: Some("\"v1\"".into()),
        })
    );
    let heads = upstream.heads();
    let [head] = heads.as_slice() else {
        panic!("expected one request, got {heads:?}");
    };
    assert!(head.starts_with("GET /prices/maps/cost.json"), "{head}");
    let authorization = header(head, "authorization").unwrap_or_default();
    assert!(
        authorization.starts_with("AWS4-HMAC-SHA256 Credential=AKIDMODELCOST/")
            && authorization.contains("/us-west-2/s3/aws4_request"),
        "{authorization}"
    );
}

#[rstest]
#[case::missing_key(NOT_FOUND)]
#[case::denied("HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")]
#[tokio::test]
async fn an_s3_error_fails_without_a_retry(#[case] response: &'static str) {
    let upstream = Upstream::serve(vec![Answer::Raw(response), Answer::Raw(OK)]);
    let url = "s3://prices/cost.json";
    let failed = fetched_from_s3(&s3_target(url, &upstream), &keys())
        .await
        .unwrap_err();
    assert!(
        failed
            .reason
            .starts_with(&format!("S3 error fetching {url}: ")),
        "{}",
        failed.reason
    );
    assert!(!failed.retryable);
    assert_eq!(upstream.heads().len(), 1);
}

#[rstest]
#[tokio::test]
async fn an_s3_fetch_without_credentials_fails_without_a_request() {
    let upstream = Upstream::serve(vec![Answer::Raw(OK)]);
    let url = "s3://prices/cost.json";
    let failed = fetched_from_s3(&s3_target(url, &upstream), &NoCredentials)
        .await
        .unwrap_err();
    assert!(
        failed
            .reason
            .starts_with(&format!("S3 error fetching {url}: ")),
        "{}",
        failed.reason
    );
    assert!(!failed.retryable);
    assert!(upstream.heads().is_empty());
}
