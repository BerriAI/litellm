use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use super::*;

async fn server() -> (Url, tokio::task::JoinHandle<Vec<u8>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = Url::parse(&format!(
        "http://{}/request?version=one",
        listener.local_addr().unwrap()
    ))
    .unwrap();
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut wire = Vec::new();
        let mut buffer = [0; 4096];
        loop {
            let count = socket.read(&mut buffer).await.unwrap();
            assert_ne!(count, 0);
            wire.extend_from_slice(&buffer[..count]);
            let text = String::from_utf8_lossy(&wire);
            if let Some((headers, body)) = text.split_once("\r\n\r\n") {
                let length = headers
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length: ")
                            .and_then(|value| value.parse::<usize>().ok())
                    })
                    .unwrap_or(0);
                if body.len() >= length {
                    break;
                }
            }
        }
        socket
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}")
            .await
            .unwrap();
        wire
    });
    (url, task)
}

fn client() -> Client {
    Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap()
}

fn request(url: Url, body: OutboundBody) -> OutboundOperation {
    OutboundOperation {
        method: Method::POST,
        url,
        headers: HeaderMap::from_iter([(
            CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        )]),
        body,
        operation: OutboundOperationKind::Request,
    }
}

struct RecordingHeaderAuthorizer {
    calls: AtomicUsize,
    bodies: Mutex<Vec<Option<Vec<u8>>>>,
}

impl RequestAuthorizer for RecordingHeaderAuthorizer {
    fn declared_headers(&self) -> &[HeaderName] {
        std::slice::from_ref(&AUTHORIZATION)
    }

    fn authorize<'a>(
        &'a self,
        input: AuthorizationInput<'a>,
    ) -> OperationFuture<'a, AuthorizationMutation> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.bodies.lock().unwrap().push(input.body.map(Vec::from));
            Ok(AuthorizationMutation {
                set_headers: vec![(AUTHORIZATION, HeaderValue::from_static("Bearer recorded"))],
                remove_headers: Vec::new(),
            })
        })
    }
}

struct BodySigner;

impl RequestAuthorizer for BodySigner {
    fn declared_headers(&self) -> &[HeaderName] {
        static SIGNATURE: HeaderName = HeaderName::from_static("x-body-signature");
        std::slice::from_ref(&SIGNATURE)
    }

    fn authorize<'a>(
        &'a self,
        input: AuthorizationInput<'a>,
    ) -> OperationFuture<'a, AuthorizationMutation> {
        Box::pin(async move {
            let signature = format!("{:x}", Sha256::digest(input.body.unwrap_or_default()));
            Ok(AuthorizationMutation {
                set_headers: vec![(
                    HeaderName::from_static("x-body-signature"),
                    signature.parse().unwrap(),
                )],
                remove_headers: Vec::new(),
            })
        })
    }
}

#[tokio::test]
async fn header_authorizer_observes_final_replacement_and_send_occurs_once() {
    let (url, server) = server().await;
    let authorizer = Arc::new(RecordingHeaderAuthorizer {
        calls: AtomicUsize::new(0),
        bodies: Mutex::new(Vec::new()),
    });
    let control = OperationControl::with_timeout(Duration::from_secs(2));
    let response = send_once(
        &client(),
        &url,
        request(
            url.clone(),
            OutboundBody::JsonObject(serde_json::Map::new()),
        ),
        |view| async move {
            assert!(view.headers.get(AUTHORIZATION).is_none());
            Ok(BodyDecision::Replace(serde_json::json!({"masked": true})))
        },
        &FixedAuthorization::with_prepared_headers(
            authorizer.clone(),
            HeaderMap::from_iter([(AUTHORIZATION, HeaderValue::from_static("Bearer recorded"))]),
            Vec::new(),
        ),
        &control,
    )
    .await
    .unwrap();
    let wire = server.await.unwrap();
    assert_eq!(response.body, b"{}");
    assert_eq!(authorizer.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        authorizer.bodies.lock().unwrap().as_slice(),
        [Some(br#"{"masked":true}"#.to_vec())]
    );
    assert!(String::from_utf8_lossy(&wire).contains("authorization: Bearer recorded\r\n"));
}

#[tokio::test]
async fn body_sensitive_signer_signs_the_exact_wire_bytes() {
    let (url, server) = server().await;
    let final_body = br#"{"a":1,"z":"last"}"#;
    send_once(
        &client(),
        &url,
        request(
            url.clone(),
            OutboundBody::JsonObject(
                serde_json::json!({"private": true})
                    .as_object()
                    .unwrap()
                    .clone(),
            ),
        ),
        |_| async {
            Ok(BodyDecision::Replace(
                serde_json::json!({"a": 1, "z": "last"}),
            ))
        },
        &FixedAuthorization::new(Arc::new(BodySigner)),
        &OperationControl::with_timeout(Duration::from_secs(2)),
    )
    .await
    .unwrap();
    let wire = server.await.unwrap();
    let header_end = wire
        .windows(4)
        .position(|value| value == b"\r\n\r\n")
        .unwrap()
        + 4;
    assert_eq!(&wire[header_end..], final_body);
    let expected = format!("{:x}", Sha256::digest(final_body));
    assert!(
        String::from_utf8_lossy(&wire[..header_end])
            .contains(&format!("x-body-signature: {expected}\r\n"))
    );
}

#[tokio::test]
async fn bodyless_rejection_and_invalid_replacement_never_authorize() {
    let authorizer = Arc::new(RecordingHeaderAuthorizer {
        calls: AtomicUsize::new(0),
        bodies: Mutex::new(Vec::new()),
    });
    let url = Url::parse("http://127.0.0.1:9/").unwrap();
    for (body, decision, expected) in [
        (
            OutboundBody::Bodyless,
            BodyDecision::Replace(serde_json::json!({})),
            "bodyless",
        ),
        (
            OutboundBody::JsonObject(serde_json::Map::new()),
            BodyDecision::Reject("blocked".into()),
            "blocked",
        ),
        (
            OutboundBody::JsonObject(serde_json::Map::new()),
            BodyDecision::Replace(serde_json::json!([1, 2])),
            "JSON object",
        ),
    ] {
        let error = send_once(
            &client(),
            &url,
            request(url.clone(), body),
            |_| async { Ok(decision) },
            &FixedAuthorization::new(authorizer.clone()),
            &OperationControl::with_timeout(Duration::from_secs(1)),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains(expected));
    }
    assert_eq!(authorizer.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn bodyless_request_is_authorized_and_sent_without_a_body() {
    let (url, server) = server().await;
    let authorizer = Arc::new(RecordingHeaderAuthorizer {
        calls: AtomicUsize::new(0),
        bodies: Mutex::new(Vec::new()),
    });
    let mut operation = request(url.clone(), OutboundBody::Bodyless);
    operation.method = Method::GET;
    operation.headers.clear();
    send_once(
        &client(),
        &url,
        operation,
        |_| async { Ok(BodyDecision::Unchanged) },
        &FixedAuthorization::new(authorizer.clone()),
        &OperationControl::with_timeout(Duration::from_secs(2)),
    )
    .await
    .unwrap();
    let wire = server.await.unwrap();
    let header_end = wire
        .windows(4)
        .position(|value| value == b"\r\n\r\n")
        .unwrap()
        + 4;
    assert!(wire[header_end..].is_empty());
    assert_eq!(authorizer.bodies.lock().unwrap().as_slice(), [None]);
}

struct UndeclaredAuthorizer;

impl RequestAuthorizer for UndeclaredAuthorizer {
    fn declared_headers(&self) -> &[HeaderName] {
        &[]
    }

    fn authorize<'a>(
        &'a self,
        _: AuthorizationInput<'a>,
    ) -> OperationFuture<'a, AuthorizationMutation> {
        Box::pin(async {
            Ok(AuthorizationMutation {
                set_headers: vec![(AUTHORIZATION, HeaderValue::from_static("forbidden"))],
                remove_headers: Vec::new(),
            })
        })
    }
}

#[tokio::test]
async fn undeclared_auth_mutation_and_disallowed_destination_are_rejected() {
    let url = Url::parse("http://127.0.0.1:9/").unwrap();
    let error = send_once(
        &client(),
        &url,
        request(url.clone(), OutboundBody::Bodyless),
        |_| async { Ok(BodyDecision::Unchanged) },
        &FixedAuthorization::new(Arc::new(UndeclaredAuthorizer)),
        &OperationControl::with_timeout(Duration::from_secs(1)),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, Error::Auth(_)));

    let foreign = Url::parse("http://example.com/").unwrap();
    let authorizer = Arc::new(RecordingHeaderAuthorizer {
        calls: AtomicUsize::new(0),
        bodies: Mutex::new(Vec::new()),
    });
    let error = send_once(
        &client(),
        &url,
        request(foreign, OutboundBody::Bodyless),
        |_| async { Ok(BodyDecision::Unchanged) },
        &FixedAuthorization::new(authorizer.clone()),
        &OperationControl::with_timeout(Duration::from_secs(1)),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, Error::Auth(_)));
    assert_eq!(authorizer.calls.load(Ordering::SeqCst), 0);
}

struct PendingAuthorization;

impl AuthorizationProvider for PendingAuthorization {
    fn prepare(&self) -> OperationFuture<'_, AuthorizationPreparation> {
        Box::pin(async {
            std::future::pending::<()>().await;
            unreachable!()
        })
    }
}

#[tokio::test]
async fn deadline_and_cancellation_cover_post_callback_credential_preparation() {
    let url = Url::parse("http://127.0.0.1:9/").unwrap();
    let callback_calls = Arc::new(AtomicUsize::new(0));
    let callback_counter = callback_calls.clone();
    let error = send_once(
        &client(),
        &url,
        request(url.clone(), OutboundBody::Bodyless),
        move |_| async move {
            callback_counter.fetch_add(1, Ordering::SeqCst);
            Ok(BodyDecision::Unchanged)
        },
        &PendingAuthorization,
        &OperationControl::with_timeout(Duration::from_millis(10)),
    )
    .await
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "upstream network error: Request timed out"
    );
    assert_eq!(callback_calls.load(Ordering::SeqCst), 1);

    let control = OperationControl::with_timeout(Duration::from_secs(1));
    control.cancellation.cancel();
    let error = send_once(
        &client(),
        &url,
        request(url.clone(), OutboundBody::Bodyless),
        |_| async { Ok(BodyDecision::Unchanged) },
        &PendingAuthorization,
        &control,
    )
    .await
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "upstream network error: Request cancelled"
    );
}
