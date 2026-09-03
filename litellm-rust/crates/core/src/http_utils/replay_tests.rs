use super::*;
use bytes::Bytes;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

use crate::http_utils::body::JsonPayload;

struct Received {
    method: String,
    headers: String,
    length: usize,
    digest: String,
}

async fn listener() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    (listener, url)
}

async fn serve(listener: TcpListener, replies: Vec<(u16, String, Duration)>) -> Vec<Received> {
    let mut received = Vec::new();
    for (status, location, delay) in replies {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = BufReader::new(socket);
        let mut line = String::new();
        socket.read_line(&mut line).await.unwrap();
        let method = line.split_whitespace().next().unwrap().to_owned();
        let mut headers = String::new();
        loop {
            line.clear();
            socket.read_line(&mut line).await.unwrap();
            if line == "\r\n" {
                break;
            }
            headers.push_str(&line.to_ascii_lowercase());
        }
        let length: usize = headers
            .lines()
            .find_map(|line| line.strip_prefix("content-length: "))
            .map(|length| length.parse().unwrap())
            .unwrap_or(0);
        let mut left = length;
        let mut digest = Sha256::new();
        let mut buffer = [0_u8; 8192];
        while left > 0 {
            let read = socket.read(&mut buffer[..left.min(8192)]).await.unwrap();
            assert_ne!(read, 0);
            digest.update(&buffer[..read]);
            left -= read;
            tokio::task::yield_now().await;
        }
        received.push(Received {
            method,
            headers,
            length,
            digest: format!("{:x}", digest.finalize()),
        });
        tokio::time::sleep(delay).await;
        if status == 0 {
            continue;
        }
        let location = if location.is_empty() {
            String::new()
        } else {
            format!("Location: {location}\r\n")
        };
        let response = format!(
            "HTTP/1.1 {status} Test\r\n{location}Content-Length: 2\r\nConnection: close\r\n\r\n{{}}"
        );
        let _ = socket.write_all(response.as_bytes()).await;
    }
    received
}

fn body() -> PreparedJsonBody {
    PreparedJsonBody::streamed(JsonPayload::object([(
        "audio",
        JsonPayload::Base64(Bytes::from(vec![7; 1024 * 1024 + 1])),
    )]))
    .unwrap()
}

#[tokio::test]
async fn redirects_replay_identical_bodies_resign_same_origin_and_strip_cross_origin_secrets() {
    let (first, url) = listener().await;
    let (second, next_url) = listener().await;
    let first = tokio::spawn(serve(
        first,
        vec![
            (307, "/again".into(), Duration::ZERO),
            (308, next_url, Duration::ZERO),
        ],
    ));
    let second = tokio::spawn(serve(second, vec![(200, String::new(), Duration::ZERO)]));
    let signed = AtomicUsize::new(0);
    let body = body();
    let signer = |url: &Url, digest: &str, headers: &HeaderMap| {
        assert_eq!(digest, body.sha256());
        assert_eq!(headers[CONTENT_LENGTH], body.content_length().to_string());
        signed.fetch_add(1, Ordering::SeqCst);
        let mut headers = headers.clone();
        headers.insert("authorization", HeaderValue::from_str(url.path()).unwrap());
        headers.insert("x-amz-security-token", HeaderValue::from_static("secret"));
        Ok(headers)
    };
    let response = send_json(
        &Client::new(),
        &url,
        &body,
        &[
            ("x-api-key".into(), "secret".into()),
            ("cookie".into(), "secret".into()),
        ],
        Duration::from_secs(10),
        Some(&signer),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(signed.load(Ordering::SeqCst), 2);
    let requests = first.await.unwrap();
    assert!(requests[0].headers.contains("authorization: /\r\n"));
    assert!(requests[1].headers.contains("authorization: /again\r\n"));
    let cross_origin = second.await.unwrap();
    assert!(!cross_origin[0].headers.contains("secret"));
    assert!(!cross_origin[0].headers.contains("authorization:"));
    for request in requests.into_iter().chain(cross_origin) {
        assert_eq!(request.method, "POST");
        assert_eq!(request.length as u64, body.content_length());
        assert_eq!(request.digest, body.sha256());
        assert!(!request.headers.contains("transfer-encoding:"));
    }
}

#[tokio::test]
async fn redirects_that_change_post_to_get_drop_the_body() {
    for status in [301, 302, 303] {
        let (listener, url) = listener().await;
        let server = tokio::spawn(serve(
            listener,
            vec![
                (status, "/next".into(), Duration::ZERO),
                (200, String::new(), Duration::ZERO),
            ],
        ));
        send_json(
            &Client::new(),
            &url,
            &body(),
            &[],
            Duration::from_secs(10),
            None,
        )
        .await
        .unwrap();
        let requests = server.await.unwrap();
        assert_eq!(requests[1].method, "GET");
        assert_eq!(requests[1].length, 0);
        assert!(!requests[1].headers.contains("content-type:"));
    }
}

#[tokio::test]
async fn redirects_share_a_deadline_and_have_a_limit() {
    let (listener, url) = listener().await;
    let server = tokio::spawn(serve(
        listener,
        vec![
            (307, "/next".into(), Duration::from_millis(70)),
            (200, String::new(), Duration::from_millis(70)),
        ],
    ));
    let body = PreparedJsonBody::streamed("data:test".into()).unwrap();
    let start = Instant::now();
    assert!(
        send_json(
            &Client::new(),
            &url,
            &body,
            &[],
            Duration::from_millis(110),
            None
        )
        .await
        .is_err()
    );
    assert!(start.elapsed() < Duration::from_millis(190));
    assert_eq!(server.await.unwrap().len(), 2);
    let (listener, url) = self::listener().await;
    let server = tokio::spawn(serve(
        listener,
        vec![(307, "/next".into(), Duration::ZERO); JSON_BODY_MAX_REDIRECTS + 1],
    ));
    let error = send_json(
        &Client::new(),
        &url,
        &body,
        &[],
        Duration::from_secs(10),
        None,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("too many"));
    assert_eq!(server.await.unwrap().len(), JSON_BODY_MAX_REDIRECTS + 1);
}

#[tokio::test]
async fn provider_errors_and_disconnects_are_not_retried() {
    for status in [429, 500, 0] {
        let (listener, url) = listener().await;
        let server = tokio::spawn(serve(
            listener,
            vec![(status, String::new(), Duration::ZERO)],
        ));
        let result = send_json(
            &Client::new(),
            &url,
            &body(),
            &[],
            Duration::from_secs(10),
            None,
        )
        .await;
        if status == 0 {
            assert!(result.is_err());
        } else {
            assert_eq!(result.unwrap().status().as_u16(), status);
        }
        assert_eq!(server.await.unwrap().len(), 1);
    }
}

#[tokio::test]
async fn http2_refused_streams_are_replayed_at_most_twice() {
    for failures in [2, 3] {
        let (listener, url) = listener().await;
        let attempts = std::sync::Arc::new(AtomicUsize::new(0));
        let observed = attempts.clone();
        let server = tokio::spawn(async move {
            loop {
                let (socket, _) = listener.accept().await.unwrap();
                let observed = observed.clone();
                tokio::spawn(async move {
                    let mut connection = h2::server::handshake(socket).await.unwrap();
                    while let Some(request) = connection.accept().await {
                        let Ok((_, mut response)) = request else {
                            break;
                        };
                        let attempt = observed.fetch_add(1, Ordering::SeqCst);
                        if attempt < failures {
                            response.send_reset(h2::Reason::REFUSED_STREAM);
                        } else {
                            response
                                .send_response(http::Response::new(()), true)
                                .unwrap();
                        }
                    }
                });
            }
        });
        let client = Client::builder()
            .http2_prior_knowledge()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .build()
            .unwrap();
        let result = send_json(&client, &url, &body(), &[], Duration::from_secs(10), None).await;
        if failures == 2 {
            assert_eq!(result.unwrap().status(), 200);
        } else {
            assert!(result.is_err());
        }
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        server.abort();
    }
}

#[tokio::test]
async fn cancellation_releases_media_while_a_slow_consumer_stalls_the_upload() {
    use std::sync::Arc;
    struct Owner(Arc<Vec<u8>>);
    impl AsRef<[u8]> for Owner {
        fn as_ref(&self) -> &[u8] {
            &self.0
        }
    }
    let owner = Arc::new(vec![b'A'; 16 * 1024 * 1024]);
    let body = PreparedJsonBody::streamed(JsonPayload::object([(
        "audio",
        JsonPayload::Base64(Bytes::from_owner(Owner(owner.clone()))),
    )]))
    .unwrap();
    let (listener, url) = listener().await;
    let (started, ready) = tokio::sync::oneshot::channel();
    let (close, closed) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut socket = BufReader::new(socket);
        loop {
            let mut line = String::new();
            socket.read_line(&mut line).await.unwrap();
            if line == "\r\n" {
                break;
            }
        }
        started.send(()).unwrap();
        let _ = closed.await;
    });
    let client = replay_client().unwrap().clone();
    let upload = tokio::spawn(async move {
        send_json(&client, &url, &body, &[], Duration::from_secs(30), None).await
    });
    ready.await.unwrap();
    assert_eq!(Arc::strong_count(&owner), 2);
    assert!(!upload.is_finished());
    upload.abort();
    assert!(upload.await.unwrap_err().is_cancelled());
    tokio::time::timeout(Duration::from_secs(2), async {
        while Arc::strong_count(&owner) > 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    close.send(()).unwrap();
    server.await.unwrap();
}

#[tokio::test]
async fn media_upload_uses_the_supplied_clients_configuration() {
    let (listener, url) = listener().await;
    let server = tokio::spawn(serve(listener, vec![(200, String::new(), Duration::ZERO)]));
    let mut headers = HeaderMap::new();
    headers.insert("x-client", HeaderValue::from_static("supplied"));
    let client = Client::builder()
        .default_headers(headers)
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .build()
        .unwrap();
    let response = send_json(&client, &url, &body(), &[], Duration::from_secs(5), None)
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert!(
        server.await.unwrap()[0]
            .headers
            .contains("x-client: supplied")
    );
}

#[tokio::test]
async fn http2_unprocessed_goaway_is_replayed() {
    let (listener, url) = listener().await;
    let server = tokio::spawn(async move {
        for attempt in 0..3 {
            let (socket, _) = listener.accept().await.unwrap();
            let mut connection = h2::server::handshake(socket).await.unwrap();
            if attempt < 2 {
                connection.abrupt_shutdown(h2::Reason::NO_ERROR);
            } else {
                let (_, mut response) = connection.accept().await.unwrap().unwrap();
                response
                    .send_response(http::Response::new(()), true)
                    .unwrap();
            }
            let _ = std::future::poll_fn(|cx| connection.poll_closed(cx)).await;
        }
    });
    let client = Client::builder()
        .http2_prior_knowledge()
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .build()
        .unwrap();
    let response = send_json(&client, &url, &body(), &[], Duration::from_secs(5), None)
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    drop(response);
    drop(client);
    tokio::time::timeout(Duration::from_secs(5), server)
        .await
        .unwrap()
        .unwrap();
}
