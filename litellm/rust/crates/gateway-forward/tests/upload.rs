mod support;

use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use axum::{
    Router,
    body::{Body, Bytes},
    extract::Request,
    http::StatusCode,
    response::Response,
};
use http_body_util::BodyExt;
use rstest::rstest;
use support::{Gateway, Raw, client, raw_upstream, recording, serve, within};
use tokio::{
    io::AsyncWriteExt,
    sync::{Notify, mpsc},
};

fn ok() -> Response {
    Response::new(Body::from("ok"))
}

#[rstest]
#[tokio::test]
async fn streams_upload_and_response_incrementally_without_a_body_limit() {
    let upstream = serve(
        Router::new().fallback(|request: Request| async { Response::new(request.into_body()) }),
    )
    .await;
    let gateway = Gateway::start(upstream).await;
    let (sender, receiver) = mpsc::channel::<Result<Bytes, Infallible>>(1);
    let stream = futures_util::stream::unfold(receiver, |mut receiver| async {
        receiver.recv().await.map(|chunk| (chunk, receiver))
    });
    sender.send(Ok(Bytes::from_static(b"first"))).await.unwrap();
    let mut response = within(
        client().request(
            Request::post(gateway.url("/v1/files"))
                .body(Body::from_stream(stream))
                .unwrap(),
        ),
    )
    .await
    .unwrap();
    let frame = within(response.body_mut().frame()).await.unwrap().unwrap();
    assert_eq!(frame.into_data().unwrap(), "first");
    let large = Bytes::from(vec![b'x'; 8 * 1024 * 1024]);
    sender.send(Ok(large.clone())).await.unwrap();
    drop(sender);
    let rest = within(response.into_body().collect())
        .await
        .unwrap()
        .to_bytes();
    assert_eq!(rest, large);
}

/// hyper sends the 100 as soon as the request is forwarded, where uvicorn waits for the app to
/// read the body, so an upload Python rejects on headers alone is still sent in full.
#[rstest]
#[tokio::test]
async fn answers_expect_continue_so_large_uploads_start() {
    let (upstream, mut received) = recording(ok).await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("POST /v1/audio/transcriptions HTTP/1.1\r\nhost: x\r\nexpect: 100-continue\r\ncontent-length: 4\r\n\r\n")
        .await;
    let interim = within(raw.response_head()).await.unwrap();
    assert_eq!(interim.status, StatusCode::CONTINUE);
    raw.send("data").await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::OK
    );
    assert_eq!(received.recv().await.unwrap().body, "data");
}

/// curl sends `expect: 100-continue` for any body over 1 MiB and waits for the 100 before sending
/// the multipart body, which must reach Python intact even where it looks like framing.
#[rstest]
#[tokio::test]
async fn expect_continue_delivers_a_large_multipart_upload_byte_for_byte() {
    let (upstream, mut received) = recording(ok).await;
    let gateway = Gateway::start(upstream).await;
    let boundary = "------------------------d74496d66958873e";
    let audio: Vec<u8> = (0..3 * 1024 * 1024)
        .map(|index: u32| (index.wrapping_mul(2_654_435_761) >> 24) as u8)
        .chain(*b"\r\n0\r\n\r\nPOST /smuggled HTTP/1.1\r\n\r\n--")
        .collect();
    let body = [
        format!("--{boundary}\r\ncontent-disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n--{boundary}\r\ncontent-disposition: form-data; name=\"file\"; filename=\"a b.wav\"\r\ncontent-type: audio/wav\r\n\r\n").into_bytes(),
        audio,
        format!("\r\n--{boundary}--\r\n").into_bytes(),
    ]
    .concat();
    let content_type = format!("multipart/form-data; boundary={boundary}");
    let mut raw = gateway.connect().await;
    raw.send(format!(
        "POST /v1/audio/transcriptions HTTP/1.1\r\nhost: x\r\ncontent-type: {content_type}\r\nexpect: 100-continue\r\ncontent-length: {}\r\n\r\n",
        body.len()
    ))
    .await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::CONTINUE
    );
    raw.send(&body).await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::OK
    );
    let request = received.recv().await.unwrap();
    assert_eq!(request.headers["content-type"], content_type.as_str());
    assert_eq!(request.body, body);
}

#[rstest]
#[case::get("GET")]
#[case::delete("DELETE")]
#[case::options("OPTIONS")]
#[case::post("POST")]
#[tokio::test]
async fn forwards_a_chunked_body_on_any_method(#[case] method: &str) {
    let (upstream, mut received) = recording(ok).await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send(format!(
        "{method} /v1/search HTTP/1.1\r\nhost: x\r\ntransfer-encoding: chunked\r\n\r\n4\r\nbody\r\n0\r\n\r\n"
    ))
    .await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::OK
    );
    assert_eq!(received.recv().await.unwrap().body, "body");
}

#[rstest]
#[case::no_framing("", "", &[], "")]
#[case::empty_content_length("content-length: 0\r\n", "", &[("content-length", "0")], "")]
#[case::empty_chunked(
    "transfer-encoding: chunked\r\n",
    "0\r\n\r\n",
    &[("transfer-encoding", "chunked")],
    ""
)]
#[case::sized(
    "content-length: 7\r\n",
    "payload",
    &[("content-length", "7")],
    "payload"
)]
#[case::chunk_extensions_and_trailers(
    "transfer-encoding: chunked\r\n",
    "4;name=\"v\"\r\nbody\r\n0\r\nx-checksum: 1\r\n\r\n",
    &[("transfer-encoding", "chunked")],
    "body"
)]
#[tokio::test]
async fn a_body_reaches_python_with_the_framing_the_client_chose(
    #[values("POST", "GET")] method: &str,
    #[case] framing: &str,
    #[case] wire_body: &str,
    #[case] python_framing: &[(&str, &str)],
    #[case] python_body: &str,
) {
    let (upstream, mut received) = recording(ok).await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send(format!(
        "{method} /v1/files HTTP/1.1\r\nhost: x\r\n{framing}\r\n{wire_body}"
    ))
    .await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::OK
    );

    let request = received.recv().await.unwrap();
    let forwarded: Vec<(&str, &str)> = ["content-length", "transfer-encoding"]
        .into_iter()
        .flat_map(|name| {
            request
                .headers
                .get_all(name)
                .iter()
                .map(move |value| (name, value.to_str().unwrap()))
        })
        .collect();
    assert_eq!(request.method, method);
    assert_eq!(forwarded, python_framing);
    assert_eq!(request.body, python_body);
}

#[rstest]
#[tokio::test]
async fn a_large_upload_is_paced_by_python_reading_it() {
    const LENGTH: usize = 128 * 1024 * 1024;
    let start_reading = Arc::new(Notify::new());
    let reading = start_reading.clone();
    let upstream = raw_upstream(move |mut python| {
        let reading = reading.clone();
        async move {
            let head = python.request_head().await.unwrap();
            reading.notified().await;
            let mut remaining = head.content_length().unwrap();
            while remaining > 0 {
                remaining -= python.body(remaining.min(1 << 20)).await.len();
            }
            python
                .send("HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n")
                .await;
        }
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let (read, mut write) = gateway.connect().await.into_stream().into_split();
    let sent = Arc::new(AtomicUsize::new(0));
    let sending = sent.clone();
    let upload = tokio::spawn(async move {
        write
            .write_all(
                format!("POST /v1/files HTTP/1.1\r\nhost: x\r\ncontent-length: {LENGTH}\r\n\r\n")
                    .as_bytes(),
            )
            .await
            .unwrap();
        let chunk = [0x5a; 64 * 1024];
        while sending.load(Ordering::SeqCst) < LENGTH {
            write.write_all(&chunk).await.unwrap();
            sending.fetch_add(chunk.len(), Ordering::SeqCst);
        }
        write
    });

    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(
        sent.load(Ordering::SeqCst) < LENGTH / 4,
        "the forwarder buffered an upload Python was not reading"
    );
    start_reading.notify_one();
    let write = tokio::time::timeout(Duration::from_secs(60), upload)
        .await
        .unwrap()
        .unwrap();
    let mut raw = Raw::new(read.reunite(write).unwrap());
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::OK
    );
}

#[rstest]
#[case::between_chunks("5\r\nhello\r\n")]
#[case::mid_chunk("100\r\nhello")]
#[tokio::test]
async fn a_client_abandoning_a_chunked_upload_never_ends_python_body_cleanly(
    #[case] sent: &'static str,
) {
    let (forwarded, mut received) = mpsc::unbounded_channel();
    let upstream = raw_upstream(move |mut python| {
        let forwarded = forwarded.clone();
        async move {
            python.request_head().await.unwrap();
            forwarded.send(python.until_closed().await).unwrap();
        }
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send(format!(
        "POST /v1/files HTTP/1.1\r\nhost: x\r\ntransfer-encoding: chunked\r\n\r\n{sent}"
    ))
    .await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    drop(raw);

    let body = within(received.recv()).await.unwrap();
    assert!(body.starts_with(b"5\r\nhello\r\n"));
    assert!(
        !body.ends_with(b"0\r\n\r\n"),
        "python saw a truncated upload end cleanly"
    );
}

#[rstest]
#[tokio::test]
async fn an_early_answer_during_an_upload_keeps_both_connections_in_sync() {
    let (event, mut events) = mpsc::unbounded_channel();
    let upstream = raw_upstream(move |mut python| {
        let event = event.clone();
        async move {
            while let Some(head) = python.request_head().await {
                let length = head.content_length();
                event.send(head.path.into_bytes()).unwrap();
                let Some(length) = length else {
                    python
                        .send("HTTP/1.1 200 OK\r\ncontent-length: 4\r\n\r\nnext")
                        .await;
                    continue;
                };
                python
                    .send("HTTP/1.1 413 Payload Too Large\r\ncontent-length: 2\r\n\r\n{}")
                    .await;
                event.send(python.body(length).await).unwrap();
            }
        }
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let upload = b"GET /smuggled HTTP/1.1\r\nhost: x\r\n\r\n".repeat(1024);
    let (first_half, second_half) = upload.split_at(upload.len() / 2);
    let mut raw = gateway.connect().await;
    raw.send(format!(
        "POST /v1/files HTTP/1.1\r\nhost: x\r\ncontent-length: {}\r\n\r\n",
        upload.len()
    ))
    .await;
    raw.send(first_half).await;

    let rejected = within(raw.response_head()).await.unwrap();
    assert_eq!(rejected.status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(raw.body(2).await, b"{}");
    raw.send(second_half).await;
    raw.send("GET /next HTTP/1.1\r\nhost: x\r\n\r\n").await;
    let next = within(raw.response_head()).await.unwrap();
    assert_eq!(raw.body(next.content_length().unwrap()).await, b"next");

    assert_eq!(within(events.recv()).await.unwrap(), b"/v1/files");
    assert_eq!(within(events.recv()).await.unwrap(), upload);
    assert_eq!(within(events.recv()).await.unwrap(), b"/next");
}
