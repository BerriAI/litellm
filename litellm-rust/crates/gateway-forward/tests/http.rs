mod support;

use std::convert::Infallible;

use axum::{
    Router,
    body::{Body, Bytes},
    extract::Request,
    http::{HeaderValue, StatusCode},
    response::Response,
};
use http_body_util::BodyExt;
use litellm_gateway_forward::PEER_HEADER;
use rstest::rstest;
use support::{Gateway, client, raw_upstream, recording, serve, within};
use tokio::sync::mpsc;

fn ok() -> Response {
    Response::new(Body::from("ok"))
}

#[rstest]
#[tokio::test]
async fn forwards_the_request_as_the_client_sent_it_minus_hop_headers() {
    let (upstream, mut received) = recording(|| {
        Response::builder()
            .status(StatusCode::UNAUTHORIZED)
            .header("connection", "x-upstream-hop")
            .header("x-upstream-hop", "remove")
            .header("x-response", "keep")
            .body(Body::from("python error"))
            .unwrap()
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send(
        "PATCH /v1/messages?x=a%2Fb&x=c HTTP/1.1\r\n\
         host: public.example\r\n\
         authorization: Bearer test\r\n\
         x-forwarded-for: 192.0.2.1\r\n\
         x-forwarded-for: 192.0.2.2\r\n\
         x-forwarded-proto: https\r\n\
         forwarded: for=192.0.2.1;proto=https\r\n\
         x-litellm-rust-peer: 10.1.2.3:1\r\n\
         connection: x-remove, keep-alive\r\n\
         x-remove: remove\r\n\
         proxy-authorization: remove\r\n\
         te: trailers\r\n\
         content-length: 7\r\n\r\npayload",
    )
    .await;
    let head = within(raw.response_head()).await.unwrap();
    assert_eq!(head.status, StatusCode::UNAUTHORIZED);
    assert_eq!(head.header("x-response"), Some(&b"keep"[..]));
    assert_eq!(head.header("x-upstream-hop"), None);
    let body = raw.body(head.content_length().unwrap()).await;
    assert_eq!(body, b"python error");

    let request = received.recv().await.unwrap();
    assert_eq!(request.method, "PATCH");
    assert_eq!(request.uri, "/v1/messages?x=a%2Fb&x=c");
    assert_eq!(request.headers["host"], "public.example");
    assert_eq!(request.headers["authorization"], "Bearer test");
    let forwarded_for: Vec<_> = request.headers.get_all("x-forwarded-for").iter().collect();
    assert_eq!(forwarded_for, ["192.0.2.1", "192.0.2.2"]);
    assert_eq!(request.headers["x-forwarded-proto"], "https");
    assert_eq!(request.headers["forwarded"], "for=192.0.2.1;proto=https");
    let peers: Vec<_> = request.headers.get_all(PEER_HEADER).iter().collect();
    assert_eq!(peers, [raw.local_address().to_string().as_str()]);
    assert!(!request.headers.contains_key("x-remove"));
    assert!(!request.headers.contains_key("proxy-authorization"));
    assert!(!request.headers.contains_key("te"));
    assert!(!request.headers.contains_key("connection"));
    assert_eq!(request.body, "payload");
}

#[rstest]
#[case::get("GET")]
#[case::post("POST")]
#[case::put("PUT")]
#[case::delete("DELETE")]
#[case::options("OPTIONS")]
#[case::extension("PURGE")]
#[tokio::test]
async fn forwards_every_method(#[case] method: &str) {
    let (upstream, mut received) = recording(ok).await;
    let gateway = Gateway::start(upstream).await;
    let response = client()
        .request(
            Request::builder()
                .method(method)
                .uri(gateway.url("/mcp"))
                .body(Body::from("body"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let request = received.recv().await.unwrap();
    assert_eq!(request.method, method);
    assert_eq!(request.body, "body");
}

#[rstest]
#[case::double_slash("//v1//models")]
#[case::encoded_slash("/v1/files/a%2Fb/content")]
#[case::encoded_query("/v1/models?q=%E2%9C%93&empty=&flag")]
#[case::utf8_bytes_escaped("/mcp/%F0%9F%A6%80")]
#[tokio::test]
async fn forwards_the_request_target_byte_for_byte(#[case] target: &str) {
    let (upstream, mut received) = recording(ok).await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send(format!("GET {target} HTTP/1.1\r\nhost: x\r\n\r\n"))
        .await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::OK
    );
    assert_eq!(received.recv().await.unwrap().uri, target);
}

#[rstest]
#[tokio::test]
async fn absolute_form_targets_reach_python_as_origin_form() {
    let (upstream, mut received) = recording(ok).await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send(
        "GET http://elsewhere.example/v1/models?x=1 HTTP/1.1\r\nhost: elsewhere.example\r\n\r\n",
    )
    .await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::OK
    );
    let request = received.recv().await.unwrap();
    assert_eq!(request.uri, "/v1/models?x=1");
}

#[rstest]
#[tokio::test]
async fn http2_clients_reach_python_with_their_public_host() {
    let (upstream, mut received) = recording(ok).await;
    let gateway = Gateway::start(upstream).await;
    let stream = tokio::net::TcpStream::connect(gateway.address)
        .await
        .unwrap();
    let (mut sender, connection) = hyper::client::conn::http2::handshake(
        hyper_util::rt::TokioExecutor::new(),
        hyper_util::rt::TokioIo::new(stream),
    )
    .await
    .unwrap();
    tokio::spawn(connection);
    let request = axum::http::Request::get("http://public.example/v1/models")
        .body(Body::empty())
        .unwrap();

    let response = within(sender.send_request(request)).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        received.recv().await.unwrap().headers["host"],
        "public.example"
    );
}

#[rstest]
#[tokio::test]
async fn relays_a_response_with_more_headers_than_hyper_allows_by_default() {
    let upstream = serve(axum::Router::new().fallback(|| async {
        let headers = (0..150).map(|index| (format!("llm_provider-x-{index}"), "v"));
        let mut response = ok();
        for (name, value) in headers {
            response.headers_mut().insert(
                axum::http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                HeaderValue::from_static(value),
            );
        }
        response
    }))
    .await;
    let gateway = Gateway::start(upstream).await;

    let response = within(
        client().request(
            axum::http::Request::get(gateway.url("/v1/chat/completions"))
                .body(Body::empty())
                .unwrap(),
        ),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["llm_provider-x-149"], "v");
}

#[rstest]
#[tokio::test]
async fn keeps_repeated_response_headers_and_opaque_header_bytes() {
    let (upstream, mut received) = recording(|| {
        Response::builder()
            .header("set-cookie", "a=1; Path=/")
            .header("set-cookie", "b=2; HttpOnly")
            .header("x-opaque", HeaderValue::from_bytes(b"caf\xe9").unwrap())
            .body(Body::empty())
            .unwrap()
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send(b"GET / HTTP/1.1\r\nhost: x\r\nx-opaque: caf\xe9\r\n\r\n")
        .await;
    let head = within(raw.response_head()).await.unwrap();
    let cookies: Vec<_> = head
        .headers
        .iter()
        .filter(|(name, _)| name == "set-cookie")
        .map(|(_, value)| value.as_slice())
        .collect();
    assert_eq!(cookies, [&b"a=1; Path=/"[..], b"b=2; HttpOnly"]);
    assert_eq!(head.header("x-opaque"), Some(&b"caf\xe9"[..]));
    assert_eq!(
        received.recv().await.unwrap().headers["x-opaque"].as_bytes(),
        b"caf\xe9"
    );
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

#[rstest]
#[tokio::test]
async fn answers_pipelined_requests_in_order() {
    let upstream = serve(
        Router::new().fallback(|request: Request| async move { request.uri().path().to_owned() }),
    )
    .await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("GET /first HTTP/1.1\r\nhost: x\r\n\r\nHEAD /second HTTP/1.1\r\nhost: x\r\n\r\nGET /third HTTP/1.1\r\nhost: x\r\n\r\n")
        .await;
    let first = within(raw.response_head()).await.unwrap();
    assert_eq!(raw.body(first.content_length().unwrap()).await, b"/first");
    let second = within(raw.response_head()).await.unwrap();
    assert_eq!(second.content_length(), Some(7));
    let third = within(raw.response_head()).await.unwrap();
    assert_eq!(raw.body(third.content_length().unwrap()).await, b"/third");
}

#[rstest]
#[case::no_content(StatusCode::NO_CONTENT)]
#[case::not_modified(StatusCode::NOT_MODIFIED)]
#[tokio::test]
async fn bodiless_statuses_keep_the_connection_reusable(#[case] status: StatusCode) {
    let upstream = serve(Router::new().fallback(move |request: Request| async move {
        match request.uri().path() {
            "/next" => (StatusCode::OK, "next"),
            _ => (status, ""),
        }
    }))
    .await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("GET /empty HTTP/1.1\r\nhost: x\r\n\r\nGET /next HTTP/1.1\r\nhost: x\r\n\r\n")
        .await;
    assert_eq!(within(raw.response_head()).await.unwrap().status, status);
    let next = within(raw.response_head()).await.unwrap();
    assert_eq!(raw.body(next.content_length().unwrap()).await, b"next");
}

#[rstest]
#[tokio::test]
async fn relays_a_body_delimited_by_the_upstream_closing() {
    let upstream = raw_upstream(|mut python| async move {
        python.request_head().await.unwrap();
        python
            .send("HTTP/1.1 200 OK\r\nconnection: close\r\n\r\nuntil eof")
            .await;
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let response = client().request(Request::get(gateway.url("/")).body(Body::empty()).unwrap());
    let body = within(response).await.unwrap().into_body().collect();
    assert_eq!(within(body).await.unwrap().to_bytes(), "until eof");
}

#[rstest]
#[tokio::test]
async fn an_upstream_dying_mid_stream_fails_the_client_body_instead_of_ending_it() {
    let upstream = raw_upstream(|mut python| async move {
        python.request_head().await.unwrap();
        python
            .send("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n6\r\ndata: \r\n")
            .await;
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let response =
        within(client().request(Request::get(gateway.url("/")).body(Body::empty()).unwrap()))
            .await
            .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(within(response.into_body().collect()).await.is_err());
}

#[rstest]
#[tokio::test]
async fn non_websocket_upgrades_reach_python_as_plain_requests() {
    let (upstream, mut received) = recording(ok).await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("GET / HTTP/1.1\r\nhost: x\r\nconnection: upgrade, http2-settings\r\nupgrade: h2c\r\nhttp2-settings: AAMAAABkAAQAoAAAAAIAAAAA\r\n\r\n")
        .await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::OK
    );
    let request = received.recv().await.unwrap();
    assert!(!request.headers.contains_key("upgrade"));
    assert!(!request.headers.contains_key("connection"));
    assert!(!request.headers.contains_key("http2-settings"));
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
