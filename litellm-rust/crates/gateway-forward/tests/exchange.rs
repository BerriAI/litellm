mod support;

use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use axum::{
    Router,
    body::{Body, Bytes},
    extract::Request,
    http::{HeaderValue, StatusCode},
    response::Response,
};
use litellm_gateway_forward::PEER_HEADER;
use rstest::rstest;
use support::{Gateway, Head, client, raw_upstream, recording, serve, within};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    sync::mpsc,
};

fn ok() -> Response {
    Response::new(Body::from("ok"))
}

/// An upstream that reports each bodiless request head exactly as it arrived on the wire.
async fn heads() -> (std::net::SocketAddr, mpsc::UnboundedReceiver<Head>) {
    let (sender, receiver) = mpsc::unbounded_channel();
    let address = raw_upstream(move |mut python| {
        let sender = sender.clone();
        async move {
            while let Some(head) = python.request_head().await {
                sender.send(head).unwrap();
                python
                    .send("HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nok")
                    .await;
            }
        }
    })
    .await;
    (address, receiver)
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
    assert!(!request.headers.contains_key("x-remove"));
    assert!(!request.headers.contains_key("proxy-authorization"));
    assert!(!request.headers.contains_key("te"));
    assert!(!request.headers.contains_key("connection"));
    assert_eq!(request.body, "payload");
}

#[rstest]
#[case::host("HoSt", "public.example")]
#[case::length("Content-Length", "7")]
#[case::authorization("AUTHORIZATION", "Bearer test")]
#[case::cookie("Cookie", "session=test")]
#[case::api_key("X-Api-Key", "test")]
#[case::azure_key("Api-Key", "test")]
#[case::litellm_key("Litellm-Api-Key", "test")]
#[case::forwarded("Forwarded", "for=192.0.2.1;proto=https")]
#[case::forwarded_for("X-Forwarded-For", "192.0.2.1")]
#[case::forwarded_proto("X-Forwarded-Proto", "https")]
#[case::forwarded_host("X-Forwarded-Host", "public.example")]
#[case::content_type("Content-Type", "application/octet-stream")]
#[case::content_encoding("Content-Encoding", "identity")]
#[tokio::test]
async fn connection_cannot_remove_end_to_end_headers(#[case] name: &str, #[case] value: &str) {
    let (upstream, mut received) = recording(ok).await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    let host = if name.eq_ignore_ascii_case("host") {
        ""
    } else {
        "host: public.example\r\n"
    };
    let length = if name.eq_ignore_ascii_case("content-length") {
        ""
    } else {
        "content-length: 7\r\n"
    };
    raw.send(format!(
        "POST / HTTP/1.1\r\n{host}{length}{name}: {value}\r\nconnection: x-remove\r\nconnection: {name}, keep-alive\r\nx-remove: remove\r\n\r\npayload"
    )).await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::OK
    );
    let request = within(received.recv()).await.unwrap();
    assert_eq!(request.headers[name], value);
    assert_eq!(request.body, "payload");
    assert!(!request.headers.contains_key("connection"));
    assert!(!request.headers.contains_key("x-remove"));
}

#[rstest]
#[tokio::test]
async fn connection_cannot_remove_response_credentials_or_representation_metadata() {
    let (upstream, _received) = recording(|| {
        Response::builder()
            .status(StatusCode::UNAUTHORIZED)
            .header("connection", "Content-Length, Set-Cookie, WWW-Authenticate, Authentication-Info, Content-Type, Content-Encoding, x-remove")
            .header("content-length", "2")
            .header("set-cookie", "session=; Max-Age=0")
            .header("www-authenticate", "Bearer")
            .header("authentication-info", "nextnonce=test")
            .header("content-type", "text/plain")
            .header("content-encoding", "identity")
            .header("x-remove", "remove")
            .body(Body::from("ok"))
            .unwrap()
    }).await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("GET / HTTP/1.1\r\nhost: x\r\n\r\n").await;
    let head = within(raw.response_head()).await.unwrap();
    assert_eq!(head.status, StatusCode::UNAUTHORIZED);
    assert_eq!(head.content_length(), Some(2));
    assert_eq!(head.header("set-cookie"), Some(&b"session=; Max-Age=0"[..]));
    assert_eq!(head.header("www-authenticate"), Some(&b"Bearer"[..]));
    assert_eq!(
        head.header("authentication-info"),
        Some(&b"nextnonce=test"[..])
    );
    assert_eq!(head.header("content-type"), Some(&b"text/plain"[..]));
    assert_eq!(head.header("content-encoding"), Some(&b"identity"[..]));
    assert_eq!(head.header("x-remove"), None);
    assert_eq!(raw.body(2).await, b"ok");
}

#[rstest]
#[case::get("GET")]
#[case::post("POST")]
#[case::put("PUT")]
#[case::delete("DELETE")]
#[case::options("OPTIONS")]
#[case::extension("PURGE")]
#[case::webdav("PROPFIND")]
#[case::lowercase("get")]
#[case::punctuated("M-SEARCH")]
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
#[case::encoded_hash_in_query("/v1/files?purpose=batch%23retry&limit=1")]
#[case::encoded_hash_in_path("/v1/files/report%231.pdf/content")]
#[case::encoded_hash_before_query("/gemini/v1beta/files/a%23b?key=k")]
#[case::encoded_query("/v1/models?q=%E2%9C%93&empty=&flag")]
#[case::utf8_bytes_escaped("/mcp/%F0%9F%A6%80")]
#[case::lowercase_encoded_slash("/v1/files/a%2fb/content")]
#[case::dot_segments("/openai/./v1/../v1/models")]
#[case::encoded_dot_segments("/openai/%2e%2e/v1/%2E/models")]
#[case::matrix_params("/langfuse/api;v=1/public;x=y/traces")]
#[case::plus_and_encoded_space("/gemini/v1beta/models?q=a+b&r=a%20b&s=%2B")]
#[case::empty_query("/v1/models?")]
#[case::question_marks_in_query("/v1/models??a=b?c")]
#[case::colons_in_segments(
    "/v1/projects/p/locations/l/publishers/google/models/m:streamGenerateContent?alt=sse"
)]
#[case::sub_delims("/~u/@:!$&'()*+,=/x")]
#[case::raw_bytes_browsers_leave_unescaped("/a{b}|c^d\\e\"f?g={}|^`")]
#[case::invalid_percent_escapes("/a%zz%00b?%")]
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
async fn forwards_a_query_string_as_long_as_uvicorn_accepts() {
    let (upstream, mut received) = recording(ok).await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    let target = format!("/vertex_ai/v1/models?key=%2F{}&end", "a%20b".repeat(3000));
    raw.send(format!("GET {target} HTTP/1.1\r\nhost: x\r\n\r\n"))
        .await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::OK
    );
    assert_eq!(received.recv().await.unwrap().uri, target.as_str());
}

/// h11 splits the target at the first `?` and keeps a `#` as ordinary bytes of the path or
/// query, so a client that leaves `#` unescaped still reaches its route with every byte.
#[rstest]
#[case::in_query("/v1/files?purpose=batch#retry&limit=1")]
#[case::in_path("/v1/files/report#1.pdf/content")]
#[case::before_query("/gemini/v1beta/files/a#b?key=k")]
#[ignore = "bug: http::Uri drops everything from `#`, so Python gets a truncated path or query"]
#[tokio::test]
async fn forwards_an_unescaped_hash_in_the_target(#[case] target: &str) {
    let (upstream, mut received) = heads().await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send(format!("GET {target} HTTP/1.1\r\nhost: x\r\n\r\n"))
        .await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::OK
    );
    assert_eq!(within(received.recv()).await.unwrap().path, target);
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
    let mut raw = gateway.connect().await;

    raw.send("GET /v1/chat/completions HTTP/1.1\r\nhost: public.example\r\n\r\n")
        .await;

    let head = within(raw.response_head()).await.unwrap();
    assert_eq!(head.status, StatusCode::OK);
    assert_eq!(head.header("llm_provider-x-149"), Some(&b"v"[..]));
}

/// uvicorn's h11 parser has no header count limit, only a 16 KiB cap on the whole head.
#[rstest]
#[tokio::test]
async fn forwards_a_request_with_as_many_headers_as_uvicorn_accepts() {
    let (sender, mut received) = mpsc::unbounded_channel();
    let upstream = raw_upstream(move |python| {
        let sender = sender.clone();
        async move {
            let mut stream = python.into_stream();
            let mut head = Vec::new();
            while !head.ends_with(b"\r\n\r\n") {
                head.push(stream.read_u8().await.unwrap());
            }
            sender.send(head).unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n")
                .await
                .unwrap();
        }
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    let headers: String = (0..1000).map(|index| format!("x-{index}: v\r\n")).collect();
    raw.send(format!(
        "GET /v1/chat/completions HTTP/1.1\r\nhost: x\r\n{headers}\r\n"
    ))
    .await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::OK
    );
    let head = String::from_utf8(within(received.recv()).await.unwrap()).unwrap();
    assert!(head.contains(&headers));
}

async fn header_fields_python_receives(fields: &str) -> Vec<(String, Vec<u8>)> {
    let (upstream, mut received) = heads().await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send(format!(
        "GET /anthropic/v1/messages HTTP/1.1\r\nhost: x\r\n{fields}\r\n"
    ))
    .await;
    within(raw.response_head()).await.unwrap();
    within(received.recv())
        .await
        .unwrap()
        .headers
        .into_iter()
        .filter(|(name, _)| name.starts_with("x-") && name != PEER_HEADER)
        .collect()
}

fn fields(expected: &[(&str, &str)]) -> Vec<(String, Vec<u8>)> {
    expected
        .iter()
        .map(|(name, value)| ((*name).to_owned(), value.as_bytes().to_vec()))
        .collect()
}

/// Python reads `scope["headers"]` as the list the client sent, in that order.
#[rstest]
#[ignore = "divergence: HeaderMap::remove can move the headers after a dropped hop-by-hop one"]
#[tokio::test]
async fn dropping_a_hop_by_hop_header_keeps_the_others_in_order() {
    let received = header_fields_python_receives(
        "x-a: 1\r\nconnection: keep-alive\r\nx-b: 2\r\nx-c: 3\r\nx-d: 4\r\n",
    )
    .await;
    assert_eq!(
        received,
        fields(&[("x-a", "1"), ("x-b", "2"), ("x-c", "3"), ("x-d", "4")])
    );
}

/// uvicorn speaks only HTTP/1.1, so an HTTP/2 prior-knowledge client never reaches Python through
/// Rust either. hyper closes the connection where h11 answers 400, which no HTTP/2 client reads.
#[rstest]
#[tokio::test]
async fn an_http2_client_never_reaches_python() {
    let (upstream, mut received) = recording(ok).await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send(b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n\x00\x00\x00\x04\x00\x00\x00\x00\x00")
        .await;

    assert!(within(raw.until_closed()).await.is_empty());
    assert!(received.try_recv().is_err());
}

#[rstest]
#[ignore = "divergence: HeaderMap groups every value of a name at its first position"]
#[tokio::test]
async fn interleaved_repeated_headers_reach_python_interleaved() {
    let received = header_fields_python_receives("x-a: 1\r\nx-b: 2\r\nx-a: 3\r\n").await;
    assert_eq!(
        received,
        fields(&[("x-a", "1"), ("x-b", "2"), ("x-a", "3")])
    );
}

/// h11 trims optional whitespace around a value and keeps everything inside it, including empty
/// values and names using every token character.
#[rstest]
#[tokio::test]
async fn python_sees_header_values_as_h11_would_parse_them() {
    let (upstream, mut received) = recording(ok).await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("GET / HTTP/1.1\r\nhost: x\r\nx-empty:\r\nx-blank:   \r\nx-inner:  a \t b \t\r\nX-Token!#$%&'*+.^_`|~: v\r\nx-empty: second\r\n\r\n")
        .await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::OK
    );
    let request = received.recv().await.unwrap();
    let values = |name: &str| -> Vec<&[u8]> {
        request
            .headers
            .get_all(name)
            .iter()
            .map(HeaderValue::as_bytes)
            .collect()
    };
    assert_eq!(values("x-empty"), [&b""[..], b"second"]);
    assert_eq!(values("x-blank"), [&b""[..]]);
    assert_eq!(values("x-inner"), [&b"a \t b"[..]]);
    assert_eq!(values("x-token!#$%&'*+.^_`|~"), [&b"v"[..]]);
}

/// RFC 9112 section 5.2 lets a server either reject obs-fold or unfold it, and h11 unfolds.
#[rstest]
#[ignore = "divergence: hyper rejects obs-fold with a 400 where uvicorn unfolds it to one space"]
#[tokio::test]
async fn a_folded_header_reaches_python_unfolded() {
    let (upstream, mut received) = recording(ok).await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("GET / HTTP/1.1\r\nhost: x\r\nx-a: b\r\n c\r\n\r\n")
        .await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::OK
    );
    assert_eq!(received.recv().await.unwrap().headers["x-a"], "b c");
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

/// An HTTP/1.0 client cannot decode chunked framing, so a streamed answer reaches it delimited by
/// the connection closing, as uvicorn sends it.
#[rstest]
#[tokio::test]
async fn an_http10_client_gets_a_streamed_answer_unframed_and_python_gets_its_body() {
    let (upstream, mut received) = recording(|| {
        Response::new(Body::from_stream(futures_util::stream::iter(
            ["str", "eamed"].map(|piece| Ok::<_, Infallible>(Bytes::from_static(piece.as_bytes()))),
        )))
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("POST /v1/completions HTTP/1.0\r\nhost: x\r\ncontent-length: 7\r\n\r\npayload")
        .await;
    let head = within(raw.response_head()).await.unwrap();
    assert_eq!(head.status, StatusCode::OK);
    assert_eq!(head.header("transfer-encoding"), None);
    assert_eq!(within(raw.until_closed()).await, b"streamed");
    let request = received.recv().await.unwrap();
    assert_eq!(request.method, "POST");
    assert_eq!(request.body, "payload");
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
async fn informational_responses_keep_the_connection_reusable() {
    let upstream = raw_upstream(|mut python| async move {
        while let Some(head) = python.request_head().await {
            match head.path.as_str() {
                "/hints" => {
                    python
                        .send("HTTP/1.1 103 Early Hints\r\nlink: </ui/app.css>; rel=preload\r\n\r\nHTTP/1.1 200 OK\r\ncontent-length: 5\r\n\r\nfinal")
                        .await
                }
                _ => {
                    python
                        .send("HTTP/1.1 200 OK\r\ncontent-length: 4\r\n\r\nnext")
                        .await
                }
            }
        }
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("GET /hints HTTP/1.1\r\nhost: x\r\n\r\nGET /next HTTP/1.1\r\nhost: x\r\n\r\n")
        .await;
    let mut hinted = within(raw.response_head()).await.unwrap();
    while hinted.status.is_informational() {
        hinted = within(raw.response_head()).await.unwrap();
    }
    assert_eq!(hinted.status, StatusCode::OK);
    assert_eq!(raw.body(hinted.content_length().unwrap()).await, b"final");
    let next = within(raw.response_head()).await.unwrap();
    assert_eq!(raw.body(next.content_length().unwrap()).await, b"next");
}

#[rstest]
#[case::found(
    StatusCode::FOUND,
    "https://idp.example/authorize?state=a%2Fb&redirect_uri=https%3A%2F%2Fpublic.example%2Fsso%2Fcallback"
)]
#[case::see_other(StatusCode::SEE_OTHER, "/ui/?login=success")]
#[case::temporary(StatusCode::TEMPORARY_REDIRECT, "./mcp/oauth/authorize?client_id=x")]
#[tokio::test]
async fn redirects_reach_the_client_with_location_and_cookies_unchanged(
    #[case] status: StatusCode,
    #[case] location: &'static str,
) {
    let (upstream, _received) = recording(move || {
        Response::builder()
            .status(status)
            .header("location", location)
            .header(
                "set-cookie",
                "token=a%2Fb; Path=/; HttpOnly; Secure; SameSite=Lax",
            )
            .header("set-cookie", "state=; Max-Age=0")
            .header(
                "set-cookie",
                "session=x; Expires=Wed, 04 Nov 2026 10:00:00 GMT; Path=/",
            )
            .body(Body::empty())
            .unwrap()
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("GET /sso/key/generate HTTP/1.1\r\nhost: public.example\r\n\r\n")
        .await;
    let head = within(raw.response_head()).await.unwrap();
    assert_eq!(head.status, status);
    assert_eq!(head.header("location"), Some(location.as_bytes()));
    let cookies: Vec<_> = head
        .headers
        .iter()
        .filter(|(name, _)| name == "set-cookie")
        .map(|(_, value)| value.as_slice())
        .collect();
    assert_eq!(
        cookies,
        [
            &b"token=a%2Fb; Path=/; HttpOnly; Secure; SameSite=Lax"[..],
            b"state=; Max-Age=0",
            b"session=x; Expires=Wed, 04 Nov 2026 10:00:00 GMT; Path=/",
        ]
    );
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
#[tokio::test]
async fn forwards_an_asterisk_form_options_request() {
    let (upstream, mut received) = recording(ok).await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("OPTIONS * HTTP/1.1\r\nhost: x\r\n\r\n").await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::OK
    );
    let request = received.recv().await.unwrap();
    assert_eq!(request.method, "OPTIONS");
    assert_eq!(request.uri, "*");
}

#[rstest]
#[tokio::test]
async fn an_authority_form_target_is_a_400_python_never_sees() {
    let (upstream, mut received) = recording(ok).await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("CONNECT api.example:443 HTTP/1.1\r\nhost: api.example:443\r\n\r\n")
        .await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::BAD_REQUEST
    );
    assert!(received.try_recv().is_err());
}

#[rstest]
#[tokio::test]
async fn a_client_connection_close_ends_only_the_client_connection() {
    let connections = Arc::new(AtomicUsize::new(0));
    let accepted = connections.clone();
    let upstream = raw_upstream(move |mut python| {
        accepted.fetch_add(1, Ordering::SeqCst);
        async move {
            while python.request_head().await.is_some() {
                python
                    .send("HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nok")
                    .await;
            }
        }
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    for _ in 0..2 {
        let mut raw = gateway.connect().await;
        raw.send("GET / HTTP/1.1\r\nhost: x\r\nconnection: close\r\n\r\n")
            .await;
        let head = within(raw.response_head()).await.unwrap();
        assert_eq!(head.header("connection"), Some(&b"close"[..]));
        assert_eq!(raw.body(2).await, b"ok");
        within(raw.closed()).await;
    }
    assert_eq!(connections.load(Ordering::SeqCst), 1);
}

#[rstest]
#[tokio::test]
async fn a_python_connection_close_ends_only_the_upstream_connection() {
    let connections = Arc::new(AtomicUsize::new(0));
    let accepted = connections.clone();
    let upstream = raw_upstream(move |mut python| {
        accepted.fetch_add(1, Ordering::SeqCst);
        async move {
            python.request_head().await.unwrap();
            python
                .send("HTTP/1.1 500 Internal Server Error\r\ncontent-length: 5\r\nconnection: close\r\n\r\nerror")
                .await;
        }
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    for _ in 0..2 {
        raw.send("GET / HTTP/1.1\r\nhost: x\r\n\r\n").await;
        let head = within(raw.response_head()).await.unwrap();
        assert_eq!(head.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(head.header("connection"), None);
        assert_eq!(raw.body(5).await, b"error");
    }
    assert_eq!(connections.load(Ordering::SeqCst), 2);
}

/// uvicorn answers HEAD on a streaming route with chunked framing and no body, so the client
/// must not be told the body is empty, and must not wait for one before its next request.
#[rstest]
#[tokio::test]
async fn head_answered_with_chunked_framing_claims_no_length_and_keeps_the_connection() {
    let upstream = raw_upstream(|mut python| async move {
        while let Some(head) = python.request_head().await {
            match head.path.as_str() {
                "/next" => {
                    python
                        .send("HTTP/1.1 200 OK\r\ncontent-length: 4\r\n\r\nnext")
                        .await
                }
                _ => {
                    python
                        .send("HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
                        .await
                }
            }
        }
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("HEAD /mcp HTTP/1.1\r\nhost: x\r\n\r\nGET /next HTTP/1.1\r\nhost: x\r\n\r\n")
        .await;
    let head = within(raw.response_head()).await.unwrap();
    assert_eq!(head.status, StatusCode::OK);
    assert_eq!(head.content_length(), None);
    let next = within(raw.response_head()).await.unwrap();
    assert_eq!(raw.body(next.content_length().unwrap()).await, b"next");
}

const GZIP: &[u8] = &[
    0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x4b, 0x49, 0x2c, 0x49, 0xb4, 0x52,
    0x30, 0xe4, 0xe2, 0x02, 0x00, 0xce, 0xcd, 0x78, 0x54, 0x09, 0x00, 0x00, 0x00,
];

#[rstest]
#[case::fixed_length(false)]
#[case::streamed(true)]
#[tokio::test]
async fn content_codings_cross_unchanged_both_ways(#[case] streamed: bool) {
    let (upstream, mut received) = recording(move || {
        let body = match streamed {
            true => Body::from_stream(futures_util::stream::iter(
                GZIP.chunks(5)
                    .map(|piece| Ok::<_, Infallible>(Bytes::from_static(piece))),
            )),
            false => Body::from(GZIP),
        };
        Response::builder()
            .header("content-encoding", "gzip")
            .header("vary", "Accept-Encoding")
            .body(body)
            .unwrap()
    })
    .await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send(
        [
            format!(
                "POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-encoding: gzip\r\naccept-encoding: gzip, br;q=0.5\r\ncontent-length: {}\r\n\r\n",
                GZIP.len()
            )
            .as_bytes(),
            GZIP,
        ]
        .concat(),
    )
    .await;

    let head = within(raw.response_head()).await.unwrap();
    assert_eq!(head.header("content-encoding"), Some(&b"gzip"[..]));
    assert_eq!(head.header("vary"), Some(&b"Accept-Encoding"[..]));
    let body = match head.content_length() {
        Some(length) => raw.body(length).await,
        None => within(raw.chunked_body()).await,
    };
    assert_eq!(body, GZIP);
    assert_eq!(head.content_length().is_none(), streamed);
    let request = received.recv().await.unwrap();
    assert_eq!(request.headers["content-encoding"], "gzip");
    assert_eq!(request.headers["accept-encoding"], "gzip, br;q=0.5");
    assert_eq!(request.body, GZIP);
}

#[rstest]
#[tokio::test]
async fn python_sees_no_accept_encoding_the_client_did_not_send() {
    let (upstream, mut received) = recording(ok).await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("GET /metrics/ HTTP/1.1\r\nhost: x\r\n\r\n").await;
    within(raw.response_head()).await.unwrap();
    assert!(
        !received
            .recv()
            .await
            .unwrap()
            .headers
            .contains_key("accept-encoding")
    );
}
