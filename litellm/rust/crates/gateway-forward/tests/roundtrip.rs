mod support;

use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
};

use axum::{
    body::{Body, Bytes},
    http::{HeaderMap, HeaderName, HeaderValue, Method, Request, StatusCode},
    response::Response,
};
use http_body_util::BodyExt;
use litellm_gateway_forward::PEER_HEADER;
use proptest::{prelude::*, test_runner::TestRunner};
use support::{Gateway, client, recording, within};

#[derive(Clone, Debug)]
struct Message {
    headers: Vec<(String, Vec<Vec<u8>>)>,
    body: Vec<u8>,
    chunked: bool,
}

#[derive(Clone, Debug)]
struct Exchange {
    method: Method,
    target: String,
    request: Message,
    status: StatusCode,
    response: Message,
}

/// Any value h11 reads back unchanged: visible ASCII, obs-text and interior whitespace.
fn header_value() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(
        prop_oneof![8 => 0x21u8..=0x7e, 2 => 0x80u8..=0xff, 1 => Just(b' '), 1 => Just(b'\t')],
        0..24,
    )
    .prop_map(|value| value.trim_ascii().to_vec())
}

fn method() -> impl Strategy<Value = Method> {
    prop_oneof![
        Just(Method::GET),
        Just(Method::POST),
        Just(Method::PUT),
        Just(Method::PATCH),
        Just(Method::DELETE),
        Just(Method::OPTIONS),
        "[A-Za-z0-9!#$%&'*+.^_`|~-]{1,16}"
            .prop_filter("HEAD and CONNECT frame differently", |method| {
                !["HEAD", "CONNECT"].contains(&method.as_str())
            })
            .prop_map(|method| Method::from_bytes(method.as_bytes()).unwrap()),
    ]
}

fn message() -> impl Strategy<Value = Message> {
    (
        prop::collection::btree_map(
            "x-[a-z0-9!#$%&'*+.^_`|~-]{1,12}",
            prop::collection::vec(header_value(), 1..4),
            0..12,
        ),
        prop::collection::vec(any::<u8>(), 0..32 * 1024),
        any::<bool>(),
    )
        .prop_map(|(headers, body, chunked)| Message {
            headers: headers.into_iter().collect(),
            body,
            chunked,
        })
}

fn target() -> impl Strategy<Value = String> {
    let segment = prop_oneof![
        "[A-Za-z0-9._~!$&'()*+,;=:@-]{0,8}",
        "%[0-9A-Fa-f]{2}",
        Just(".".to_owned()),
        Just("..".to_owned()),
        Just(String::new()),
    ];
    (
        prop::collection::vec(segment, 0..6),
        prop::option::of("[A-Za-z0-9=&._~%/?:@!$'()*+,;-]{0,40}"),
    )
        .prop_map(|(segments, query)| {
            let path = format!("/{}", segments.join("/"));
            match query {
                Some(query) => format!("{path}?{query}"),
                None => path,
            }
        })
}

fn exchange() -> impl Strategy<Value = Exchange> {
    let status = (200u16..600)
        .prop_filter("bodiless status", |code| ![204, 205, 304].contains(code))
        .prop_map(|code| StatusCode::from_u16(code).unwrap());
    (method(), target(), message(), status, message()).prop_map(
        |(method, target, request, status, response)| Exchange {
            method,
            target,
            request,
            status,
            response,
        },
    )
}

fn body(message: &Message) -> Body {
    if !message.chunked {
        return Body::from(message.body.clone());
    }
    let chunks: Vec<Result<Bytes, Infallible>> = message
        .body
        .chunks(1000)
        .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
        .collect();
    Body::from_stream(futures_util::stream::iter(chunks))
}

fn apply(headers: &mut HeaderMap, message: &Message) {
    if message.chunked {
        headers.insert("transfer-encoding", HeaderValue::from_static("chunked"));
    }
    for (name, values) in &message.headers {
        for value in values {
            headers.append(
                HeaderName::from_bytes(name.as_bytes()).unwrap(),
                HeaderValue::from_bytes(value).unwrap(),
            );
        }
    }
}

fn assert_carried(headers: &HeaderMap, message: &Message, framing_and_hop: &[&str]) {
    for (name, values) in &message.headers {
        let carried: Vec<&[u8]> = headers
            .get_all(name)
            .iter()
            .map(HeaderValue::as_bytes)
            .collect();
        let sent: Vec<&[u8]> = values.iter().map(Vec::as_slice).collect();
        assert_eq!(carried, sent, "header {name}");
    }
    for name in headers.keys() {
        assert!(
            framing_and_hop.contains(&name.as_str())
                || message
                    .headers
                    .iter()
                    .any(|(sent, _)| sent == name.as_str()),
            "unexpected header {name}"
        );
    }
}

/// Any request a client can send and any response Python can give cross the forwarder unchanged,
/// apart from the framing and peer headers the forwarder owns.
#[test]
fn every_exchange_crosses_the_forwarder_unchanged() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let current = Arc::new(Mutex::new(None::<Exchange>));
    let responding = current.clone();
    let (gateway, received) = runtime.block_on(async {
        let (upstream, received) = recording(move || {
            let exchange = responding.lock().unwrap().clone().unwrap();
            let mut response = Response::new(body(&exchange.response));
            *response.status_mut() = exchange.status;
            apply(response.headers_mut(), &exchange.response);
            response
        })
        .await;
        (
            Gateway::start(upstream).await,
            tokio::sync::Mutex::new(received),
        )
    });
    let client = client();
    TestRunner::new(ProptestConfig::with_cases(256))
        .run(&exchange(), |exchange| {
            *current.lock().unwrap() = Some(exchange.clone());
            runtime.block_on(async {
                while received.lock().await.try_recv().is_ok() {}
                let mut request = Request::builder()
                    .method(exchange.method.clone())
                    .uri(gateway.url(&exchange.target))
                    .body(body(&exchange.request))
                    .unwrap();
                apply(request.headers_mut(), &exchange.request);
                let response = support::within(client.request(request)).await.unwrap();

                let python = received.lock().await.recv().await.unwrap();
                assert_eq!(python.method, exchange.method);
                assert_eq!(python.uri, exchange.target.as_str());
                assert_eq!(python.body, exchange.request.body);
                assert_carried(
                    &python.headers,
                    &exchange.request,
                    &["host", "content-length", "transfer-encoding", PEER_HEADER],
                );
                assert_eq!(python.headers.get_all(PEER_HEADER).iter().count(), 1);

                assert_eq!(response.status(), exchange.status);
                let headers = response.headers().clone();
                assert_carried(
                    &headers,
                    &exchange.response,
                    &["date", "content-length", "transfer-encoding"],
                );
                let body = support::within(response.into_body().collect())
                    .await
                    .unwrap();
                assert_eq!(body.to_bytes(), exchange.response.body);
            });
            Ok(())
        })
        .unwrap();
}

#[derive(Clone, Debug)]
enum Framing {
    Unframed,
    Length,
    Chunked(usize),
}

#[derive(Clone, Debug)]
struct WireRequest {
    method: Method,
    target: String,
    headers: Vec<(String, Vec<u8>)>,
    body: Vec<u8>,
    framing: Framing,
}

fn wire_request() -> impl Strategy<Value = WireRequest> {
    let framing = prop_oneof![
        Just(Framing::Unframed),
        Just(Framing::Length),
        (1usize..2048).prop_map(Framing::Chunked),
    ];
    (
        method(),
        target(),
        prop::collection::vec(
            ("[xX]-[A-Za-z0-9!#$%&'*+.^_`|~-]{1,12}", header_value()),
            0..24,
        ),
        prop::collection::vec(any::<u8>(), 0..8 * 1024),
        framing,
    )
        .prop_map(|(method, target, headers, body, framing)| WireRequest {
            method,
            target,
            headers,
            body: match framing {
                Framing::Unframed => Vec::new(),
                _ => body,
            },
            framing,
        })
}

fn encode(request: &WireRequest) -> Vec<u8> {
    let framing = match request.framing {
        Framing::Unframed => String::new(),
        Framing::Length => format!("Content-Length: {}\r\n", request.body.len()),
        Framing::Chunked(_) => "Transfer-Encoding: chunked\r\n".to_owned(),
    };
    let fields = request
        .headers
        .iter()
        .flat_map(|(name, value)| [name.as_bytes(), b": ", value, b"\r\n"].concat());
    let body = match request.framing {
        Framing::Chunked(size) => request
            .body
            .chunks(size)
            .flat_map(|chunk| {
                [format!("{:x}\r\n", chunk.len()).as_bytes(), chunk, b"\r\n"].concat()
            })
            .chain(*b"0\r\n\r\n")
            .collect(),
        _ => request.body.clone(),
    };
    [
        format!(
            "{} {} HTTP/1.1\r\nHost: x\r\n",
            request.method, request.target
        )
        .into_bytes(),
        fields.collect(),
        framing.into_bytes(),
        b"\r\n".to_vec(),
        body,
    ]
    .concat()
}

fn by_name<'a>(fields: impl Iterator<Item = (&'a str, &'a [u8])>) -> Vec<(String, &'a [u8])> {
    let mut fields: Vec<_> = fields
        .map(|(name, value)| (name.to_ascii_lowercase(), value))
        .collect();
    fields.sort_by(|(left, _), (right, _)| left.cmp(right));
    fields
}

/// Pipelined requests in any framing, written in arbitrary pieces, reach Python one by one with
/// the bytes h11 would have read from the client directly.
#[test]
fn pipelined_wire_requests_reach_python_as_sent() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let (gateway, received) = runtime.block_on(async {
        let (upstream, received) = recording(|| Response::new(Body::from("ok"))).await;
        (
            Gateway::start(upstream).await,
            tokio::sync::Mutex::new(received),
        )
    });
    let requests = (prop::collection::vec(wire_request(), 1..5), 1usize..4096);
    TestRunner::new(ProptestConfig::with_cases(256))
        .run(&requests, |(requests, piece)| {
            runtime.block_on(async {
                let mut received = received.lock().await;
                while received.try_recv().is_ok() {}
                let mut raw = gateway.connect().await;
                let wire: Vec<u8> = requests.iter().flat_map(encode).collect();
                for bytes in wire.chunks(piece) {
                    raw.send(bytes).await;
                }
                for request in &requests {
                    let head = within(raw.response_head()).await.unwrap();
                    assert_eq!(head.status, StatusCode::OK, "{request:?}");
                    assert_eq!(raw.body(2).await, b"ok");
                    let python = within(received.recv()).await.unwrap();
                    assert_eq!(python.method, request.method);
                    assert_eq!(python.uri, request.target.as_str());
                    assert_eq!(python.body, request.body);
                    assert_eq!(python.headers["host"], "x");
                    let forwarded = python.headers.iter().filter(|(name, _)| {
                        !["host", "content-length", "transfer-encoding", PEER_HEADER]
                            .contains(&name.as_str())
                    });
                    assert_eq!(
                        by_name(forwarded.map(|(name, value)| (name.as_str(), value.as_bytes()))),
                        by_name(
                            request
                                .headers
                                .iter()
                                .map(|(name, value)| (name.as_str(), value.as_slice()))
                        )
                    );
                }
            });
            Ok(())
        })
        .unwrap();
}
