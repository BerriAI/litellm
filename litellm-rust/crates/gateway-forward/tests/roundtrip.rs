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
use support::{Gateway, client, recording};

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

fn header_value() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(prop_oneof![0x21u8..=0x7e, 0x80u8..=0xff], 1..24)
}

fn message() -> impl Strategy<Value = Message> {
    (
        prop::collection::btree_map(
            "x-[a-z0-9-]{1,12}",
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
    let segment = prop_oneof!["[A-Za-z0-9._~!$&'()*+,;=:@-]{0,8}", "%[0-9A-F]{2}"];
    (
        prop::collection::vec(segment, 0..5),
        prop::option::of("[A-Za-z0-9=&._~-]{0,20}"),
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
    let method = prop_oneof![
        Just(Method::GET),
        Just(Method::POST),
        Just(Method::PUT),
        Just(Method::PATCH),
        Just(Method::DELETE),
        Just(Method::OPTIONS),
        Just(Method::from_bytes(b"PURGE").unwrap()),
    ];
    let status = (200u16..600)
        .prop_filter("bodiless status", |code| ![204, 205, 304].contains(code))
        .prop_map(|code| StatusCode::from_u16(code).unwrap());
    (method, target(), message(), status, message()).prop_map(
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
