//! Which request bodies the native `/v1/messages` answers the way the Python proxy does.
//!
//! The proxy reads its own parameters out of a request body (`user`, `tags`, `timeout`,
//! `num_retries`, `fallbacks` and more) and never sends them to the provider. The native route
//! sends every key it does not type on to the provider, which refuses it. So a body is admitted
//! only when the native route's own request type reads each of its top-level keys.
//!
//! Judging the body means reading it before the request is admitted or refused. A refused request
//! still goes to Python with the bytes the client sent.

use axum::{
    body::{Body, Bytes},
    http::{header::CONTENT_ENCODING, request::Parts},
};
use futures_util::{Stream, StreamExt, future, stream};
use http_body_util::BodyExt;
use litellm_gateway_inference::MAX_BODY_BYTES;
use litellm_inference_messages::messages_body;
use serde_json::{Map, Value};

/// A body the native route's request type reads every top-level key of. `cache_control`,
/// `safeguards` and `thread` are read too and still left to Python, since the gateway has not been
/// compared with the proxy on them.
fn is_honored(body: &[u8]) -> bool {
    let Ok(body) = serde_json::from_slice::<Map<String, Value>>(body) else {
        return false;
    };
    messages_body(body).is_ok_and(|request| {
        let params = request.params;
        params.extra.is_empty()
            && params.cache_control.is_none()
            && params.safeguards.is_none()
            && params.thread.is_none()
    })
}

/// `read` followed by `rest`, as one body.
fn replayed(
    read: Vec<Bytes>,
    rest: impl Stream<Item = Result<Bytes, axum::Error>> + Send + 'static,
) -> Body {
    Body::from_stream(stream::iter(read).map(Ok).chain(rest))
}

/// `body` to send on, carrying the bytes the client sent, and whether the native route answers
/// it as the proxy would. A body that is compressed, or larger than the native route reads, is
/// left to Python without being judged, and one that breaks is sent on broken.
pub(super) async fn judged(request: &Parts, mut body: Body) -> (Body, bool) {
    if request.headers.contains_key(CONTENT_ENCODING) {
        return (body, false);
    }
    let mut read = Vec::new();
    let mut size = 0;
    while let Some(frame) = body.frame().await {
        let data = match frame.map(|frame| frame.into_data()) {
            Ok(Ok(data)) => data,
            Ok(Err(_trailers)) => continue,
            Err(broke) => {
                return (
                    replayed(read, stream::once(future::ready(Err(broke)))),
                    false,
                );
            }
        };
        size += data.len();
        read.push(data);
        if size > MAX_BODY_BYTES {
            return (replayed(read, body.into_data_stream()), false);
        }
    }
    let whole = Bytes::from(read.concat());
    let honored = is_honored(&whole);
    (Body::from(whole), honored)
}

#[cfg(test)]
mod tests {
    use axum::http::Request;
    use rstest::rstest;

    use super::*;

    fn parts(encoding: Option<&str>) -> Parts {
        encoding
            .into_iter()
            .fold(Request::post("/v1/messages"), |request, encoding| {
                request.header(CONTENT_ENCODING, encoding)
            })
            .body(())
            .unwrap()
            .into_parts()
            .0
    }

    async fn sent_on(body: Body) -> Bytes {
        axum::body::to_bytes(body, usize::MAX).await.unwrap()
    }

    #[rstest]
    #[case::messages_params(
        r#"{"model":"claude","max_tokens":16,"stream":true,"messages":[]}"#,
        true
    )]
    #[case::a_proxy_param(r#"{"model":"claude","messages":[],"user":"end-user"}"#, false)]
    #[case::an_unknown_param(r#"{"model":"claude","messages":[],"not_a_param":1}"#, false)]
    #[case::cache_control(
        r#"{"model":"claude","messages":[],"cache_control":{"type":"ephemeral"}}"#,
        false
    )]
    #[case::safeguards(
        r#"{"model":"claude","messages":[],"safeguards":[{"type":"dangerous_tool_use"}]}"#,
        false
    )]
    #[case::thread(
        r#"{"model":"claude","messages":[],"thread":{"type":"create"}}"#,
        false
    )]
    #[case::no_model(r#"{"messages":[]}"#, false)]
    #[case::not_an_object(r#"["model"]"#, false)]
    #[case::not_json("model=claude", false)]
    #[tokio::test]
    async fn a_body_of_messages_params_only_is_honored(
        #[case] sent: &'static str,
        #[case] expected: bool,
    ) {
        let (body, honored) = judged(&parts(None), Body::from(sent)).await;

        assert_eq!(
            (honored, sent_on(body).await),
            (expected, Bytes::from(sent))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_compressed_body_is_left_to_python_unread() {
        let sent = r#"{"model":"claude","messages":[]}"#;

        let (body, honored) = judged(&parts(Some("gzip")), Body::from(sent)).await;

        assert_eq!((honored, sent_on(body).await), (false, Bytes::from(sent)));
    }

    #[rstest]
    #[tokio::test]
    async fn a_body_larger_than_the_native_route_reads_is_sent_on_whole() {
        let chunk = Bytes::from(vec![b' '; 1024 * 1024]);
        let chunks = MAX_BODY_BYTES / chunk.len() + 2;
        let sent = Body::from_stream(stream::iter(
            std::iter::repeat_n(chunk.clone(), chunks).map(Ok::<_, axum::Error>),
        ));

        let (body, honored) = judged(&parts(None), sent).await;

        assert_eq!(
            (honored, sent_on(body).await.len()),
            (false, chunks * chunk.len())
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_body_that_breaks_is_sent_on_broken() {
        let sent = Body::from_stream(stream::iter([
            Ok(Bytes::from_static(b"{\"model\":")),
            Err(std::io::Error::other("the client went away")),
        ]));

        let (body, honored) = judged(&parts(None), sent).await;

        assert!(!honored);
        assert!(axum::body::to_bytes(body, usize::MAX).await.is_err());
    }
}
