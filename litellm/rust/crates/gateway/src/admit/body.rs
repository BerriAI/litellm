//! Reading a request's body before it is admitted or refused, without changing what is sent on.

use axum::{
    body::{Body, Bytes},
    http::{header::CONTENT_LENGTH, request::Parts},
};
use futures_util::{Stream, StreamExt, future, stream};
use http_body_util::BodyExt;

use litellm_gateway_inference::MAX_BODY_BYTES;

/// `read` followed by `rest`, as one body.
fn replayed(
    read: Vec<Bytes>,
    rest: impl Stream<Item = Result<Bytes, axum::Error>> + Send + 'static,
) -> Body {
    Body::from_stream(stream::iter(read).map(Ok).chain(rest))
}

fn declares_more_than_the_route_reads(request: &Parts) -> bool {
    request
        .headers
        .get(CONTENT_LENGTH)
        .and_then(|length| length.to_str().ok()?.parse::<usize>().ok())
        .is_some_and(|length| length > MAX_BODY_BYTES)
}

/// `body` to send on, carrying the bytes the client sent, and those bytes when the native route
/// would read all of them. A body that says it is larger than the native route reads is sent on
/// untouched, one that turns out to be is sent on from where the reading stopped, and one that
/// breaks is sent on broken.
pub(super) async fn read(request: &Parts, mut body: Body) -> (Body, Option<Bytes>) {
    if declares_more_than_the_route_reads(request) {
        return (body, None);
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
                    None,
                );
            }
        };
        size += data.len();
        read.push(data);
        if size > MAX_BODY_BYTES {
            return (replayed(read, body.into_data_stream()), None);
        }
    }
    let whole = Bytes::from(read.concat());
    (Body::from(whole.clone()), Some(whole))
}

#[cfg(test)]
mod tests {
    use axum::http::Request;
    use rstest::rstest;

    use super::*;

    fn parts(length: Option<usize>) -> Parts {
        length
            .into_iter()
            .fold(Request::post("/v1/messages"), |request, length| {
                request.header(CONTENT_LENGTH, length)
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
    #[tokio::test]
    async fn a_body_is_read_whole_and_sent_on_as_the_client_sent_it() {
        let sent = Body::from_stream(stream::iter([
            Ok::<_, axum::Error>(Bytes::from_static(b"{\"model\":")),
            Ok(Bytes::from_static(b"\"claude\"}")),
        ]));

        let (body, whole) = read(&parts(Some(18)), sent).await;

        assert_eq!(
            (whole, sent_on(body).await),
            (
                Some(Bytes::from_static(b"{\"model\":\"claude\"}")),
                Bytes::from_static(b"{\"model\":\"claude\"}")
            )
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_body_larger_than_the_native_route_reads_is_sent_on_whole() {
        let chunk = Bytes::from(vec![b' '; 1024 * 1024]);
        let chunks = MAX_BODY_BYTES / chunk.len() + 2;
        let sent = Body::from_stream(stream::iter(
            std::iter::repeat_n(chunk.clone(), chunks).map(Ok::<_, axum::Error>),
        ));

        let (body, whole) = read(&parts(None), sent).await;

        assert_eq!(
            (whole, sent_on(body).await.len()),
            (None, chunks * chunk.len())
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_body_that_says_it_is_larger_than_the_native_route_reads_is_not_read() {
        let (body, whole) = read(&parts(Some(MAX_BODY_BYTES + 1)), Body::from("{}")).await;

        assert_eq!((whole, sent_on(body).await), (None, Bytes::from("{}")));
    }

    #[rstest]
    #[tokio::test]
    async fn a_body_that_breaks_is_sent_on_broken() {
        let sent = Body::from_stream(stream::iter([
            Ok(Bytes::from_static(b"{\"model\":")),
            Err(std::io::Error::other("the client went away")),
        ]));

        let (body, whole) = read(&parts(None), sent).await;

        assert_eq!(whole, None);
        assert!(axum::body::to_bytes(body, usize::MAX).await.is_err());
    }
}
