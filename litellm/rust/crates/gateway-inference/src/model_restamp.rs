//! The `model` an answer reports: the name the client asked for, in place of the provider's name
//! for the deployment that served it, as the proxy's `_override_openai_response_model` restamps
//! the answer of every route. A complete answer a fallback served keeps the provider's name, and
//! a stream's event is restamped either way.
//!
//! A route only says which model its answer has to report, by attaching a [`ReportedModel`] to
//! its response. [`restamp`] is the one layer that applies it, to a JSON body and to an event
//! stream alike, so no route rewrites its own answer.

use axum::{
    body::{Body, to_bytes},
    http::{
        HeaderMap, StatusCode,
        header::{CONTENT_LENGTH, CONTENT_TYPE},
    },
    response::{IntoResponse, Response},
};
use bytes::{Bytes, BytesMut};
use futures_util::{
    StreamExt, TryStreamExt, future,
    stream::{self, BoxStream},
};
use litellm_framer::sse::{SseCodec, SseEvent, block_end, decode_all};
use litellm_router_core::RouterResponse;
use serde_json::{Map, Value};
use tokio_util::codec::Encoder;

/// How much of a stream is held while its first event is still incomplete. A stream that sends
/// more than this without closing an event passes through as it is.
const MAX_HELD_BYTES: usize = 65536;

/// What one event of a stream says about the model the stream reports.
pub(crate) enum Stamp {
    /// Not the event that names the model, and one may still come: a keepalive.
    Later,
    /// The stream names no model from here on, or already names the right one.
    Never,
    /// The event, naming the model it was asked to.
    Now(Map<String, Value>),
}

/// Reads one event of a route's stream, given the model the stream has to report.
pub(crate) type StampEvent = fn(&Map<String, Value>, &str) -> Stamp;

/// The model the answer of one routed call reports. A route attaches it to its response as an
/// extension, and [`restamp`] applies it.
#[derive(Clone)]
pub(crate) struct ReportedModel {
    requested: String,
    fell_back: bool,
    stream_event: Option<StampEvent>,
}

impl ReportedModel {
    /// For the name the client asked for and the call that served it.
    pub(crate) fn of<R>(requested: &str, routed: &RouterResponse<R>) -> Self {
        Self {
            requested: requested.to_owned(),
            fell_back: routed.attempted_fallbacks > 0,
            stream_event: None,
        }
    }

    /// Also for the route's stream, whose model `stamp` finds.
    pub(crate) fn in_stream(self, stamp: StampEvent) -> Self {
        Self {
            stream_event: Some(stamp),
            ..self
        }
    }
}

fn is(headers: &HeaderMap, media_type: &str) -> bool {
    headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with(media_type))
}

/// The layer that makes a served answer report the model its route attached. A JSON answer has
/// its top-level `model` replaced unless a fallback served it. An event stream has the event its
/// route names rewritten, fallback or not. An answer with no [`ReportedModel`], and one that is
/// not a success, pass through.
pub(crate) async fn restamp(response: Response) -> Response {
    let (mut parts, body) = response.into_parts();
    let reported = parts
        .extensions
        .remove::<ReportedModel>()
        .filter(|_| parts.status.is_success());
    let Some(reported) = reported else {
        return Response::from_parts(parts, body);
    };
    match reported.stream_event {
        Some(stamp) if is(&parts.headers, "text/event-stream") => {
            let events = events(body.into_data_stream().boxed(), reported.requested, stamp);
            Response::from_parts(parts, Body::from_stream(events))
        }
        _ if is(&parts.headers, "application/json") && !reported.fell_back => {
            let Ok(sent) = to_bytes(body, usize::MAX).await else {
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            };
            let body = named(&sent, &reported.requested).unwrap_or(sent);
            parts.headers.remove(CONTENT_LENGTH);
            Response::from_parts(parts, Body::from(body))
        }
        _ => Response::from_parts(parts, body),
    }
}

/// `body` with its top-level `model` set to `model`, `None` for a body that is not a JSON object
/// or names no model, as `_override_openai_response_model` leaves those alone.
fn named(body: &[u8], model: &str) -> Option<Bytes> {
    let fields: Map<String, Value> = serde_json::from_slice(body).ok()?;
    if !fields.contains_key("model") {
        return None;
    }
    let named: Map<String, Value> = fields
        .into_iter()
        .map(|(name, value)| match name.as_str() {
            "model" => (name, Value::from(model)),
            _ => (name, value),
        })
        .collect();
    serde_json::to_vec(&named).ok().map(Bytes::from)
}

/// `chunks` with the event `stamp` picks naming `requested`. Chunks are raw transport reads, so
/// that event can arrive split across several or together with later ones. Bytes are held until
/// an event closes, and every chunk after the event that settles it passes through untouched.
fn events<E: Send + 'static>(
    chunks: BoxStream<'static, Result<Bytes, E>>,
    requested: String,
    stamp: StampEvent,
) -> BoxStream<'static, Result<Bytes, E>> {
    stream::unfold(
        (chunks.fuse(), Some(BytesMut::new())),
        move |(mut chunks, held)| {
            let requested = requested.clone();
            async move {
                match (chunks.next().await, held) {
                    (Some(Ok(bytes)), Some(held)) => {
                        let (held, passed) = read(held, &bytes, &requested, stamp);
                        Some((Ok(passed), (chunks, held)))
                    }
                    (Some(chunk), held) => Some((chunk, (chunks, held))),
                    (None, Some(held)) => Some((Ok(held.freeze()), (chunks, None))),
                    (None, None) => None,
                }
            }
        },
    )
    .try_filter(|bytes| future::ready(!bytes.is_empty()))
    .boxed()
}

/// `held` followed by `bytes`, as what stays held and what goes to the client now. Nothing stays
/// held once an event settled which model the stream reports.
fn read(
    mut held: BytesMut,
    bytes: &[u8],
    requested: &str,
    stamp: StampEvent,
) -> (Option<BytesMut>, Bytes) {
    held.extend_from_slice(bytes);
    let mut passed = BytesMut::new();
    while let Some(end) = block_end(&held) {
        let block = held.split_to(end);
        let Some((event, data)) = event_in(&block) else {
            passed.unsplit(block);
            continue;
        };
        let settled = match stamp(&data, requested) {
            Stamp::Later => {
                passed.unsplit(block);
                continue;
            }
            Stamp::Never => block,
            Stamp::Now(data) => written(event, data).unwrap_or(block),
        };
        passed.unsplit(settled);
        passed.unsplit(held);
        return (None, passed.freeze());
    }
    if held.len() > MAX_HELD_BYTES {
        passed.unsplit(held);
        return (None, passed.freeze());
    }
    (Some(held), passed.freeze())
}

/// The event `block` carries and its data, `None` for a block that carries no event. Data that
/// is not a JSON object reads as an empty one, which no route stamps.
fn event_in(block: &[u8]) -> Option<(SseEvent, Map<String, Value>)> {
    let event = decode_all(block).ok()?.into_iter().next()?;
    let data = serde_json::from_str(&event.data).unwrap_or_default();
    Some((event, data))
}

/// `event` written again carrying `data`.
fn written(event: SseEvent, data: Map<String, Value>) -> Option<BytesMut> {
    let mut block = BytesMut::new();
    SseCodec::default()
        .encode(
            SseEvent {
                data: Value::Object(data).to_string(),
                ..event
            },
            &mut block,
        )
        .ok()?;
    Some(block)
}

#[cfg(test)]
mod tests {
    use axum::{Extension, Json};
    use futures_util::stream;
    use litellm_router_core::Served;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    const PING: &str = "event: ping\ndata: {\"type\": \"ping\"}\n\n";
    const START: &str = "event: start\ndata: {\"type\":\"start\",\"id\":\"1\",\"model\":\"provider-model\",\"usage\":{\"input_tokens\":7}}\n\n";
    const RESTAMPED: &str = "event: start\ndata: {\"type\":\"start\",\"id\":\"1\",\"model\":\"public\",\"usage\":{\"input_tokens\":7}}\n\n";
    const DELTA: &str =
        "event: delta\ndata: {\"type\":\"delta\",\"text\":\"start provider-model\"}\n\n";
    const STOP: &str = "event: stop\ndata: {\"type\":\"stop\"}\n\n";

    /// A route whose `start` event names the model and whose `ping` is a keepalive.
    fn stamp(event: &Map<String, Value>, model: &str) -> Stamp {
        match event.get("type").and_then(Value::as_str) {
            Some("ping") => Stamp::Later,
            Some("start") => Stamp::Now(
                event
                    .iter()
                    .map(|(name, value)| match name.as_str() {
                        "model" => (name.clone(), Value::from(model)),
                        _ => (name.clone(), value.clone()),
                    })
                    .collect(),
            ),
            _ => Stamp::Never,
        }
    }

    fn reported(attempted_fallbacks: u32) -> ReportedModel {
        ReportedModel::of(
            "public",
            &RouterResponse {
                output: Served::new((), 0),
                deployment_id: "d1".to_owned(),
                name: "public".to_owned(),
                model_group: "public".to_owned(),
                attempted_retries: 0,
                max_retries: None,
                attempted_fallbacks,
            },
        )
    }

    /// What a client reads of `response` once the layer has seen it.
    async fn answered(response: Response) -> (StatusCode, String) {
        let answered = restamp(response).await;
        let status = answered.status();
        let body = to_bytes(answered.into_body(), usize::MAX).await.unwrap();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    #[rstest]
    #[case::served_by_its_own_group(
        0,
        json!({"id": "1", "model": "provider-model", "usage": {"input_tokens": 7}}),
        json!({"id": "1", "model": "public", "usage": {"input_tokens": 7}})
    )]
    #[case::served_by_a_fallback(
        1,
        json!({"id": "1", "model": "provider-model"}),
        json!({"id": "1", "model": "provider-model"})
    )]
    #[case::an_answer_that_names_no_model(0, json!({"id": "1"}), json!({"id": "1"}))]
    #[tokio::test]
    async fn a_json_answer_names_the_requested_model_unless_a_fallback_served_it(
        #[case] fallbacks: u32,
        #[case] sent: Value,
        #[case] expected: Value,
    ) {
        let response = (Extension(reported(fallbacks)), Json(sent)).into_response();

        let (status, body) = answered(response).await;

        assert_eq!(
            (status, serde_json::from_str::<Value>(&body).unwrap()),
            (StatusCode::OK, expected)
        );
    }

    #[rstest]
    #[case::an_error(
        (StatusCode::BAD_REQUEST, Extension(reported(0)), Json(json!({"model": "provider-model"})))
            .into_response()
    )]
    #[case::an_answer_no_route_reported_a_model_for(
        Json(json!({"model": "provider-model"})).into_response()
    )]
    #[tokio::test]
    async fn an_answer_that_is_not_a_served_one_passes_through(#[case] response: Response) {
        let (_, body) = answered(response).await;

        assert_eq!(body, r#"{"model":"provider-model"}"#);
    }

    #[derive(Debug, PartialEq)]
    struct Broke;

    /// Each chunk is read whole, and is written here as the events it carries.
    type Chunk = Result<Vec<&'static str>, Broke>;

    async fn streamed(chunks: Vec<Result<String, Broke>>) -> Vec<String> {
        let chunks = stream::iter(chunks)
            .map(|chunk| chunk.map(Bytes::from))
            .boxed();
        events(chunks, "public".to_owned(), stamp)
            .map(|chunk| match chunk {
                Ok(bytes) => String::from_utf8(bytes.to_vec()).unwrap(),
                Err(Broke) => "<broke>".to_owned(),
            })
            .collect()
            .await
    }

    #[rstest]
    #[case::whole_events(
        vec![Ok(vec![START]), Ok(vec![DELTA]), Ok(vec![STOP])],
        vec![vec![RESTAMPED], vec![DELTA], vec![STOP]]
    )]
    #[case::a_keepalive_first(
        vec![Ok(vec![PING]), Ok(vec![START]), Ok(vec![STOP])],
        vec![vec![PING], vec![RESTAMPED], vec![STOP]]
    )]
    #[case::events_read_together(
        vec![Ok(vec![PING, START, DELTA]), Ok(vec![STOP])],
        vec![vec![PING, RESTAMPED, DELTA], vec![STOP]]
    )]
    #[case::a_stream_that_names_no_model_passes_through(
        vec![Ok(vec![DELTA]), Ok(vec![START])],
        vec![vec![DELTA], vec![START]]
    )]
    #[case::a_break_passes_through(
        vec![Ok(vec![START]), Err(Broke), Ok(vec![STOP])],
        vec![vec![RESTAMPED], vec!["<broke>"], vec![STOP]]
    )]
    #[tokio::test]
    async fn a_stream_names_the_requested_model_in_the_event_its_route_picks(
        #[case] chunks: Vec<Chunk>,
        #[case] expected: Vec<Vec<&str>>,
    ) {
        let chunks = chunks
            .into_iter()
            .map(|chunk| chunk.map(|events| events.concat()))
            .collect();
        let expected: Vec<String> = expected.into_iter().map(|events| events.concat()).collect();

        assert_eq!(streamed(chunks).await, expected);
    }

    #[rstest]
    #[case::served_by_its_own_group(0)]
    #[case::served_by_a_fallback(1)]
    #[tokio::test]
    async fn an_event_stream_is_restamped_whether_or_not_a_fallback_served_it(
        #[case] fallbacks: u32,
    ) {
        let response = (
            [(CONTENT_TYPE, "text/event-stream")],
            Extension(reported(fallbacks).in_stream(stamp)),
            [START, STOP].concat(),
        )
            .into_response();

        assert_eq!(
            answered(response).await,
            (StatusCode::OK, [RESTAMPED, STOP].concat())
        );
    }

    #[rstest]
    #[case::inside_the_event_name(10)]
    #[case::inside_the_model(60)]
    #[case::before_the_closing_blank_line(START.len() - 1)]
    #[tokio::test]
    async fn an_event_split_across_chunks_is_held_until_it_closes(#[case] at: usize) {
        let (head, tail) = START.split_at(at);

        assert_eq!(
            streamed(vec![
                Ok(head.to_owned()),
                Ok(tail.to_owned()),
                Ok(STOP.to_owned())
            ])
            .await,
            [RESTAMPED, STOP]
        );
    }

    #[tokio::test]
    async fn a_stream_that_ends_inside_its_first_event_loses_no_bytes() {
        let (head, _) = START.split_at(40);

        assert_eq!(
            streamed(vec![Ok(PING.to_owned()), Ok(head.to_owned())]).await,
            [PING, head]
        );
    }

    #[tokio::test]
    async fn a_first_event_larger_than_the_hold_passes_through() {
        let oversized = format!("event: start\ndata: {}", "x".repeat(MAX_HELD_BYTES));

        assert_eq!(
            streamed(vec![Ok(oversized.clone()), Ok(START.to_owned())]).await,
            [oversized.as_str(), START]
        );
    }
}
