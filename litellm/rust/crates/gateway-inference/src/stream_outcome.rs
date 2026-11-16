//! How a served stream went, told to the router as soon as it is known. A streamed attempt is
//! served deferred: the router counted its request and hears the rest here, a success with the
//! tokens the stream counted once it ends, or a failure the moment the stream breaks or carries
//! the provider's error. A stream the client stops reading reports neither.

use std::{future::Future, str::FromStr};

use bytes::{Bytes, BytesMut};
use futures_util::{
    StreamExt,
    stream::{self, BoxStream},
};
use litellm_framer::sse::SseCodec;
use litellm_inference::Error as InferenceError;
use litellm_llms_types::formats::responses::streaming::ResponsesEventType;
use litellm_router_core::{AttemptError, Deployment, FailureKind, FailureReport, SuccessReport};
use serde_json::{Map, Value};
use tokio_util::codec::Decoder;

use crate::{Gateway, routing};

/// The status litellm gives a provider's error event, which arrives after a 200.
const ERROR_EVENT_STATUS: u16 = 500;

/// What a stream's own events said so far.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Told {
    /// The usage the events carried, a later event's counts over an earlier one's.
    usage: Map<String, Value>,
    /// The provider's error event, as it was sent.
    error: Option<String>,
}

impl Told {
    fn with_usage(self, usage: Option<&Value>) -> Self {
        let Some(Value::Object(counts)) = usage else {
            return self;
        };
        Self {
            usage: self
                .usage
                .into_iter()
                .chain(counts.clone())
                .filter(|(_, count)| !count.is_null())
                .collect(),
            ..self
        }
    }

    fn with_error(self, event: &Map<String, Value>) -> Self {
        Self {
            error: Some(Value::Object(event.clone()).to_string()),
            ..self
        }
    }
}

/// Reads one event of a protocol's stream into what the stream said so far.
pub(crate) type Read = fn(Told, &Map<String, Value>) -> Told;

fn event_type(event: &Map<String, Value>) -> &str {
    event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
}

/// An Anthropic Messages event: `message_start` counts the input, `message_delta` the output.
pub(crate) fn messages_event(told: Told, event: &Map<String, Value>) -> Told {
    match event_type(event) {
        "message_start" => told.with_usage(event.get("message").and_then(|m| m.get("usage"))),
        "message_delta" | "message_stop" => told.with_usage(event.get("usage")),
        "error" => told.with_error(event),
        _ => told,
    }
}

/// An OpenAI Responses event: the closing event carries the response and its usage.
pub(crate) fn responses_event(told: Told, event: &Map<String, Value>) -> Told {
    match ResponsesEventType::from_str(event_type(event)) {
        Ok(ResponsesEventType::ResponseCompleted | ResponsesEventType::ResponseIncomplete) => {
            told.with_usage(event.get("response").and_then(|r| r.get("usage")))
        }
        Ok(ResponsesEventType::ResponseFailed | ResponsesEventType::Error) => {
            told.with_error(event)
        }
        _ => told,
    }
}

/// How a stream went: the tokens it counted, or the failure that ended it.
pub(crate) type Outcome = Result<u64, AttemptError>;

/// A stream that broke under the client, as router-core reads it.
pub(crate) fn broke(error: &litellm_host_http::Error<InferenceError>) -> AttemptError {
    let lost_connection = || AttemptError::new(FailureKind::ApiConnection, None);
    match error {
        litellm_host_http::Error::Call(route) => {
            routing::route_failure(route).unwrap_or_else(lost_connection)
        }
        litellm_host_http::Error::Protocol => lost_connection(),
    }
}

struct Watched<E> {
    chunks: BoxStream<'static, Result<Bytes, E>>,
    /// The reading of a stream whose outcome is not known yet.
    reading: Option<Reading>,
}

struct Reading {
    codec: SseCodec,
    unread: BytesMut,
    told: Told,
}

impl Reading {
    fn read(self, bytes: &[u8], read: Read) -> Self {
        let mut codec = self.codec;
        let mut unread = self.unread;
        unread.extend_from_slice(bytes);
        let told = std::iter::from_fn(|| codec.decode(&mut unread).ok().flatten())
            .filter_map(|event| serde_json::from_str::<Map<String, Value>>(&event.data).ok())
            .fold(self.told, |told, event| read(told, &event));
        Self {
            codec,
            unread,
            told,
        }
    }
}

/// `chunks` as they are, with `report` told how the stream went: once, and never for a stream
/// that is dropped before its outcome is known.
pub(crate) fn watched<E, R, F>(
    chunks: BoxStream<'static, Result<Bytes, E>>,
    read: Read,
    broke: fn(&E) -> AttemptError,
    report: R,
) -> BoxStream<'static, Result<Bytes, E>>
where
    E: Send + 'static,
    R: Fn(Outcome) -> F + Send + Sync + 'static,
    F: Future<Output = ()> + Send,
{
    let watched = Watched {
        chunks,
        reading: Some(Reading {
            codec: SseCodec::default(),
            unread: BytesMut::new(),
            told: Told::default(),
        }),
    };
    stream::unfold((watched, report), move |(watched, report)| async move {
        let Watched {
            mut chunks,
            reading,
        } = watched;
        let chunk = chunks.next().await;
        let (reading, outcome) = match (&chunk, reading) {
            (_, None) => (None, None),
            (Some(Ok(bytes)), Some(reading)) => {
                let reading = reading.read(bytes, read);
                match &reading.told.error {
                    Some(sent) => (
                        None,
                        Some(Err(AttemptError::from_provider(
                            ERROR_EVENT_STATUS,
                            sent,
                            None,
                        ))),
                    ),
                    None => (Some(reading), None),
                }
            }
            (Some(Err(error)), Some(_)) => (None, Some(Err(broke(error)))),
            (None, Some(reading)) => {
                let usage = Value::Object(reading.told.usage);
                (None, Some(Ok(routing::total_tokens(Some(&usage)))))
            }
        };
        if let Some(outcome) = outcome {
            report(outcome).await;
        }
        Some((chunk?, (Watched { chunks, reading }, report)))
    })
    .boxed()
}

/// `chunks` of a stream `deployment` serves, its outcome reported to the router.
pub(crate) fn reporting_outcome<H: routing::Host>(
    gateway: &Gateway<H>,
    deployment: &Deployment,
    chunks: BoxStream<'static, Result<Bytes, litellm_host_http::Error<InferenceError>>>,
    read: Read,
) -> BoxStream<'static, Result<Bytes, litellm_host_http::Error<InferenceError>>> {
    let router = gateway.router.clone();
    let routing = gateway.routing.clone();
    let deployment_id = deployment.id.clone();
    let cooldown_time = deployment.model_info.cooldown_time.or_else(|| {
        deployment
            .raw_params
            .get("cooldown_time")
            .and_then(Value::as_f64)
    });
    watched(chunks, read, broke, move |outcome: Outcome| {
        let (router, routing, deployment_id) =
            (router.clone(), routing.clone(), deployment_id.clone());
        async move {
            let recorded = match outcome {
                Ok(total_tokens) => router
                    .report_success(
                        &*routing,
                        &deployment_id,
                        total_tokens as f64,
                        0.0,
                        SuccessReport::default(),
                    )
                    .await
                    .map(|_| ()),
                Err(failure) => router
                    .report_failure(
                        &*routing,
                        &deployment_id,
                        &failure,
                        FailureReport {
                            deployment_cooldown: true,
                            cooldown_time,
                            ..FailureReport::default()
                        },
                    )
                    .await
                    .map(|_| ()),
            };
            if let Err(error) = recorded {
                tracing::warn!(%deployment_id, %error, "stream outcome was not recorded");
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures_util::stream;
    use parking_lot::Mutex;
    use rstest::rstest;

    use super::*;

    const START: &str = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":7,\"output_tokens\":1}}}\n\n";
    const DELTA: &str = "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{},\"usage\":{\"output_tokens\":5}}\n\n";
    const STOP: &str = "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    const OVERLOADED: &str = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\n\n";
    const COMPLETED: &str = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":4,\"total_tokens\":9}}}\n\n";
    const FAILED: &str = "event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\"}}\n\n";

    #[derive(Debug, PartialEq)]
    struct Dropped;

    fn dropped(_: &Dropped) -> AttemptError {
        AttemptError::new(FailureKind::ApiConnection, None)
    }

    /// What `report` heard, as the tokens of a success or the status and kind of a failure.
    type Heard = Result<u64, (Option<u16>, FailureKind)>;

    async fn heard(
        chunks: Vec<Result<&'static str, Dropped>>,
        read: Read,
        take: usize,
    ) -> (usize, Vec<Heard>) {
        let reports = Arc::new(Mutex::new(Vec::new()));
        let sink = reports.clone();
        let chunks = stream::iter(chunks)
            .map(|chunk| chunk.map(|text| Bytes::from_static(text.as_bytes())))
            .boxed();
        let delivered = watched(chunks, read, dropped, move |outcome: Outcome| {
            sink.lock()
                .push(outcome.map_err(|failure| (failure.status, failure.kind)));
            async {}
        })
        .take(take)
        .count()
        .await;
        let reports = reports.lock().clone();
        (delivered, reports)
    }

    #[rstest]
    #[case::messages_counts_input_and_final_output(
        vec![Ok(START), Ok(DELTA), Ok(STOP)],
        messages_event as Read,
        vec![Ok(12)]
    )]
    #[case::responses_counts_the_closing_total(
        vec![Ok(COMPLETED)],
        responses_event as Read,
        vec![Ok(9)]
    )]
    #[case::a_stream_without_usage_counts_nothing(vec![Ok(STOP)], messages_event as Read, vec![Ok(0)])]
    #[case::messages_error_event_fails_once(
        vec![Ok(START), Ok(OVERLOADED), Ok(STOP)],
        messages_event as Read,
        vec![Err((Some(500), FailureKind::InternalServer))]
    )]
    #[case::responses_failed_event_fails(
        vec![Ok(FAILED)],
        responses_event as Read,
        vec![Err((Some(500), FailureKind::InternalServer))]
    )]
    #[case::a_break_fails_once_and_the_rest_passes_through(
        vec![Ok(START), Err(Dropped), Ok(STOP)],
        messages_event as Read,
        vec![Err((None, FailureKind::ApiConnection))]
    )]
    #[tokio::test]
    async fn a_stream_read_to_its_end_reports_how_it_went_once(
        #[case] chunks: Vec<Result<&'static str, Dropped>>,
        #[case] read: Read,
        #[case] expected: Vec<Heard>,
    ) {
        let sent = chunks.len();

        assert_eq!(heard(chunks, read, usize::MAX).await, (sent, expected));
    }

    #[tokio::test]
    async fn an_event_split_across_chunks_is_read_whole() {
        let (head, tail) = DELTA.split_at(40);

        assert_eq!(
            heard(
                vec![Ok(START), Ok(head), Ok(tail)],
                messages_event,
                usize::MAX
            )
            .await,
            (3, vec![Ok(12)])
        );
    }

    #[tokio::test]
    async fn a_stream_the_client_stops_reading_reports_nothing() {
        assert_eq!(
            heard(vec![Ok(START), Ok(DELTA), Ok(STOP)], messages_event, 2).await,
            (2, vec![])
        );
    }
}
