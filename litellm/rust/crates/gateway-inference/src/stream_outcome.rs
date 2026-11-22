//! How a served stream went, told to the router as soon as it is known. A streamed attempt is
//! served deferred: the router counted its request and hears the rest here, a success with the
//! tokens the stream counted once it ends, or a failure the moment the stream breaks or carries
//! the provider's error. A stream the client stops reading reports neither.
//!
//! A provider's error event is the failure the same error is before a stream opens, with the
//! same status, so the cooldown rules read both alike. The Python proxy differs: it records a
//! stream that goes out with an error event as served, unless another attempt took it over.

use std::{future::Future, sync::Arc};

use bytes::Bytes;
use futures_util::{
    StreamExt, TryStreamExt,
    stream::{self, BoxStream},
};
use litellm_framer::sse::SseBlock;
use litellm_host::{call::CallOutput, protocol::Protocol};
use litellm_inference::Error as InferenceError;
use litellm_router_core::{
    AttemptError, Deployment, FailureKind, FailureReport, Router, RouterCache, SuccessReport,
};
use serde_json::Value;

use crate::{
    Gateway, routing,
    stream_event::{Counts, Kind, Read, Reading},
};

/// What a hosted call ran to, a stream of it as the blocks it was read in.
pub(crate) type Streamed<P> = CallOutput<
    <P as Protocol>::Response,
    <P as Protocol>::StreamHead,
    SseBlock,
    litellm_host_http::Error<<P as Protocol>::Error>,
>;

/// `streamed` as [`litellm_host_http::render`] takes it, a stream of it as the bytes of its
/// blocks.
pub(crate) fn bytes<R, H, E: 'static>(
    streamed: CallOutput<R, H, SseBlock, E>,
) -> CallOutput<R, H, Bytes, E> {
    match streamed {
        CallOutput::Complete(response) => CallOutput::Complete(response),
        CallOutput::Stream { head, chunks } => CallOutput::Stream {
            head,
            chunks: chunks.map_ok(|block| block.bytes).boxed(),
        },
    }
}

/// How a stream went: what it counted, or the failure that ended it.
pub(crate) type Outcome = Result<Counts, AttemptError>;

fn lost_connection() -> AttemptError {
    AttemptError::new(FailureKind::ApiConnection, None)
}

/// A stream that broke under the client, as router-core reads it.
pub(crate) fn broke(error: &litellm_host_http::Error<InferenceError>) -> AttemptError {
    match error {
        litellm_host_http::Error::Call(route) => {
            routing::route_failure(route, lost_connection).unwrap_or_else(lost_connection)
        }
        litellm_host_http::Error::Protocol => lost_connection(),
    }
}

struct Watched<E> {
    blocks: BoxStream<'static, Result<SseBlock, E>>,
    /// What a stream whose outcome is not known yet counted so far, a later event's count over
    /// an earlier one's.
    counted: Option<Counts>,
}

/// `blocks` as they are, with `report` told how the stream went: once, and never for a stream
/// that is dropped before its outcome is known.
pub(crate) fn watched<E, R, F>(
    blocks: BoxStream<'static, Result<SseBlock, E>>,
    read: Read,
    broke: fn(&E) -> AttemptError,
    report: R,
) -> BoxStream<'static, Result<SseBlock, E>>
where
    E: Send + 'static,
    R: Fn(Outcome) -> F + Send + Sync + 'static,
    F: Future<Output = ()> + Send,
{
    let watched = Watched {
        blocks,
        counted: Some(Counts::default()),
    };
    stream::unfold((watched, report), move |(watched, report)| async move {
        let Watched {
            mut blocks,
            counted,
        } = watched;
        let block = blocks.next().await;
        let (counted, outcome) = match (&block, counted) {
            (_, None) => (None, None),
            (Some(Ok(block)), Some(counted)) => {
                match block.event.as_ref().map(|event| read(&event.data)) {
                    Some(Reading {
                        kind: Kind::Failed { status, answer },
                        ..
                    }) => (
                        None,
                        Some(Err(AttemptError::from_provider(status, &answer, None))),
                    ),
                    Some(Reading {
                        usage: Some(usage), ..
                    }) => (Some(usage.over(counted)), None),
                    Some(Reading { usage: None, .. }) | None => (Some(counted), None),
                }
            }
            (Some(Err(error)), Some(_)) => (None, Some(Err(broke(error)))),
            (None, Some(counted)) => (None, Some(Ok(counted))),
        };
        if let Some(outcome) = outcome {
            report(outcome).await;
        }
        Some((block?, (Watched { blocks, counted }, report)))
    })
    .boxed()
}

/// A stream's failure after its attempt returned it, recorded against `deployment` as Python's
/// `report_stream_failure` records it: counted, and cooled down by the rules for the
/// deployment's own `cooldown_time`.
async fn record_failure<C: RouterCache>(
    router: &Router,
    cache: &C,
    deployment: &Deployment,
    failure: &AttemptError,
) {
    let cooldown_time = deployment.model_info.cooldown_time.or_else(|| {
        deployment
            .raw_params
            .get("cooldown_time")
            .and_then(Value::as_f64)
    });
    let report = FailureReport {
        deployment_cooldown: true,
        cooldown_time,
        ..FailureReport::default()
    };
    if let Err(error) = router
        .report_failure(cache, &deployment.id, failure, report)
        .await
    {
        tracing::warn!(deployment_id = %deployment.id, %error, "stream failure was not recorded");
    }
}

/// `blocks` of a stream `deployment` serves, the stream's outcome reported to the router.
pub(crate) fn reporting_outcome<H: routing::Host>(
    gateway: &Gateway<H>,
    deployment: &Arc<Deployment>,
    blocks: routing::RouteChunks,
    read: Read,
) -> routing::RouteChunks {
    let router = gateway.router.clone();
    let routing = gateway.routing.clone();
    let deployment = deployment.clone();
    watched(blocks, read, broke, move |outcome: Outcome| {
        let (router, routing, deployment) = (router.clone(), routing.clone(), deployment.clone());
        async move {
            let total_tokens = match outcome {
                Ok(counted) => counted.total(),
                Err(failure) => {
                    return record_failure(&router, &*routing, &deployment, &failure).await;
                }
            };
            let recorded = router
                .report_success(
                    &*routing,
                    &deployment.id,
                    total_tokens as f64,
                    0.0,
                    SuccessReport::default(),
                )
                .await;
            if let Err(error) = recorded {
                tracing::warn!(deployment_id = %deployment.id, %error, "stream outcome was not recorded");
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures_util::stream;
    use litellm_framer::sse;
    use parking_lot::Mutex;
    use rstest::rstest;

    use super::*;
    use crate::messages::read_event;

    const START: &str = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":7,\"output_tokens\":1}}}\n\n";
    const DELTA: &str = "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{},\"usage\":{\"output_tokens\":5}}\n\n";
    const STOP: &str = "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    const OVERLOADED: &str = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\n\n";

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
        let chunks =
            stream::iter(chunks).map(|chunk| chunk.map(|text| Bytes::from_static(text.as_bytes())));
        let blocks = sse::blocks(chunks).boxed();
        let delivered = watched(blocks, read, dropped, move |outcome: Outcome| {
            sink.lock().push(
                outcome
                    .map(Counts::total)
                    .map_err(|failure| (failure.status, failure.kind)),
            );
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
        read_event as Read,
        vec![Ok(12)]
    )]
    #[case::a_stream_without_usage_counts_nothing(vec![Ok(STOP)], read_event as Read, vec![Ok(0)])]
    #[case::an_overloaded_messages_provider_fails_once_as_its_529(
        vec![Ok(START), Ok(OVERLOADED), Ok(STOP)],
        read_event as Read,
        vec![Err((Some(529), FailureKind::InternalServer))]
    )]
    #[case::a_break_fails_once_and_the_rest_passes_through(
        vec![Ok(START), Err(Dropped), Ok(STOP)],
        read_event as Read,
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
            heard(vec![Ok(START), Ok(head), Ok(tail)], read_event, usize::MAX).await,
            (2, vec![Ok(12)])
        );
    }

    #[tokio::test]
    async fn a_stream_the_client_stops_reading_reports_nothing() {
        assert_eq!(
            heard(vec![Ok(START), Ok(DELTA), Ok(STOP)], read_event, 2).await,
            (2, vec![])
        );
    }
}
