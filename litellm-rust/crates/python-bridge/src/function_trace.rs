use std::sync::{Arc, Mutex};

use serde::Serialize;
use tracing::span::{Attributes, Id};
use tracing::{Dispatch, Level, Subscriber};
use tracing_subscriber::filter::{LevelFilter, filter_fn};
use tracing_subscriber::layer::Context;
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::{Layer, Registry};

use crate::constants::FUNCTION_TRACE_TARGET;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FunctionTraceEvent {
    pub function: &'static str,
    pub depth: usize,
}

#[derive(Clone, Default)]
pub struct FunctionTrace {
    events: Arc<Mutex<Vec<FunctionTraceEvent>>>,
}

impl FunctionTrace {
    pub fn dispatcher(&self) -> Dispatch {
        let filter = filter_fn(|metadata| {
            metadata.is_span()
                && metadata.target() == FUNCTION_TRACE_TARGET
                && *metadata.level() == Level::TRACE
        })
        .with_max_level_hint(LevelFilter::TRACE);
        Dispatch::new(
            Registry::default().with(
                FunctionTraceLayer {
                    trace: self.clone(),
                }
                .with_filter(filter),
            ),
        )
    }

    pub fn events(&self) -> Vec<FunctionTraceEvent> {
        self.events
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }
}

struct FunctionTraceLayer {
    trace: FunctionTrace,
}

impl<S> Layer<S> for FunctionTraceLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attributes: &Attributes<'_>, id: &Id, context: Context<'_, S>) {
        let depth = context
            .span(id)
            .map(|span| span.scope().skip(1).count())
            .unwrap_or_default();
        self.trace
            .events
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(FunctionTraceEvent {
                function: attributes.metadata().name(),
                depth,
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    async fn outer() {
        tokio::task::yield_now().await;
        inner().await;
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    async fn inner() {
        tokio::task::yield_now().await;
    }

    #[tokio::test]
    async fn concurrent_futures_keep_separate_traces_across_yields() {
        use tracing::instrument::WithSubscriber;

        let first = FunctionTrace::default();
        let second = FunctionTrace::default();
        let outside = FunctionTrace::default();

        async {
            tokio::join!(
                outer().with_subscriber(first.dispatcher()),
                inner().with_subscriber(second.dispatcher()),
            );
            inner().await;
        }
        .with_subscriber(outside.dispatcher())
        .await;

        assert_eq!(
            first.events(),
            vec![
                FunctionTraceEvent {
                    function: "outer",
                    depth: 0
                },
                FunctionTraceEvent {
                    function: "inner",
                    depth: 1
                },
            ],
        );
        assert_eq!(
            second.events(),
            vec![FunctionTraceEvent {
                function: "inner",
                depth: 0
            }],
        );
        assert_eq!(
            outside.events(),
            vec![FunctionTraceEvent {
                function: "inner",
                depth: 0
            }],
        );
    }

    #[test]
    fn records_matching_spans_in_creation_order() {
        let trace = FunctionTrace::default();
        let dispatch = trace.dispatcher();

        tracing::dispatcher::with_default(&dispatch, || {
            let _ignored = tracing::trace_span!(target: "other", "ignored");
            let _first = tracing::trace_span!(target: FUNCTION_TRACE_TARGET, "same_name");
            let _wrong_level = tracing::debug_span!(target: FUNCTION_TRACE_TARGET, "wrong_level");
            let _second = tracing::trace_span!(target: FUNCTION_TRACE_TARGET, "same_name");
        });

        assert_eq!(
            trace.events(),
            vec![
                FunctionTraceEvent {
                    function: "same_name",
                    depth: 0,
                },
                FunctionTraceEvent {
                    function: "same_name",
                    depth: 0,
                },
            ]
        );
    }

    #[test]
    fn records_matching_span_nesting_depth() {
        let trace = FunctionTrace::default();
        let dispatch = trace.dispatcher();

        tracing::dispatcher::with_default(&dispatch, || {
            let outer = tracing::trace_span!(target: FUNCTION_TRACE_TARGET, "outer");
            let _outer_guard = outer.enter();
            let _inner = tracing::trace_span!(target: FUNCTION_TRACE_TARGET, "inner");
        });

        assert_eq!(
            trace.events(),
            vec![
                FunctionTraceEvent {
                    function: "outer",
                    depth: 0,
                },
                FunctionTraceEvent {
                    function: "inner",
                    depth: 1,
                },
            ]
        );
    }
}
