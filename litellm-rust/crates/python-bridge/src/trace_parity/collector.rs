use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tracing::span::{Attributes, Id};
use tracing::{Dispatch, Level, Metadata, Subscriber};
use tracing_subscriber::filter::{FilterFn, LevelFilter, filter_fn};
use tracing_subscriber::layer::Context;
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::{Layer, Registry};

use litellm_core::constants::FUNCTION_TRACE_TARGET;

fn function_trace_filter() -> FilterFn<impl Fn(&Metadata<'_>) -> bool> {
    filter_fn(|metadata| {
        metadata.is_span()
            && metadata.target() == FUNCTION_TRACE_TARGET
            && *metadata.level() == Level::TRACE
    })
    .with_max_level_hint(LevelFilter::TRACE)
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct FunctionTraceEvent {
    id: usize,
    parent_id: Option<usize>,
    function: &'static str,
    module_path: Option<&'static str>,
    file: Option<&'static str>,
    line: Option<u32>,
}

#[derive(Clone, Default)]
pub(super) struct FunctionTrace {
    events: Arc<Mutex<Vec<FunctionTraceEvent>>>,
    span_events: Arc<Mutex<HashMap<Id, usize>>>,
}

impl FunctionTrace {
    pub(super) fn dispatcher(&self) -> Dispatch {
        Dispatch::new(
            Registry::default().with(
                FunctionTraceLayer {
                    trace: self.clone(),
                }
                .with_filter(function_trace_filter()),
            ),
        )
    }

    pub(super) fn events(&self) -> Vec<FunctionTraceEvent> {
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
        let parent_id = context.span(id).and_then(|span| {
            let span_events = self
                .trace
                .span_events
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            span.scope()
                .skip(1)
                .find_map(|ancestor| span_events.get(&ancestor.id()).copied())
        });
        let mut events = self
            .trace
            .events
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let event_id = events.len();
        events.push(FunctionTraceEvent {
            id: event_id,
            parent_id,
            function: attributes.metadata().name(),
            module_path: attributes.metadata().module_path(),
            file: attributes.metadata().file(),
            line: attributes.metadata().line(),
        });
        self.trace
            .span_events
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(id.clone(), event_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing::instrument::WithSubscriber;

    fn event(
        id: usize,
        parent_id: Option<usize>,
        function: &'static str,
    ) -> (usize, Option<usize>, &'static str) {
        (id, parent_id, function)
    }

    fn structural_events(
        events: &[FunctionTraceEvent],
    ) -> Vec<(usize, Option<usize>, &'static str)> {
        events
            .iter()
            .map(|event| (event.id, event.parent_id, event.function))
            .collect()
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    async fn outer() {
        tokio::task::yield_now().await;
        inner().await;
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    async fn inner() {
        tokio::task::yield_now().await;
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    async fn concurrent_parent() {
        tokio::join!(inner(), inner());
    }

    #[tokio::test]
    async fn concurrent_futures_keep_separate_traces_across_yields() {
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
            structural_events(&first.events()),
            vec![event(0, None, "outer"), event(1, Some(0), "inner")],
        );
        assert_eq!(
            structural_events(&second.events()),
            vec![event(0, None, "inner")],
        );
        assert_eq!(
            structural_events(&outside.events()),
            vec![event(0, None, "inner")],
        );
    }

    #[tokio::test]
    async fn concurrent_siblings_keep_the_same_parent() {
        let trace = FunctionTrace::default();
        concurrent_parent()
            .with_subscriber(trace.dispatcher())
            .await;

        assert_eq!(
            structural_events(&trace.events()),
            vec![
                event(0, None, "concurrent_parent"),
                event(1, Some(0), "inner"),
                event(2, Some(0), "inner"),
            ]
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
            structural_events(&trace.events()),
            vec![event(0, None, "same_name"), event(1, None, "same_name")]
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
            structural_events(&trace.events()),
            vec![event(0, None, "outer"), event(1, Some(0), "inner")]
        );
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    async fn instrumented_with_literal_target() {}

    #[tokio::test]
    async fn literal_instrument_target_matches_filter_constant() {
        assert_eq!(FUNCTION_TRACE_TARGET, "litellm::function_trace");

        let trace = FunctionTrace::default();
        instrumented_with_literal_target()
            .with_subscriber(trace.dispatcher())
            .await;

        let events = trace.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, 0);
        assert_eq!(events[0].parent_id, None);
        assert_eq!(events[0].function, "instrumented_with_literal_target");
        assert_eq!(events[0].module_path, Some(module_path!()));
        assert_eq!(events[0].file, Some(file!()));
        assert!(events[0].line.is_some());
    }
}
