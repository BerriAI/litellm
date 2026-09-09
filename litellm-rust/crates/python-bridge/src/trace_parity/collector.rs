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
#[path = "../../tests/unit/trace_parity/collector.rs"]
mod tests;
