use tracing::span::Id;
use tracing::{Level, Metadata, Subscriber};
use tracing_subscriber::filter::{FilterFn, LevelFilter, filter_fn};
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;

use crate::constants::FUNCTION_TRACE_TARGET;

pub mod function_trace;

pub use function_trace::{FunctionTrace, FunctionTraceEvent};

pub fn function_trace_filter() -> FilterFn<impl Fn(&Metadata<'_>) -> bool> {
    filter_fn(|metadata| {
        metadata.is_span()
            && metadata.target() == FUNCTION_TRACE_TARGET
            && *metadata.level() == Level::TRACE
    })
    .with_max_level_hint(LevelFilter::TRACE)
}

pub fn span_depth<S>(context: &Context<'_, S>, id: &Id) -> usize
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    context
        .span(id)
        .map(|span| span.scope().skip(1).count())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use tracing::instrument::WithSubscriber;

    use super::*;

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
