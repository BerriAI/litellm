use super::SpanContext;

pub(super) fn matches(context: &SpanContext<'_>) -> bool {
    context.scope.starts_with("strands.")
}
