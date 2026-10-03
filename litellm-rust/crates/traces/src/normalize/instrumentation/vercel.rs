use super::SpanContext;

pub(super) fn matches(context: &SpanContext<'_>) -> bool {
    matches!(context.scope, "gen_ai" | "ai")
}
