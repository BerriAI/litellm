pub(super) fn pin(tenant: &str, session_id: &str) -> String {
    format!("{{sensitive_route:{tenant}:{session_id}}}:model")
}
