pub(super) fn cooldown(model_id: &str) -> String {
    format!("deployment:{model_id}:cooldown")
}

pub(super) fn cooldowns(model_ids: &[String]) -> Vec<String> {
    model_ids
        .iter()
        .map(|model_id| cooldown(model_id))
        .collect()
}

pub(super) fn allowed_fails(model_id: &str, suffix: Option<&str>) -> String {
    match suffix.filter(|suffix| !suffix.is_empty()) {
        Some(suffix) => format!("deployment:{model_id}:allowed_fails:{suffix}"),
        None => format!("deployment:{model_id}:allowed_fails"),
    }
}
