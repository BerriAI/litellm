pub fn is_pollable_background_interaction(status: &str, interaction_id: &str) -> bool {
    matches!(status, "in_progress" | "queued") && !interaction_id.is_empty()
}

pub fn missing_usage_is_expected(status: &str) -> bool {
    !matches!(status, "completed" | "requires_action")
}

pub fn poll_intervals(initial: f64, maximum: f64, timeout: f64) -> impl Iterator<Item = f64> {
    std::iter::successors(Some((0.0, initial)), move |(elapsed, interval)| {
        let next_elapsed = elapsed + interval;
        let next_interval = (interval * 2.0).min(maximum);
        (next_interval > 0.0 && next_elapsed + next_interval <= timeout)
            .then_some((next_elapsed, next_interval))
    })
    .take_while(move |(elapsed, interval)| *interval > 0.0 && elapsed + interval <= timeout)
    .map(|(_, interval)| interval)
}
