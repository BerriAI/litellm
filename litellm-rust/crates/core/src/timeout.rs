use std::time::Duration;

/// A caller's timeout in seconds. Non-positive and non-finite values mean no timeout,
/// matching how the Python SDK treats them.
pub fn from_seconds(seconds: Option<f64>) -> Option<Duration> {
    seconds
        .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
        .map(Duration::from_secs_f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_positive_finite_seconds_become_a_timeout() {
        assert_eq!(from_seconds(Some(1.5)), Some(Duration::from_millis(1500)));
        for seconds in [
            None,
            Some(0.0),
            Some(-1.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
        ] {
            assert_eq!(from_seconds(seconds), None, "{seconds:?}");
        }
    }
}
