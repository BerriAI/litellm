use std::{
    sync::Mutex,
    time::{Duration, Instant},
};

use litellm_circuit_breaker::{State, Transition};
use litellm_core_utils::settings::Lookup;
use litellm_redis_client::{CallEvent, FailureKind, Outcome, RedisEvents};

const DEFAULT_TIMEOUT_LOG_INTERVAL: Duration = Duration::from_secs(5);

pub struct TracingEvents {
    throttle: TimeoutLogThrottle,
}

impl TracingEvents {
    pub fn new(lookup: &dyn Lookup) -> Self {
        let interval = lookup
            .get("REDIS_TIMEOUT_LOG_INTERVAL")
            .and_then(|seconds| seconds.parse::<f64>().ok())
            .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
            .unwrap_or(DEFAULT_TIMEOUT_LOG_INTERVAL);
        Self {
            throttle: TimeoutLogThrottle::new(interval),
        }
    }
}

impl RedisEvents for TracingEvents {
    fn call(&self, event: &CallEvent) {
        let Outcome::Failed(kind) = event.outcome else {
            return;
        };
        match kind {
            FailureKind::BreakerOpen => {
                tracing::debug!(operation = ?event.operation, "redis call rejected by the open circuit breaker")
            }
            FailureKind::Timeout => match self.throttle.admit(Instant::now()) {
                None => {
                    tracing::debug!(operation = ?event.operation, duration = ?event.duration, "redis call timed out")
                }
                Some(0) => {
                    tracing::error!(operation = ?event.operation, duration = ?event.duration, "redis call timed out")
                }
                Some(suppressed) => tracing::error!(
                    operation = ?event.operation,
                    duration = ?event.duration,
                    suppressed,
                    "redis call timed out, more redis timeouts since the previous line were logged at debug"
                ),
            },
            FailureKind::Transport | FailureKind::Server | FailureKind::Decode => {
                tracing::error!(operation = ?event.operation, failure = ?kind, "redis call failed")
            }
        }
    }

    fn breaker(&self, transition: &Transition<State>) {
        match transition.to {
            State::Open { .. } => tracing::warn!(
                generation = transition.generation,
                "redis circuit breaker opened, fast-failing redis calls"
            ),
            State::HalfOpen => tracing::info!(
                generation = transition.generation,
                "redis circuit breaker half open, probing redis"
            ),
            State::Closed { .. } => tracing::info!(
                generation = transition.generation,
                "redis circuit breaker closed, redis recovered"
            ),
        }
    }
}

struct TimeoutLogThrottle {
    interval: Duration,
    state: Mutex<(Option<Instant>, u64)>,
}

impl TimeoutLogThrottle {
    fn new(interval: Duration) -> Self {
        Self {
            interval,
            state: Mutex::new((None, 0)),
        }
    }

    fn admit(&self, now: Instant) -> Option<u64> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (last_logged_at, suppressed) = &mut *state;
        if last_logged_at.is_some_and(|last| now.duration_since(last) < self.interval) {
            *suppressed += 1;
            return None;
        }
        *last_logged_at = Some(now);
        Some(std::mem::take(suppressed))
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use rstest::rstest;

    use super::TimeoutLogThrottle;

    #[rstest]
    fn admits_one_line_per_interval_and_counts_the_rest() {
        let throttle = TimeoutLogThrottle::new(Duration::from_secs(5));
        let start = Instant::now();
        let at = |seconds: u64| start + Duration::from_secs(seconds);

        let admitted = [
            throttle.admit(at(0)),
            throttle.admit(at(1)),
            throttle.admit(at(4)),
            throttle.admit(at(5)),
            throttle.admit(at(6)),
            throttle.admit(at(20)),
        ];

        assert_eq!(admitted, [Some(0), None, None, Some(2), None, Some(1)]);
    }
}
