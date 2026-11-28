//! [`WorkerOutcomes`]: this process's successes and failures per deployment, which no other worker
//! sees. Python keeps them as the `local_only` `{id}:successes` and `{id}:fails` counters of
//! `router_utils/router_callbacks/track_deployment_metrics.py`.

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use litellm_router_core::Outcomes;
use parking_lot::Mutex;

const MINUTE: Duration = Duration::from_secs(60);

/// A count that lives for a minute from its first one.
#[derive(Clone, Copy)]
struct Window {
    count: u64,
    ends: Instant,
}

impl Window {
    fn live(window: Option<Self>, now: Instant) -> Option<Self> {
        window.filter(|window| window.ends > now)
    }

    fn counted(window: Option<Self>, now: Instant) -> Self {
        match Self::live(window, now) {
            Some(live) => Self {
                count: live.count + 1,
                ..live
            },
            None => Self {
                count: 1,
                ends: now + MINUTE,
            },
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Windows {
    successes: Option<Window>,
    fails: Option<Window>,
}

/// Every deployment's windows, read and counted at the time the caller gives.
#[derive(Default)]
struct Counts(HashMap<String, Windows>);

impl Counts {
    fn of(&self, id: &str, now: Instant) -> Outcomes {
        let windows = self.0.get(id).copied().unwrap_or_default();
        let count = |window| Window::live(window, now).map_or(0, |live| live.count);
        Outcomes {
            successes: count(windows.successes),
            fails: count(windows.fails),
        }
    }

    fn count_success(&mut self, id: &str, now: Instant) {
        let windows = self.0.entry(id.to_owned()).or_default();
        windows.successes = Some(Window::counted(windows.successes, now));
    }

    fn count_failure(&mut self, id: &str, now: Instant) {
        let windows = self.0.entry(id.to_owned()).or_default();
        windows.fails = Some(Window::counted(windows.fails, now));
    }
}

/// [`Counts`] behind a lock. Each call reads the clock once it holds the lock, so a count never
/// lands in a window that ended while the call waited for it.
#[derive(Default)]
pub(crate) struct WorkerOutcomes {
    counts: Mutex<Counts>,
}

impl WorkerOutcomes {
    pub(crate) fn of(&self, id: &str) -> Outcomes {
        let counts = self.counts.lock();
        counts.of(id, Instant::now())
    }

    pub(crate) fn count_success(&self, id: &str) {
        let mut counts = self.counts.lock();
        counts.count_success(id, Instant::now());
    }

    pub(crate) fn count_failure(&self, id: &str) {
        let mut counts = self.counts.lock();
        counts.count_failure(id, Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};

    use super::*;

    #[fixture]
    fn start() -> Instant {
        Instant::now()
    }

    fn outcomes(successes: u64, fails: u64) -> Outcomes {
        Outcomes { successes, fails }
    }

    #[rstest]
    #[case::counted_both_ways("a", outcomes(2, 1))]
    #[case::counted_one_way("b", outcomes(0, 1))]
    #[case::never_counted("c", outcomes(0, 0))]
    fn each_deployment_counts_its_own_successes_and_failures(
        start: Instant,
        #[case] id: &str,
        #[case] expected: Outcomes,
    ) {
        let mut counts = Counts::default();

        counts.count_success("a", start);
        counts.count_success("a", start);
        counts.count_failure("a", start);
        counts.count_failure("b", start);

        assert_eq!(counts.of(id, start), expected);
    }

    #[rstest]
    #[case::inside_both_minutes(59, outcomes(2, 1))]
    #[case::after_the_successes_minute(60, outcomes(0, 1))]
    #[case::after_both_minutes(90, outcomes(0, 0))]
    fn a_count_ends_a_minute_after_its_first_one_whatever_came_later(
        start: Instant,
        #[case] read_after: u64,
        #[case] expected: Outcomes,
    ) {
        let after = |seconds| start + Duration::from_secs(seconds);
        let mut counts = Counts::default();

        counts.count_success("a", start);
        counts.count_failure("a", after(30));
        counts.count_success("a", after(59));

        assert_eq!(counts.of("a", after(read_after)), expected);
    }

    #[rstest]
    #[case::as_it_is_counted(60, outcomes(0, 1))]
    #[case::inside_the_new_minute(119, outcomes(0, 1))]
    #[case::after_the_new_minute(120, outcomes(0, 0))]
    fn a_count_after_the_minute_starts_a_new_one(
        start: Instant,
        #[case] read_after: u64,
        #[case] expected: Outcomes,
    ) {
        let after = |seconds| start + Duration::from_secs(seconds);
        let mut counts = Counts::default();

        counts.count_failure("a", start);
        counts.count_failure("a", after(60));

        assert_eq!(counts.of("a", after(read_after)), expected);
    }
}
