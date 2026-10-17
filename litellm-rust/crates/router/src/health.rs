use std::sync::{
    Arc,
    atomic::{AtomicU32, AtomicU64, Ordering::Relaxed},
};

use crate::CooldownPolicy;

const FAILURE_RATE_THRESHOLD: f64 = 0.5;
const FAILURE_RATE_MINIMUM_REQUESTS: u32 = 5;
const SINGLE_DEPLOYMENT_MINIMUM_REQUESTS: u32 = 1000;

/// A failed attempt the router counts against its deployment and may retry elsewhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Failure {
    /// 429
    RateLimited,
    /// 401, 402 or 404: this deployment's credentials or model are wrong
    Rejected,
    /// 408, 5xx, a refused connection, a reset or a timeout
    Unavailable,
}

/// Lock-free per-deployment counters. Times are milliseconds since the router started.
#[derive(Debug, Default)]
pub(crate) struct Health {
    in_flight: AtomicU32,
    window: AtomicU64,
    successes: AtomicU32,
    failures: AtomicU32,
    cooldown_until: AtomicU64,
}

impl Health {
    pub(crate) fn in_flight(&self) -> u32 {
        self.in_flight.load(Relaxed)
    }

    pub(crate) fn cooling(&self, now: u64) -> bool {
        self.cooldown_until.load(Relaxed) > now
    }

    pub(crate) fn record_success(&self, now: u64, policy: &CooldownPolicy) {
        self.roll(now, policy);
        self.successes.fetch_add(1, Relaxed);
    }

    /// Counts a failure and returns whether it put the deployment in cooldown. With
    /// `allowed_fails` set this is Python's v1 rule (more than `allowed_fails` failures in
    /// the window, counted across replicas when `fleet_failures` carries the shared count);
    /// without it, Python's v2 rule (a 429 or a rejected call when the group has
    /// alternatives, or a failure rate over the current minute), which Python also keeps
    /// per process.
    pub(crate) fn record_failure(
        &self,
        now: u64,
        failure: Failure,
        policy: &CooldownPolicy,
        alternatives: bool,
        fleet_failures: Option<u32>,
    ) -> bool {
        self.roll(now, policy);
        let failures = self.failures.fetch_add(1, Relaxed).saturating_add(1);
        let cool = match policy.allowed_fails {
            Some(allowed) => fleet_failures.unwrap_or(failures).max(failures) > allowed,
            None => {
                let total = failures.saturating_add(self.successes.load(Relaxed));
                let failing = (failures == total && total >= SINGLE_DEPLOYMENT_MINIMUM_REQUESTS)
                    || (alternatives
                        && total >= FAILURE_RATE_MINIMUM_REQUESTS
                        && f64::from(failures) / f64::from(total) > FAILURE_RATE_THRESHOLD);
                failing
                    || match failure {
                        Failure::RateLimited => alternatives,
                        Failure::Rejected => true,
                        Failure::Unavailable => false,
                    }
            }
        };
        if cool {
            let cooldown = u64::try_from(policy.cooldown_time.as_millis()).unwrap_or(u64::MAX);
            self.cool_until(now.saturating_add(cooldown));
        }
        cool
    }

    /// Cools the deployment until `until`, unless a longer cooldown is already running.
    pub(crate) fn cool_until(&self, until: u64) {
        self.cooldown_until.fetch_max(until, Relaxed);
    }

    fn roll(&self, now: u64, policy: &CooldownPolicy) {
        let length = u64::try_from(policy.window().as_millis())
            .unwrap_or(u64::MAX)
            .max(1);
        let window = now / length;
        let seen = self.window.load(Relaxed);
        if seen != window
            && self
                .window
                .compare_exchange(seen, window, Relaxed, Relaxed)
                .is_ok()
        {
            self.successes.store(0, Relaxed);
            self.failures.store(0, Relaxed);
        }
    }
}

/// Counts a call in flight on its deployment until dropped; least-busy routing reads it.
#[derive(Debug)]
pub struct InFlight(Arc<Health>);

impl InFlight {
    pub(crate) fn enter(health: &Arc<Health>) -> Self {
        health.in_flight.fetch_add(1, Relaxed);
        Self(health.clone())
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rstest::rstest;

    use super::*;

    const V1: CooldownPolicy = CooldownPolicy {
        allowed_fails: Some(2),
        cooldown_time: Duration::from_secs(10),
    };
    const V2: CooldownPolicy = CooldownPolicy {
        allowed_fails: None,
        cooldown_time: Duration::from_secs(5),
    };

    fn fail(health: &Health, times: u32, failure: Failure, policy: &CooldownPolicy) -> bool {
        (0..times).fold(false, |_, _| {
            health.record_failure(1_000, failure, policy, true, None)
        })
    }

    fn succeed(health: &Health, times: u32) {
        for _ in 0..times {
            health.record_success(1_000, &V2);
        }
    }

    #[rstest]
    #[case::within_allowed_fails(2, false)]
    #[case::over_allowed_fails(3, true)]
    fn allowed_fails_cools_after_more_failures_than_allowed(
        #[case] failures: u32,
        #[case] cooling: bool,
    ) {
        let health = Health::default();

        assert_eq!(fail(&health, failures, Failure::Unavailable, &V1), cooling);
        assert_eq!(health.cooling(1_000), cooling);
    }

    #[rstest]
    #[case::during(10_999, true)]
    #[case::after(11_000, false)]
    fn cooldown_lasts_cooldown_time(#[case] now: u64, #[case] cooling: bool) {
        let health = Health::default();
        fail(&health, 3, Failure::RateLimited, &V1);

        assert_eq!(health.cooling(now), cooling);
    }

    #[rstest]
    fn allowed_fails_counter_resets_with_its_window() {
        let health = Health::default();
        fail(&health, 2, Failure::Unavailable, &V1);

        assert!(!health.record_failure(12_000, Failure::Unavailable, &V1, true, None));
        assert!(!health.cooling(12_000));
    }

    #[rstest]
    #[case::rate_limited_with_alternatives(Failure::RateLimited, true, true)]
    #[case::rate_limited_alone(Failure::RateLimited, false, false)]
    #[case::rejected(Failure::Rejected, false, true)]
    #[case::one_unavailable(Failure::Unavailable, true, false)]
    fn without_allowed_fails_one_failure_follows_python_v2(
        #[case] failure: Failure,
        #[case] alternatives: bool,
        #[case] cooling: bool,
    ) {
        let health = Health::default();

        assert_eq!(
            health.record_failure(1_000, failure, &V2, alternatives, None),
            cooling
        );
    }

    #[rstest]
    #[case::majority_failing(2, 3, true)]
    #[case::half_failing(3, 3, false)]
    #[case::too_few_requests(0, 4, false)]
    fn without_allowed_fails_a_failure_rate_cools(
        #[case] successes: u32,
        #[case] failures: u32,
        #[case] cooling: bool,
    ) {
        let health = Health::default();
        succeed(&health, successes);

        assert_eq!(fail(&health, failures, Failure::Unavailable, &V2), cooling);
    }

    #[rstest]
    #[case::fleet_over_allowed(Some(3), true)]
    #[case::fleet_within_allowed(Some(2), false)]
    #[case::fleet_behind_this_replica(Some(0), false)]
    fn allowed_fails_counts_the_fleet_when_shared(
        #[case] fleet_failures: Option<u32>,
        #[case] cooling: bool,
    ) {
        let health = Health::default();

        assert_eq!(
            health.record_failure(1_000, Failure::Unavailable, &V1, true, fleet_failures),
            cooling
        );
    }

    #[rstest]
    fn a_shared_cooldown_never_shortens_a_local_one() {
        let health = Health::default();
        fail(&health, 3, Failure::Unavailable, &V1);
        health.cool_until(5_000);

        assert!(health.cooling(10_999));
        health.cool_until(20_000);
        assert!(health.cooling(19_999));
    }

    #[rstest]
    fn in_flight_counts_until_dropped() {
        let health = Arc::new(Health::default());
        let first = InFlight::enter(&health);
        let second = InFlight::enter(&health);
        drop(first);

        assert_eq!(health.in_flight(), 1);
        drop(second);
        assert_eq!(health.in_flight(), 0);
    }
}
