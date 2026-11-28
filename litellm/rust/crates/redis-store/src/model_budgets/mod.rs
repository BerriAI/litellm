//! Redis counters shared with the proxy's per-model budget limiter.

mod keys;

use std::{sync::Arc, time::Duration};

use jiff::{Span, Timestamp, tz::TimeZone};
use litellm_redis_client::commands;

pub use self::keys::{Counter, Scope};
use crate::{
    Clock, Error, Number, Remote, codec,
    dual_cache::{DualCache, MemorySettings, PipelineIncrement},
    remote::RemoteSlot,
    system_clock,
};

/// Duration of a per-model budget, measured from the instant the window began.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Period {
    Seconds(u64),
    Months(i32),
}

impl Period {
    pub fn parse(raw: &str) -> Option<Self> {
        let duration = match raw.trim().to_ascii_lowercase().as_str() {
            "hourly" => "1h",
            "daily" => "24h",
            "weekly" => "7d",
            "monthly" => "30d",
            _ => raw,
        };
        let (count, unit) = duration.split_at(duration.find(|c: char| !c.is_ascii_digit())?);
        let count: u64 = count.parse().ok()?;
        if unit.starts_with("mo") {
            return Some(Self::Months(i32::try_from(count).ok()?));
        }
        let scale = match unit.as_bytes().first()? {
            b's' => 1,
            b'm' => 60,
            b'h' => 3600,
            b'd' => 86400,
            b'w' => 604800,
            _ => return None,
        };
        Some(Self::Seconds(count.checked_mul(scale)?))
    }

    pub fn duration(self, start: Timestamp) -> Option<Duration> {
        let seconds = match self {
            Self::Seconds(seconds) => seconds,
            Self::Months(months) => {
                let end = start
                    .to_zoned(TimeZone::UTC)
                    .checked_add(Span::new().try_months(months).ok()?)
                    .ok()?
                    .timestamp();
                end.duration_since(start).as_secs().try_into().ok()?
            }
        };
        (seconds > 0).then(|| Duration::from_secs(seconds))
    }
}

/// The shared start and spend keys for a model budget.
pub struct ModelBudgets {
    cache: DualCache<Number>,
    clock: Clock,
    started_at: Duration,
}

impl ModelBudgets {
    pub fn new(remote: Option<Remote>) -> Self {
        Self::with_clock(remote, system_clock())
    }

    fn with_clock(remote: Option<Remote>, clock: Clock) -> Self {
        Self {
            started_at: clock(),
            clock: clock.clone(),
            cache: DualCache::new(
                Arc::new(RemoteSlot::new(remote)),
                MemorySettings::default(),
                clock,
            ),
        }
    }

    pub fn attach(&self, remote: Remote) -> bool {
        self.cache.redis().attach(remote)
    }

    pub async fn spend(&self, counter: &Counter<'_>) -> Result<f64, Error> {
        let key = counter.spend();
        let spend = match self.cache.redis().client() {
            Some(client) => client
                .send(commands::get::<Vec<u8>>(&key))
                .await?
                .as_deref()
                .map(codec::decode::<Number>)
                .transpose()?,
            None => self.cache.memory().get(&key)?,
        };
        Ok(spend.map(Number::as_f64).unwrap_or_default())
    }

    pub async fn spend_with_legacy(
        &self,
        counter: &Counter<'_>,
        requested: &str,
        now: Timestamp,
    ) -> Result<f64, Error> {
        let spend = self.spend(counter).await?;
        let carry_legacy = matches!(counter.scope, Scope::Key | Scope::EndUser)
            && counter.model != requested
            && Period::parse(counter.duration)
                .and_then(|period| period.duration(now))
                .is_some_and(|duration| (self.clock)().saturating_sub(self.started_at) < duration);
        if !carry_legacy {
            return Ok(spend);
        }
        self.spend(&Counter {
            model: requested,
            ..*counter
        })
        .await
        .map(|legacy| spend + legacy)
    }

    pub async fn charge(
        &self,
        counter: &Counter<'_>,
        period: Period,
        now: Timestamp,
        cost: f64,
    ) -> Result<(), Error> {
        let Some(ttl) = period.duration(now) else {
            return Ok(());
        };
        let start_key = counter.start();
        let now_seconds = now.as_second() as f64 + f64::from(now.subsec_nanosecond()) / 1e9;
        let started = Number::Float(now_seconds);
        let start = match self.cache.get(None, &start_key, None).await? {
            Some(start) => start.as_f64(),
            None => {
                self.cache.set(&start_key, started, Some(ttl)).await?;
                now_seconds
            }
        };
        let window = Timestamp::from_second(start.floor() as i64)
            .ok()
            .and_then(|start| period.duration(start))
            .unwrap_or(ttl);
        let remaining = window.as_secs_f64() - (now_seconds - start);
        let spend_key = counter.spend();
        if remaining < 0.0 {
            self.cache
                .set(&spend_key, Number::Float(cost), Some(ttl))
                .await?;
            self.cache.set(&start_key, started, Some(ttl)).await?;
        } else {
            self.cache
                .increment_pipeline(&[PipelineIncrement {
                    key: spend_key,
                    amount: Number::Float(cost),
                    ttl: Some(Duration::from_secs(remaining as u64)),
                }])
                .await;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::key_includes_legacy(Scope::Key, "provider/claude", 3599, 0.75)]
    #[case::end_user_includes_legacy(Scope::EndUser, "provider/claude", 3599, 0.75)]
    #[case::team_has_no_legacy(Scope::Team, "provider/claude", 3599, 0.25)]
    #[case::owner_has_no_legacy(Scope::User, "provider/claude", 3599, 0.25)]
    #[case::member_has_no_legacy(Scope::TeamMember, "provider/claude", 3599, 0.25)]
    #[case::same_name_only_counts_once(Scope::Key, "claude", 3599, 0.25)]
    #[case::legacy_read_stops_after_one_period(Scope::Key, "provider/claude", 3600, 0.25)]
    #[tokio::test]
    async fn legacy_spend_is_counted_only_during_the_upgrade_window(
        #[case] scope: Scope,
        #[case] requested: &str,
        #[case] elapsed: u64,
        #[case] expected: f64,
    ) {
        let ticks = Arc::new(AtomicU64::new(100));
        let reading = ticks.clone();
        let store = ModelBudgets::with_clock(
            None,
            Arc::new(move || Duration::from_secs(reading.load(Ordering::Relaxed))),
        );
        ticks.fetch_add(elapsed, Ordering::Relaxed);
        let counter = Counter {
            scope,
            id: "caller",
            model: "claude",
            duration: "1h",
        };
        let legacy = Counter {
            model: "provider/claude",
            ..counter
        };
        let now = Timestamp::from_second(1_800_000_000).unwrap();
        let period = Period::parse(counter.duration).unwrap();
        store.charge(&counter, period, now, 0.25).await.unwrap();
        store.charge(&legacy, period, now, 0.5).await.unwrap();
        assert_eq!(
            store
                .spend_with_legacy(&counter, requested, now)
                .await
                .unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::fixed("daily", "2024-01-31T12:00:00Z", 1)]
    #[case::short_month("1mo", "2024-01-31T12:00:00Z", 29)]
    #[case::long_month("1mo", "2024-03-01T12:00:00Z", 31)]
    fn duration_is_measured_from_when_the_window_started(
        #[case] period: &str,
        #[case] start: &str,
        #[case] days: u64,
    ) {
        assert_eq!(
            Period::parse(period)
                .unwrap()
                .duration(start.parse().unwrap()),
            Some(Duration::from_secs(days * 86400)),
        );
    }
}
