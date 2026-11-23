//! The memory tier in front of the rotation grace-period lookup.

use std::{
    collections::{HashMap, VecDeque},
    time::{Duration, SystemTime},
};

use chrono::NaiveDateTime;

/// How long a found grace-period mapping is served from memory before the database is asked again.
const DEPRECATED_KEY_TTL: Duration = Duration::from_secs(60);
/// How many grace-period mappings are held in memory; the oldest goes first.
const DEPRECATED_KEY_CAPACITY: usize = 1000;

/// Grace-period mappings found in the database, each served until its memory TTL or its revoke time,
/// whichever comes first. Only found mappings are held, so a miss never evicts a real one.
#[derive(Default)]
pub(super) struct DeprecatedKeys {
    entries: HashMap<String, DeprecatedEntry>,
    order: VecDeque<String>,
}

struct DeprecatedEntry {
    active_token: String,
    serve_until: SystemTime,
}

impl DeprecatedKeys {
    pub(super) fn active_token(&mut self, token: &str, now: SystemTime) -> Option<String> {
        let entry = self.entries.get(token)?;
        if now < entry.serve_until {
            return Some(entry.active_token.clone());
        }
        self.remove(token);
        None
    }

    pub(super) fn insert(
        &mut self,
        token: &str,
        active_token: &str,
        now: SystemTime,
        revoke_at: NaiveDateTime,
    ) {
        let revoke_at = SystemTime::from(revoke_at.and_utc());
        let serve_until = (now + DEPRECATED_KEY_TTL).min(revoke_at);
        if self.entries.contains_key(token) {
            self.remove(token);
        }
        while self.entries.len() >= DEPRECATED_KEY_CAPACITY {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
        self.order.push_back(token.to_owned());
        self.entries.insert(
            token.to_owned(),
            DeprecatedEntry {
                active_token: active_token.to_owned(),
                serve_until,
            },
        );
    }

    fn remove(&mut self, token: &str) {
        if self.entries.remove(token).is_some() {
            self.order.retain(|held| held != token);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use chrono::{DateTime, NaiveDateTime, Utc};
    use rstest::rstest;

    use super::{DEPRECATED_KEY_CAPACITY, DeprecatedKeys};

    fn at(seconds: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1_800_000_000 + seconds)
    }

    fn naive(time: SystemTime) -> NaiveDateTime {
        DateTime::<Utc>::from(time).naive_utc()
    }

    #[rstest]
    #[case::fresh(at(30), Some("active"))]
    #[case::memory_ttl_elapsed(at(60), None)]
    fn deprecated_mapping_is_served_for_sixty_seconds(
        #[case] now: SystemTime,
        #[case] expected: Option<&str>,
    ) {
        let mut keys = DeprecatedKeys::default();
        keys.insert("old", "active", at(0), naive(at(3600)));

        assert_eq!(keys.active_token("old", now).as_deref(), expected);
    }

    #[rstest]
    fn deprecated_mapping_stops_at_revoke_time() {
        let mut keys = DeprecatedKeys::default();
        keys.insert("old", "active", at(0), naive(at(10)));

        assert_eq!(
            (
                keys.active_token("old", at(9)).as_deref(),
                keys.active_token("old", at(10))
            ),
            (Some("active"), None)
        );
    }

    #[rstest]
    fn oldest_deprecated_mapping_is_evicted_at_capacity() {
        let mut keys = DeprecatedKeys::default();
        for index in 0..=DEPRECATED_KEY_CAPACITY {
            keys.insert(&format!("old-{index}"), "active", at(0), naive(at(3600)));
        }

        assert_eq!(
            (
                keys.active_token("old-0", at(1)),
                keys.active_token(&format!("old-{DEPRECATED_KEY_CAPACITY}"), at(1))
                    .as_deref(),
                keys.entries.len(),
            ),
            (None, Some("active"), DEPRECATED_KEY_CAPACITY)
        );
    }
}
