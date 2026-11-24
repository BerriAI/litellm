use std::{collections::HashMap, hash::Hash};

use indexmap::IndexMap;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SeriesLimits {
    pub max_series: Option<i64>,
    pub ttl_seconds: Option<f64>,
    pub cleanup_interval_seconds: Option<f64>,
}

impl SeriesLimits {
    pub const fn enabled(&self) -> bool {
        self.max_series.is_some() || self.ttl_seconds.is_some()
    }
}

pub struct Admission<V> {
    pub admitted: bool,
    pub removed: Vec<Vec<V>>,
}

pub struct Tracker<V> {
    series: HashMap<String, IndexMap<Vec<V>, f64>>,
    last_cleanup: HashMap<String, f64>,
}

impl<V> Default for Tracker<V> {
    fn default() -> Self {
        Self {
            series: HashMap::new(),
            last_cleanup: HashMap::new(),
        }
    }
}

impl<V: Clone + Eq + Hash> Tracker<V> {
    fn cleanup_due(&mut self, name: &str, now: f64, interval: Option<f64>) -> bool {
        let due = match (interval, self.last_cleanup.get(name)) {
            (Some(interval), Some(last)) if interval > 0.0 => now - last >= interval,
            _ => true,
        };
        if due {
            self.last_cleanup.insert(name.to_owned(), now);
        }
        due
    }

    fn expired(&self, name: &str, now: f64, ttl: f64) -> Vec<Vec<V>> {
        self.series.get(name).map_or_else(Vec::new, |series| {
            series
                .iter()
                .filter(|(_, last_seen)| now - **last_seen > ttl)
                .map(|(values, _)| values.clone())
                .collect()
        })
    }

    pub fn admit(
        &mut self,
        name: &str,
        values: &[V],
        now: f64,
        limits: &SeriesLimits,
    ) -> Admission<V> {
        self.series.entry(name.to_owned()).or_default();
        let removed = match limits.ttl_seconds {
            Some(ttl) if self.cleanup_due(name, now, limits.cleanup_interval_seconds) => {
                let expired = self.expired(name, now, ttl);
                for gone in &expired {
                    self.forget(name, gone);
                }
                expired
            }
            _ => Vec::new(),
        };
        let series = self.series.entry(name.to_owned()).or_default();
        let full = limits
            .max_series
            .is_some_and(|max| i64::try_from(series.len()).unwrap_or(i64::MAX) >= max);
        if !series.contains_key(values) && full {
            return Admission {
                admitted: false,
                removed,
            };
        }
        series.shift_remove(values);
        series.insert(values.to_vec(), now);
        Admission {
            admitted: true,
            removed,
        }
    }

    pub fn track_due(
        &mut self,
        name: &str,
        values: &[V],
        now: f64,
        limits: &SeriesLimits,
    ) -> Vec<Vec<V>> {
        let series = self.series.entry(name.to_owned()).or_default();
        series.shift_remove(values);
        series.insert(values.to_vec(), now);
        match limits.ttl_seconds {
            Some(ttl) if self.cleanup_due(name, now, limits.cleanup_interval_seconds) => {
                self.expired(name, now, ttl)
            }
            _ => Vec::new(),
        }
    }

    pub fn over_cap(&self, name: &str, limits: &SeriesLimits) -> Option<Vec<V>> {
        let max = limits.max_series.filter(|max| *max > 0)?;
        let series = self.series.get(name)?;
        if i64::try_from(series.len()).unwrap_or(i64::MAX) <= max {
            return None;
        }
        series.keys().next().cloned()
    }

    pub fn forget(&mut self, name: &str, values: &[V]) {
        if let Some(series) = self.series.get_mut(name) {
            series.shift_remove(values);
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn limits(max: Option<i64>, ttl: Option<f64>, interval: Option<f64>) -> SeriesLimits {
        SeriesLimits {
            max_series: max,
            ttl_seconds: ttl,
            cleanup_interval_seconds: interval,
        }
    }

    #[rstest]
    fn a_full_tracker_admits_only_the_label_sets_it_already_holds() {
        let mut tracker = Tracker::<String>::default();
        let cap = limits(Some(1), None, None);
        assert!(tracker.admit("m", &["a".into()], 0.0, &cap).admitted);
        assert!(!tracker.admit("m", &["b".into()], 1.0, &cap).admitted);
        assert!(tracker.admit("m", &["a".into()], 2.0, &cap).admitted);
    }

    #[rstest]
    fn an_idle_label_set_expires_once_the_cleanup_interval_passed() {
        let mut tracker = Tracker::<String>::default();
        let ttl = limits(None, Some(10.0), Some(60.0));
        let _ = tracker.admit("m", &["a".into()], 0.0, &ttl);
        assert!(
            tracker
                .admit("m", &["b".into()], 30.0, &ttl)
                .removed
                .is_empty()
        );
        let removed = tracker.admit("m", &["b".into()], 61.0, &ttl).removed;
        assert_eq!(removed, vec![vec!["a".to_owned()], vec!["b".to_owned()]]);
    }

    #[rstest]
    fn the_least_recently_seen_label_set_is_over_the_cap() {
        let mut tracker = Tracker::<Option<String>>::default();
        let cap = limits(Some(1), None, None);
        let _ = tracker.track_due("m", &[Some("a".into())], 0.0, &cap);
        let _ = tracker.track_due("m", &[None], 1.0, &cap);
        assert_eq!(
            tracker.over_cap("m", &cap),
            Some(vec![Some("a".to_owned())])
        );
    }
}
