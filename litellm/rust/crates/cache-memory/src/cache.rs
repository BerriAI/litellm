use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use litellm_cache::Error;
use parking_lot::Mutex;

use crate::state::{Deadline, State};

const DEFAULT_MAX_SIZE_IN_MEMORY: usize = 200;
const DEFAULT_TTL: Duration = Duration::from_secs(600);

type ValueMeasure<V> = Arc<dyn Fn(&V) -> Result<usize, Error> + Send + Sync>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheWrite {
    Stored,
    Disabled,
    TooLarge,
}

/// The TTL a write gives its key, `None` being the cache's default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expiry {
    /// Only a key the cache does not hold takes the TTL, so a counter keeps its window.
    KeepExisting(Option<Duration>),
    Refresh(Option<Duration>),
}

pub struct InMemoryCache<V: Clone> {
    state: Mutex<State<V>>,
    max_size_in_memory: AtomicUsize,
    default_ttl: Duration,
    max_entry_bytes: Option<usize>,
    measure_value: Option<ValueMeasure<V>>,
    now: Arc<dyn Fn() -> Duration + Send + Sync>,
}

impl<V: Clone> Default for InMemoryCache<V> {
    fn default() -> Self {
        Self::new(None, None)
    }
}

impl<V: Clone> InMemoryCache<V> {
    pub fn new(max_size_in_memory: Option<usize>, default_ttl: Option<Duration>) -> Self {
        Self::with_clock(max_size_in_memory, default_ttl, || {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
        })
    }

    pub fn with_clock(
        max_size_in_memory: Option<usize>,
        default_ttl: Option<Duration>,
        now: impl Fn() -> Duration + Send + Sync + 'static,
    ) -> Self {
        Self::with_clock_and_size_measurement(max_size_in_memory, default_ttl, None, None, now)
    }

    pub fn with_clock_and_size_measurement(
        max_size_in_memory: Option<usize>,
        default_ttl: Option<Duration>,
        max_entry_bytes: Option<usize>,
        measure_value: Option<ValueMeasure<V>>,
        now: impl Fn() -> Duration + Send + Sync + 'static,
    ) -> Self {
        Self {
            state: Mutex::new(State::new()),
            max_size_in_memory: AtomicUsize::new(
                max_size_in_memory.unwrap_or(DEFAULT_MAX_SIZE_IN_MEMORY),
            ),
            default_ttl: default_ttl
                .filter(|ttl| !ttl.is_zero())
                .unwrap_or(DEFAULT_TTL),
            max_entry_bytes,
            measure_value,
            now: Arc::new(now),
        }
    }

    pub fn set_cache(
        &self,
        key: impl Into<String>,
        value: V,
        expiry: Expiry,
    ) -> Result<CacheWrite, Error> {
        let now = (self.now)();
        self.write(&mut self.state.lock(), key.into(), value, expiry, now)
    }

    pub fn get_cache(&self, key: &str) -> Result<Option<V>, Error> {
        let now = (self.now)();
        Ok(self.state.lock().get(key, now).cloned())
    }

    /// Reads `key` and writes back under one lock. `update` answers the value to write, or `None`
    /// to leave the entry as it is, next to what the caller gets back.
    pub fn update<T>(
        &self,
        key: &str,
        expiry: Expiry,
        update: impl FnOnce(Option<&V>) -> Result<(Option<V>, T), Error>,
    ) -> Result<T, Error> {
        let now = (self.now)();
        let mut state = self.state.lock();
        let (value, output) = update(state.get(key, now))?;
        if let Some(value) = value {
            self.write(&mut state, key.to_owned(), value, expiry, now)?;
        }
        Ok(output)
    }

    /// Always `true` without a limit and a measure, since typed values have no generic size.
    pub fn check_value_size(&self, value: &V) -> Result<bool, Error> {
        match (self.max_entry_bytes, &self.measure_value) {
            (Some(limit), Some(measure)) => Ok(measure(value)? <= limit),
            _ => Ok(true),
        }
    }

    pub fn len(&self) -> Result<usize, Error> {
        let now = (self.now)();
        Ok(self.state.lock().len(now))
    }

    pub fn is_empty(&self) -> Result<bool, Error> {
        Ok(self.len()? == 0)
    }

    pub fn max_size_in_memory(&self) -> usize {
        self.max_size_in_memory.load(Ordering::Relaxed)
    }

    /// `None` restores the default. Entries already held stay until the next write trims to the
    /// new size.
    pub fn update_max_size_in_memory(&self, max_size_in_memory: Option<usize>) {
        self.max_size_in_memory.store(
            max_size_in_memory.unwrap_or(DEFAULT_MAX_SIZE_IN_MEMORY),
            Ordering::Relaxed,
        );
    }

    pub fn default_ttl(&self) -> Duration {
        self.default_ttl
    }

    pub fn expires_at(&self, key: &str) -> Result<Option<Duration>, Error> {
        let now = (self.now)();
        Ok(self.state.lock().expires_at(key, now))
    }

    pub async fn async_get_ttl(&self, key: &str) -> Result<Option<Duration>, Error> {
        self.expires_at(key)
    }

    pub async fn async_get_oldest_n_keys(&self, count: usize) -> Result<Vec<String>, Error> {
        let now = (self.now)();
        Ok(self
            .state
            .lock()
            .keys_by_expiry(now)
            .take(count)
            .map(str::to_owned)
            .collect())
    }

    pub fn entries(&self) -> Result<Vec<(String, V)>, Error> {
        let now = (self.now)();
        Ok(self
            .state
            .lock()
            .live(now)
            .map(|(key, value)| (key.to_owned(), value.clone()))
            .collect())
    }

    pub fn delete_cache(&self, key: &str) -> Result<(), Error> {
        self.state.lock().remove(key);
        Ok(())
    }

    pub fn flush_cache(&self) -> Result<(), Error> {
        self.state.lock().clear();
        Ok(())
    }

    fn write(
        &self,
        state: &mut State<V>,
        key: String,
        value: V,
        expiry: Expiry,
        now: Duration,
    ) -> Result<CacheWrite, Error> {
        let capacity = self.max_size_in_memory();
        if capacity == 0 {
            return Ok(CacheWrite::Disabled);
        }
        if !self.check_value_size(&value)? {
            return Ok(CacheWrite::TooLarge);
        }
        let expires_at =
            |ttl: Option<Duration>| now.saturating_add(ttl.unwrap_or(self.default_ttl));
        let deadline = match expiry {
            Expiry::KeepExisting(ttl) => Deadline::UnlessLive(expires_at(ttl)),
            Expiry::Refresh(ttl) => Deadline::Always(expires_at(ttl)),
        };
        state.write(key, value, deadline, capacity, now);
        Ok(CacheWrite::Stored)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::Arc,
        time::Duration,
    };

    use litellm_cache::Error;
    use rstest::rstest;

    use super::{CacheWrite, Expiry, InMemoryCache};
    use crate::test_support::{Clock, at, cache, clock, secs};

    const DEFAULT: Expiry = Expiry::KeepExisting(None);

    fn keep(seconds: u64) -> Expiry {
        Expiry::KeepExisting(secs(seconds))
    }

    fn measured(capacity: usize) -> InMemoryCache<String> {
        InMemoryCache::with_clock_and_size_measurement(
            Some(capacity),
            secs(60),
            Some(4),
            Some(Arc::new(|value: &String| {
                if value.is_empty() {
                    return Err(Error::InvalidEntry(None));
                }
                Ok(value.len())
            })),
            || Duration::from_secs(100),
        )
    }

    fn count(current: Option<&u64>) -> Result<(Option<u64>, u64), Error> {
        let next = current.copied().unwrap_or_default() + 1;
        Ok((Some(next), next))
    }

    #[rstest]
    #[case::unset(None, secs(600))]
    #[case::zero(Some(Duration::ZERO), secs(600))]
    #[case::explicit(secs(5), secs(5))]
    fn a_write_without_a_ttl_takes_the_default(
        #[case] default_ttl: Option<Duration>,
        #[case] expected: Option<Duration>,
    ) {
        let cache = InMemoryCache::<String>::with_clock(None, default_ttl, || Duration::ZERO);

        cache.set_cache("key", "value".into(), DEFAULT).unwrap();

        assert_eq!(cache.expires_at("key"), Ok(expected));
        assert_eq!(cache.max_size_in_memory(), 200);
    }

    #[rstest]
    #[case::before(109, Some("value"))]
    #[case::at_the_deadline(110, None)]
    #[case::after(111, None)]
    fn an_entry_is_readable_until_its_deadline(
        clock: Clock,
        #[case] now: u64,
        #[case] expected: Option<&str>,
    ) {
        let cache = cache::<String>(&clock, 4);
        cache.set_cache("key", "value".into(), keep(10)).unwrap();

        at(&clock, now);

        assert_eq!(cache.get_cache("key"), Ok(expected.map(str::to_owned)));
    }

    #[rstest]
    #[case::keep_existing(keep(10), secs(160))]
    #[case::refresh(Expiry::Refresh(secs(10)), secs(115))]
    fn a_write_to_a_live_key_keeps_its_deadline_unless_it_refreshes(
        clock: Clock,
        #[case] expiry: Expiry,
        #[case] expected: Option<Duration>,
    ) {
        let cache = cache::<String>(&clock, 4);
        cache.set_cache("key", "first".into(), DEFAULT).unwrap();
        at(&clock, 105);

        cache.set_cache("key", "second".into(), expiry).unwrap();

        assert_eq!(cache.get_cache("key"), Ok(Some("second".into())));
        assert_eq!(cache.expires_at("key"), Ok(expected));
    }

    #[rstest]
    fn an_expired_key_takes_the_ttl_of_the_next_write(clock: Clock) {
        let cache = cache::<String>(&clock, 4);
        cache.set_cache("key", "first".into(), keep(10)).unwrap();
        at(&clock, 110);

        cache.set_cache("key", "second".into(), keep(10)).unwrap();

        assert_eq!(cache.expires_at("key"), Ok(secs(120)));
    }

    #[rstest]
    #[case::keep_existing(Expiry::KeepExisting(Some(Duration::MAX)))]
    #[case::refresh(Expiry::Refresh(Some(Duration::MAX)))]
    fn a_ttl_past_the_end_of_time_never_expires(clock: Clock, #[case] forever: Expiry) {
        let cache = cache::<String>(&clock, 3);

        cache.set_cache("forever", "a".into(), forever).unwrap();
        at(&clock, 1_000_000_000);
        cache.set_cache("after", "b".into(), keep(1)).unwrap();

        assert_eq!(cache.get_cache("forever"), Ok(Some("a".into())));
        assert_eq!(cache.get_cache("after"), Ok(Some("b".into())));
    }

    #[rstest]
    fn a_panic_inside_an_update_leaves_the_cache_usable(clock: Clock) {
        let cache = cache::<String>(&clock, 4);
        cache.set_cache("held", "a".into(), DEFAULT).unwrap();

        let panicked = catch_unwind(AssertUnwindSafe(|| {
            cache.update::<()>("held", DEFAULT, |_| panic!("the caller's update panics"))
        }));

        assert!(panicked.is_err());
        assert_eq!(
            cache.set_cache("next", "b".into(), DEFAULT),
            Ok(CacheWrite::Stored)
        );
        assert_eq!(cache.get_cache("held"), Ok(Some("a".into())));
    }

    #[rstest]
    fn a_full_cache_admits_a_new_key_by_evicting_the_soonest_to_expire(clock: Clock) {
        let cache = cache::<String>(&clock, 2);
        cache.set_cache("early", "a".into(), keep(10)).unwrap();
        cache.set_cache("late", "b".into(), keep(20)).unwrap();

        cache.set_cache("new", "c".into(), keep(5)).unwrap();

        assert_eq!(cache.get_cache("early"), Ok(None));
        assert_eq!(cache.get_cache("late"), Ok(Some("b".into())));
        assert_eq!(cache.get_cache("new"), Ok(Some("c".into())));
    }

    #[rstest]
    #[tokio::test]
    async fn a_key_deleted_and_set_again_is_tracked_once(clock: Clock) {
        let cache = cache::<String>(&clock, 4);
        cache.set_cache("older", "pin".into(), keep(3600)).unwrap();
        at(&clock, 101);
        cache.set_cache("pinned", "pin".into(), keep(3600)).unwrap();
        at(&clock, 102);
        cache.delete_cache("pinned").unwrap();
        cache.set_cache("pinned", "pin".into(), keep(3600)).unwrap();
        at(&clock, 103);
        cache
            .set_cache("pinned", "pin".into(), Expiry::Refresh(secs(3600)))
            .unwrap();

        assert_eq!(
            cache.async_get_oldest_n_keys(10).await.unwrap(),
            ["older", "pinned"]
        );
        assert_eq!(cache.expires_at("pinned"), Ok(secs(3703)));
    }

    #[rstest]
    fn a_smaller_capacity_applies_on_the_next_write(clock: Clock) {
        let cache = cache::<String>(&clock, 3);
        cache.set_cache("a", "1".into(), keep(10)).unwrap();
        cache.set_cache("b", "2".into(), keep(20)).unwrap();
        cache.set_cache("c", "3".into(), keep(30)).unwrap();

        cache.update_max_size_in_memory(Some(1));
        let held = cache.len();
        cache.set_cache("d", "4".into(), keep(40)).unwrap();

        assert_eq!(held, Ok(3));
        assert_eq!(cache.entries(), Ok(vec![("d".to_owned(), "4".to_owned())]));
    }

    #[rstest]
    fn an_unset_capacity_restores_the_default(clock: Clock) {
        let cache = cache::<String>(&clock, 3);

        cache.update_max_size_in_memory(None);

        assert_eq!(cache.max_size_in_memory(), 200);
    }

    #[rstest]
    fn a_cache_without_capacity_stores_nothing(clock: Clock) {
        let cache = cache::<u64>(&clock, 0);

        let written = cache.set_cache("key", 1, DEFAULT);
        let counted = cache.update("key", DEFAULT, count);

        assert_eq!(written, Ok(CacheWrite::Disabled));
        assert_eq!(counted, Ok(1));
        assert_eq!(cache.is_empty(), Ok(true));
    }

    #[rstest]
    #[case::fits("ok", Ok(true))]
    #[case::at_limit("four", Ok(true))]
    #[case::too_large("oversized", Ok(false))]
    #[case::measure_error("", Err(Error::InvalidEntry(None)))]
    fn check_value_size_applies_the_entry_limit(
        #[case] value: &str,
        #[case] expected: Result<bool, Error>,
    ) {
        assert_eq!(measured(2).check_value_size(&value.to_owned()), expected);
    }

    #[rstest]
    fn values_are_unbounded_without_a_measure() {
        let cache = InMemoryCache::<String>::default();

        assert_eq!(cache.check_value_size(&"x".repeat(1 << 20)), Ok(true));
    }

    #[rstest]
    #[case::too_large("oversized", Ok(CacheWrite::TooLarge))]
    #[case::rejected_by_the_measure("", Err(Error::InvalidEntry(None)))]
    fn a_value_that_does_not_fit_is_not_stored_and_keeps_the_held_one(
        #[case] value: &str,
        #[case] expected: Result<CacheWrite, Error>,
    ) {
        let cache = measured(2);
        cache.set_cache("key", "ok".into(), DEFAULT).unwrap();

        let written = cache.set_cache("key", value.to_owned(), DEFAULT);

        assert_eq!(written, expected);
        assert_eq!(cache.get_cache("key"), Ok(Some("ok".into())));
    }

    #[rstest]
    fn an_update_writes_what_it_answers_and_keeps_the_window(clock: Clock) {
        let cache = cache::<u64>(&clock, 4);

        let first = cache.update("count", keep(10), count);
        at(&clock, 105);
        let second = cache.update("count", keep(10), count);

        assert_eq!((first, second), (Ok(1), Ok(2)));
        assert_eq!(cache.get_cache("count"), Ok(Some(2)));
        assert_eq!(cache.expires_at("count"), Ok(secs(110)));
    }

    #[rstest]
    fn an_update_that_answers_no_value_leaves_the_entry_as_it_is(clock: Clock) {
        let cache = cache::<u64>(&clock, 4);
        cache.set_cache("count", 5, keep(10)).unwrap();
        at(&clock, 105);

        let read = cache.update("count", Expiry::Refresh(secs(50)), |current| {
            Ok((None, current.copied()))
        });

        assert_eq!(read, Ok(Some(5)));
        assert_eq!(cache.get_cache("count"), Ok(Some(5)));
        assert_eq!(cache.expires_at("count"), Ok(secs(110)));
    }

    #[rstest]
    fn an_update_that_fails_writes_nothing(clock: Clock) {
        let cache = cache::<u64>(&clock, 4);

        let failed = cache.update::<()>("count", DEFAULT, |_| Err(Error::InvalidEntry(None)));

        assert_eq!(failed, Err(Error::InvalidEntry(None)));
        assert_eq!(cache.get_cache("count"), Ok(None));
    }

    #[rstest]
    fn an_update_fails_when_its_value_cannot_be_measured() {
        let updated = measured(2).update("key", DEFAULT, |_| Ok((Some(String::new()), ())));

        assert_eq!(updated, Err(Error::InvalidEntry(None)));
    }

    #[rstest]
    fn only_live_entries_are_counted_and_listed(clock: Clock) {
        let cache = cache::<String>(&clock, 4);
        cache.set_cache("short", "a".into(), keep(1)).unwrap();
        cache.set_cache("long", "b".into(), keep(50)).unwrap();

        at(&clock, 101);

        assert_eq!(cache.len(), Ok(1));
        assert_eq!(cache.is_empty(), Ok(false));
        assert_eq!(
            cache.entries(),
            Ok(vec![("long".to_owned(), "b".to_owned())])
        );
        assert_eq!(cache.expires_at("short"), Ok(None));
    }

    #[rstest]
    #[case::none(0, &[])]
    #[case::one(1, &["first"])]
    #[case::more_than_held(10, &["first", "later", "latest"])]
    #[tokio::test]
    async fn the_oldest_keys_are_the_soonest_to_expire(
        clock: Clock,
        #[case] count: usize,
        #[case] expected: &[&str],
    ) {
        let cache = cache::<String>(&clock, 3);
        cache.set_cache("later", "2".into(), keep(20)).unwrap();
        cache.set_cache("first", "1".into(), keep(10)).unwrap();
        cache.set_cache("latest", "3".into(), keep(30)).unwrap();

        assert_eq!(
            cache.async_get_oldest_n_keys(count).await.unwrap(),
            expected
        );
    }

    #[rstest]
    #[tokio::test]
    async fn the_ttl_of_a_key_is_its_deadline(clock: Clock) {
        let cache = cache::<String>(&clock, 3);
        cache.set_cache("key", "1".into(), keep(10)).unwrap();

        assert_eq!(cache.async_get_ttl("key").await, Ok(secs(110)));
        assert_eq!(cache.async_get_ttl("missing").await, Ok(None));
    }

    #[rstest]
    fn delete_drops_one_key_and_flush_drops_the_rest(clock: Clock) {
        let cache = cache::<String>(&clock, 4);
        cache.set_cache("a", "1".into(), DEFAULT).unwrap();
        cache.set_cache("b", "2".into(), DEFAULT).unwrap();
        cache.set_cache("c", "3".into(), DEFAULT).unwrap();

        cache.delete_cache("a").unwrap();
        let after_delete = (cache.get_cache("a"), cache.len());
        cache.flush_cache().unwrap();

        assert_eq!(after_delete, (Ok(None), Ok(2)));
        assert_eq!(cache.is_empty(), Ok(true));
    }
}
