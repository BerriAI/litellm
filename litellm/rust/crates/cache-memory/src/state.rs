use std::{
    collections::{BTreeSet, HashMap},
    time::Duration,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Deadline {
    UnlessLive(Duration),
    Always(Duration),
}

struct Entry<V> {
    value: V,
    expires_at: Duration,
}

/// Entries with an exact index of their deadlines: `by_expiry` holds one record per entry, so it
/// never outgrows `entries`. Every read and write first drops what has expired by `now`.
pub(crate) struct State<V> {
    entries: HashMap<String, Entry<V>>,
    by_expiry: BTreeSet<(Duration, String)>,
}

impl<V> State<V> {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            by_expiry: BTreeSet::new(),
        }
    }

    pub(crate) fn get(&mut self, key: &str, now: Duration) -> Option<&V> {
        self.drop_expired(now);
        self.entries.get(key).map(|entry| &entry.value)
    }

    pub(crate) fn expires_at(&mut self, key: &str, now: Duration) -> Option<Duration> {
        self.drop_expired(now);
        self.entries.get(key).map(|entry| entry.expires_at)
    }

    pub(crate) fn len(&mut self, now: Duration) -> usize {
        self.drop_expired(now);
        self.entries.len()
    }

    pub(crate) fn live(&mut self, now: Duration) -> impl Iterator<Item = (&str, &V)> {
        self.drop_expired(now);
        self.entries
            .iter()
            .map(|(key, entry)| (key.as_str(), &entry.value))
    }

    pub(crate) fn keys_by_expiry(&mut self, now: Duration) -> impl Iterator<Item = &str> {
        self.drop_expired(now);
        self.by_expiry.iter().map(|(_, key)| key.as_str())
    }

    /// A key the cache does not hold evicts the entries soonest to expire until it fits `capacity`.
    pub(crate) fn write(
        &mut self,
        key: String,
        value: V,
        deadline: Deadline,
        capacity: usize,
        now: Duration,
    ) {
        self.drop_expired(now);
        let live = self.take(&key).map(|entry| entry.expires_at);
        let expires_at = match (deadline, live) {
            (Deadline::UnlessLive(_), Some(live)) => live,
            (Deadline::UnlessLive(at) | Deadline::Always(at), _) => at,
        };
        while self.entries.len() >= capacity {
            let Some((_, evicted)) = self.by_expiry.pop_first() else {
                break;
            };
            self.entries.remove(&evicted);
        }
        self.by_expiry.insert((expires_at, key.clone()));
        self.entries.insert(key, Entry { value, expires_at });
    }

    pub(crate) fn remove(&mut self, key: &str) {
        self.take(key);
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.by_expiry.clear();
    }

    fn take(&mut self, key: &str) -> Option<Entry<V>> {
        let entry = self.entries.remove(key)?;
        self.by_expiry.remove(&(entry.expires_at, key.to_owned()));
        Some(entry)
    }

    fn drop_expired(&mut self, now: Duration) {
        while self
            .by_expiry
            .first()
            .is_some_and(|(expires_at, _)| *expires_at <= now)
        {
            if let Some((_, key)) = self.by_expiry.pop_first() {
                self.entries.remove(&key);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rstest::{fixture, rstest};

    use super::{Deadline, State};

    const START: Duration = Duration::ZERO;

    fn secs(seconds: u64) -> Duration {
        Duration::from_secs(seconds)
    }

    fn put(state: &mut State<&'static str>, key: &str, value: &'static str, expires_at: u64) {
        state.write(
            key.to_owned(),
            value,
            Deadline::UnlessLive(secs(expires_at)),
            2,
            START,
        );
    }

    /// Full at capacity 2: `early` expires at 10 s and `late` at 20 s.
    #[fixture]
    fn full() -> State<&'static str> {
        let mut state = State::new();
        put(&mut state, "early", "a", 10);
        put(&mut state, "late", "b", 20);
        state
    }

    #[rstest]
    #[case::before(9, Some("a"))]
    #[case::at_the_deadline(10, None)]
    #[case::after(11, None)]
    fn an_entry_is_held_until_its_deadline(
        mut full: State<&'static str>,
        #[case] now: u64,
        #[case] expected: Option<&'static str>,
    ) {
        assert_eq!(full.get("early", secs(now)).copied(), expected);
    }

    #[rstest]
    #[case::keeps_its_deadline(Deadline::UnlessLive(secs(50)), 10)]
    #[case::takes_the_new_deadline(Deadline::Always(secs(50)), 50)]
    fn a_write_to_a_live_key_replaces_the_value_and(
        mut full: State<&'static str>,
        #[case] deadline: Deadline,
        #[case] expected: u64,
    ) {
        full.write("early".to_owned(), "new", deadline, 2, secs(5));

        assert_eq!(full.get("early", secs(5)).copied(), Some("new"));
        assert_eq!(full.expires_at("early", secs(5)), Some(secs(expected)));
    }

    #[rstest]
    fn an_expired_key_takes_the_deadline_of_the_next_write(mut full: State<&'static str>) {
        full.write(
            "early".to_owned(),
            "new",
            Deadline::UnlessLive(secs(30)),
            2,
            secs(10),
        );

        assert_eq!(full.expires_at("early", secs(10)), Some(secs(30)));
    }

    #[rstest]
    fn a_new_key_at_capacity_evicts_the_entry_soonest_to_expire(mut full: State<&'static str>) {
        put(&mut full, "new", "c", 5);

        assert_eq!(
            full.keys_by_expiry(START).collect::<Vec<_>>(),
            ["new", "late"]
        );
    }

    #[rstest]
    fn entries_with_one_deadline_are_evicted_in_key_order() {
        let mut state = State::new();
        put(&mut state, "b", "2", 10);
        put(&mut state, "a", "1", 10);

        put(&mut state, "c", "3", 10);

        assert_eq!(state.keys_by_expiry(START).collect::<Vec<_>>(), ["b", "c"]);
    }

    #[rstest]
    fn an_expired_entry_makes_room_before_a_live_one_is_evicted(mut full: State<&'static str>) {
        full.write(
            "new".to_owned(),
            "c",
            Deadline::UnlessLive(secs(15)),
            2,
            secs(10),
        );

        assert_eq!(
            full.keys_by_expiry(secs(10)).collect::<Vec<_>>(),
            ["new", "late"]
        );
    }

    #[rstest]
    #[case::keeping_its_deadline(Deadline::UnlessLive(secs(50)))]
    #[case::taking_a_new_deadline(Deadline::Always(secs(50)))]
    fn a_write_to_a_held_key_at_capacity_evicts_nothing(
        mut full: State<&'static str>,
        #[case] deadline: Deadline,
    ) {
        full.write("late".to_owned(), "new", deadline, 2, START);

        assert_eq!(full.get("early", START).copied(), Some("a"));
        assert_eq!(full.len(START), 2);
    }

    #[rstest]
    fn a_key_given_a_later_deadline_is_evicted_by_that_deadline(mut full: State<&'static str>) {
        full.write(
            "early".to_owned(),
            "a",
            Deadline::Always(secs(30)),
            2,
            START,
        );

        put(&mut full, "new", "c", 40);

        assert_eq!(
            full.keys_by_expiry(START).collect::<Vec<_>>(),
            ["early", "new"]
        );
    }

    #[rstest]
    fn a_removed_key_written_again_is_indexed_once(mut full: State<&'static str>) {
        full.remove("early");
        put(&mut full, "early", "a", 30);
        full.remove("early");
        put(&mut full, "early", "a", 40);

        assert_eq!(
            full.keys_by_expiry(START).collect::<Vec<_>>(),
            ["late", "early"]
        );
        assert_eq!(full.len(START), 2);
    }

    #[rstest]
    #[case::a_new_key("new", &["new"])]
    #[case::a_held_key("early", &["early"])]
    fn a_smaller_capacity_applies_on_the_next_write_of(
        mut full: State<&'static str>,
        #[case] key: &str,
        #[case] kept: &[&str],
    ) {
        full.write(key.to_owned(), "c", Deadline::Always(secs(40)), 1, START);

        assert_eq!(full.keys_by_expiry(START).collect::<Vec<_>>(), kept);
    }

    #[rstest]
    fn an_expired_entry_is_not_listed(mut full: State<&'static str>) {
        assert_eq!(full.live(secs(10)).collect::<Vec<_>>(), [("late", &"b")]);
    }

    #[rstest]
    fn an_expired_entry_is_not_counted(mut full: State<&'static str>) {
        assert_eq!(full.len(secs(10)), 1);
    }

    #[rstest]
    fn an_expired_entry_has_no_deadline(mut full: State<&'static str>) {
        assert_eq!(full.expires_at("early", secs(10)), None);
    }

    #[rstest]
    fn an_expired_entry_is_not_among_the_keys_by_expiry(mut full: State<&'static str>) {
        assert_eq!(full.keys_by_expiry(secs(10)).collect::<Vec<_>>(), ["late"]);
    }

    #[rstest]
    fn clear_drops_every_entry_and_its_deadline(mut full: State<&'static str>) {
        full.clear();

        assert_eq!(full.len(START), 0);
        assert_eq!(full.keys_by_expiry(START).count(), 0);
    }

    #[rstest]
    fn a_deadline_at_the_end_of_time_is_never_reached() {
        let mut state = State::new();
        state.write(
            "forever".to_owned(),
            "a",
            Deadline::Always(Duration::MAX),
            2,
            START,
        );

        assert_eq!(state.get("forever", secs(u64::MAX)).copied(), Some("a"));
    }
}
