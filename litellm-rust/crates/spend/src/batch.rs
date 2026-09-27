use std::collections::{BTreeMap, btree_map::Entry};

use crate::{Cost, DailyKey, DailyTally, EntityKey, Key, Tally, WindowKey};

pub type Totals = Batch<EntityKey, Cost>;
pub type DailyRollups = Batch<DailyKey, DailyTally>;
pub type WindowSpend = Batch<WindowKey, Cost>;

#[derive(Clone, Debug, PartialEq)]
pub struct Batch<K, V>(BTreeMap<K, V>);

impl<K, V> Default for Batch<K, V> {
    fn default() -> Self {
        Self(BTreeMap::new())
    }
}

impl<K: Key, V: Tally> Batch<K, V> {
    pub fn from_entries(entries: impl IntoIterator<Item = (K, V)>) -> Self {
        let mut tallies = BTreeMap::new();
        for (key, tally) in entries {
            add(&mut tallies, key, tally);
        }
        Self(tallies)
    }

    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        let Self(mut tallies) = self;
        for (key, tally) in other.0 {
            add(&mut tallies, key, tally);
        }
        Self(tallies)
    }

    #[must_use]
    pub fn split_at(self, max: usize) -> (Self, Self) {
        let Self(mut head) = self;
        let Some(boundary) = head.keys().nth(max).cloned() else {
            return (Self(head), Self::default());
        };
        let tail = head.split_off(&boundary);
        (Self(head), Self(tail))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.0.get(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.0.iter()
    }
}

fn add<K: Key, V: Tally>(tallies: &mut BTreeMap<K, V>, key: K, tally: V) {
    match tallies.entry(key) {
        Entry::Vacant(slot) => {
            slot.insert(tally);
        }
        Entry::Occupied(mut slot) => {
            let so_far = std::mem::take(slot.get_mut());
            slot.insert(so_far.merge(tally));
        }
    }
}
