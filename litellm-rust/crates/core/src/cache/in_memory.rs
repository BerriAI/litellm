use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

pub struct InMemoryCache<K, V> {
    capacity: usize,
    entries: HashMap<K, V>,
    order: VecDeque<K>,
}

impl<K, V> InMemoryCache<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        self.entries.get(key).cloned()
    }

    pub fn get_or_insert(&mut self, key: K, value: V) -> V {
        if let Some(existing) = self.entries.get(&key) {
            return existing.clone();
        }
        while self.entries.len() >= self.capacity {
            let Some(evicted) = self.order.pop_front() else {
                break;
            };
            self.entries.remove(&evicted);
        }
        self.order.push_back(key.clone());
        self.entries.insert(key, value.clone());
        value
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::InMemoryCache;

    #[test]
    fn evicts_oldest_entry_at_capacity() {
        let mut cache = InMemoryCache::new(2);
        cache.get_or_insert(1, "one");
        cache.get_or_insert(2, "two");
        cache.get_or_insert(3, "three");

        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&2), Some("two"));
        assert_eq!(cache.get(&3), Some("three"));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn reuses_existing_entry_without_replacement() {
        let mut cache = InMemoryCache::new(2);

        assert!(cache.is_empty());
        assert_eq!(cache.get_or_insert(1, "first"), "first");
        assert_eq!(cache.get_or_insert(1, "replacement"), "first");
        assert_eq!(cache.get(&1), Some("first"));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn zero_capacity_still_retains_one_entry() {
        let mut cache = InMemoryCache::new(0);
        cache.get_or_insert(1, "one");
        cache.get_or_insert(2, "two");

        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&2), Some("two"));
        assert_eq!(cache.len(), 1);
    }
}
