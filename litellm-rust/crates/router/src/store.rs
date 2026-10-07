//! Router runtime state with the keys, values and TTLs Python writes, so both backends can
//! share one Redis. Memory semantics follow `InMemoryCache`: a write keeps an unexpired TTL.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, PoisonError},
    time::{SystemTime, UNIX_EPOCH},
};

use redis::{AsyncCommands, aio::ConnectionManager};
use tokio::sync::OnceCell;

use crate::pyrepr::{self, Literal, PyNumber, str_repr};

pub trait Clock: Send + Sync {
    /// Seconds since the epoch, as `time.time()`.
    fn now(&self) -> f64;
}

pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0.0, |elapsed| elapsed.as_secs_f64())
    }
}

/// `CooldownCacheValue`.
#[derive(Clone, Debug, PartialEq)]
pub struct Cooldown {
    pub exception_received: String,
    pub status_code: String,
    pub timestamp: f64,
    pub cooldown_time: PyNumber,
}

impl Cooldown {
    pub fn key(model_id: &str) -> String {
        format!("deployment:{model_id}:cooldown")
    }

    fn remaining(&self, now: f64) -> f64 {
        self.timestamp + self.cooldown_time.as_f64() - now
    }

    /// `str(value)` of the dict Python stores.
    pub fn repr(&self) -> String {
        format!(
            "{{'exception_received': {}, 'status_code': {}, 'timestamp': {}, 'cooldown_time': {}}}",
            str_repr(&self.exception_received),
            str_repr(&self.status_code),
            pyrepr::float_repr(self.timestamp),
            self.cooldown_time.repr()
        )
    }

    fn from_literal(literal: Literal) -> Option<Self> {
        let Literal::Dict(entries) = literal else {
            return None;
        };
        let field = |name: &str| {
            entries
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value)
        };
        let text = |name: &str| match field(name)? {
            Literal::Str(text) => Some(text.clone()),
            _ => None,
        };
        let number = |name: &str| match field(name)? {
            Literal::Number(number) => Some(*number),
            _ => None,
        };
        Some(Self {
            exception_received: text("exception_received")?,
            status_code: text("status_code")?,
            timestamp: number("timestamp")?.as_f64(),
            cooldown_time: number("cooldown_time")?,
        })
    }
}

struct Memory<V> {
    entries: HashMap<String, (V, f64)>,
}

impl<V: Clone> Memory<V> {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    fn get(&mut self, key: &str, now: f64) -> Option<V> {
        match self.entries.get(key) {
            Some((_, expires)) if *expires <= now => {
                self.entries.remove(key);
                None
            }
            Some((value, _)) => Some(value.clone()),
            None => None,
        }
    }

    fn set(&mut self, key: &str, value: V, ttl: f64, now: f64) {
        let expires = match self.entries.get(key) {
            Some((_, expires)) if *expires >= now => *expires,
            _ => now + ttl,
        };
        self.entries.insert(key.to_owned(), (value, expires));
    }

    fn replace(&mut self, key: &str, value: V, ttl: f64, now: f64) {
        self.entries.insert(key.to_owned(), (value, now + ttl));
    }

    fn remove(&mut self, key: &str) {
        self.entries.remove(key);
    }
}

#[derive(Clone, Copy)]
enum Counter {
    Int(i64),
    Float(f64),
}

impl Counter {
    fn as_f64(self) -> f64 {
        match self {
            Self::Int(value) => value as f64,
            Self::Float(value) => value,
        }
    }
}

/// Where Redis lives, if anywhere. The connection opens on first use inside the call's runtime.
pub struct RedisTarget {
    client: redis::Client,
    connection: OnceCell<ConnectionManager>,
}

impl RedisTarget {
    pub fn open(url: &str) -> redis::RedisResult<Self> {
        Ok(Self {
            client: redis::Client::open(url)?,
            connection: OnceCell::new(),
        })
    }

    async fn connection(&self) -> redis::RedisResult<ConnectionManager> {
        self.connection
            .get_or_try_init(|| ConnectionManager::new(self.client.clone()))
            .await
            .cloned()
    }
}

/// Cooldown entries live apart from the counters, as Python keeps them in their own
/// `DualCache`, so unrelated keys never evict a cooldown and Redis is re-read on its own
/// interval.
pub struct Store {
    clock: Arc<dyn Clock>,
    counters: Mutex<Memory<Counter>>,
    cooldowns: Mutex<Memory<Cooldown>>,
    redis_reads: Mutex<HashMap<String, f64>>,
    redis: Option<RedisTarget>,
    redis_read_interval: f64,
}

const CORRECTED_COOLDOWN_TTL_CAP: f64 = 60.0;
const DEFAULT_IN_MEMORY_TTL: f64 = 600.0;

impl Store {
    pub fn new(
        clock: Arc<dyn Clock>,
        redis: Option<RedisTarget>,
        redis_read_interval: f64,
    ) -> Self {
        Self {
            clock,
            counters: Mutex::new(Memory::new()),
            cooldowns: Mutex::new(Memory::new()),
            redis_reads: Mutex::new(HashMap::new()),
            redis,
            redis_read_interval,
        }
    }

    pub fn now(&self) -> f64 {
        self.clock.now()
    }

    /// `cache.increment_cache(local_only=True, ...)`.
    pub fn increment_local(&self, key: &str, ttl: f64) -> i64 {
        let now = self.now();
        let mut counters = self.counters.lock().unwrap_or_else(PoisonError::into_inner);
        let next = match counters.get(key, now) {
            Some(Counter::Int(value)) => value + 1,
            Some(Counter::Float(value)) => value as i64 + 1,
            None => 1,
        };
        counters.set(key, Counter::Int(next), ttl, now);
        next
    }

    pub fn local_count(&self, key: &str) -> f64 {
        let now = self.now();
        self.counters
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(key, now)
            .map_or(0.0, Counter::as_f64)
    }

    /// `DualCache.increment_cache`: memory first, then Redis `INCR` with the TTL set only when
    /// the key has none. A Redis failure answers with this process's own count.
    pub async fn increment_shared(&self, key: &str, ttl: f64) -> i64 {
        let local = self.increment_local(key, ttl);
        let Some(redis) = &self.redis else {
            return local;
        };
        let expire_seconds = ttl as i64;
        let result: redis::RedisResult<i64> = async {
            let mut connection = redis.connection().await?;
            let count: i64 = connection.incr(key, 1).await?;
            let current: i64 = connection.ttl(key).await?;
            if current == -1 {
                let _: () = connection.expire(key, expire_seconds).await?;
            }
            Ok(count)
        }
        .await;
        result.unwrap_or(local)
    }

    /// `DualCache.async_increment_cache`: memory, then Redis `INCRBYFLOAT` and the TTL read in
    /// one pipeline, with `EXPIRE` only when the key has no TTL.
    pub async fn increment_usage(&self, key: &str, by: f64, ttl: i64) {
        let now = self.now();
        {
            let mut counters = self.counters.lock().unwrap_or_else(PoisonError::into_inner);
            let next = counters.get(key, now).map_or(0.0, Counter::as_f64) + by;
            counters.set(key, Counter::Float(next), ttl as f64, now);
        }
        let Some(redis) = &self.redis else {
            return;
        };
        let _: redis::RedisResult<()> = async {
            let mut connection = redis.connection().await?;
            let (_, current): (f64, i64) = redis::pipe()
                .cmd("INCRBYFLOAT")
                .arg(key)
                .arg(by)
                .cmd("TTL")
                .arg(key)
                .query_async(&mut connection)
                .await?;
            if current == -1 {
                let _: () = connection.expire(key, ttl).await?;
            }
            Ok(())
        }
        .await;
    }

    /// `CooldownCache.add_deployment_to_cooldown`: memory, then Redis `SET` of the dict's
    /// `str()` with `EX int(ttl)`. A TTL below one second fails in Redis, so it is not sent.
    pub async fn set_cooldown(&self, model_id: &str, cooldown: Cooldown) {
        let key = Cooldown::key(model_id);
        let ttl = cooldown.cooldown_time.as_f64();
        let now = self.now();
        self.cooldowns
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .set(&key, cooldown.clone(), ttl, now);
        let Some(redis) = &self.redis else {
            return;
        };
        let expire_seconds = ttl as i64;
        if expire_seconds <= 0 {
            return;
        }
        let _: redis::RedisResult<()> = async {
            let mut connection = redis.connection().await?;
            connection
                .set_ex::<_, _, ()>(&key, cooldown.repr(), expire_seconds as u64)
                .await
        }
        .await;
    }

    /// Cooldown entries for `model_ids`, from memory and, for keys memory misses, from Redis at
    /// most once per read interval per key. Expired entries are not filtered.
    pub async fn cooldowns(&self, model_ids: &[&str]) -> Vec<Option<Cooldown>> {
        let now = self.now();
        let keys: Vec<String> = model_ids.iter().map(|id| Cooldown::key(id)).collect();
        let cached: Vec<Option<Cooldown>> = {
            let mut memory = self
                .cooldowns
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            keys.iter().map(|key| memory.get(key, now)).collect()
        };
        let Some(redis) = &self.redis else {
            return cached;
        };
        let due: Vec<&String> = {
            let mut reads = self
                .redis_reads
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            keys.iter()
                .zip(&cached)
                .filter(|(_, value)| value.is_none())
                .map(|(key, _)| key)
                .filter(|key| {
                    let due = reads
                        .get(key.as_str())
                        .is_none_or(|last| now - last >= self.redis_read_interval);
                    if due {
                        reads.insert((*key).clone(), now);
                    }
                    due
                })
                .collect()
        };
        if due.is_empty() {
            return cached;
        }
        let fetched: Vec<Option<String>> = match redis.connection().await {
            Ok(mut connection) => connection.mget(&due).await.unwrap_or_default(),
            Err(_) => Vec::new(),
        };
        let fetched: HashMap<&str, Cooldown> = due
            .iter()
            .zip(fetched)
            .filter_map(|(key, text)| {
                Some((
                    key.as_str(),
                    Cooldown::from_literal(pyrepr::parse(&text?)?)?,
                ))
            })
            .collect();
        let mut memory = self
            .cooldowns
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        keys.iter()
            .zip(cached)
            .map(|(key, value)| {
                value.or_else(|| {
                    let cooldown = fetched.get(key.as_str())?.clone();
                    let remaining = cooldown.remaining(now);
                    let ttl = if remaining > 0.0 {
                        remaining.min(CORRECTED_COOLDOWN_TTL_CAP)
                    } else {
                        DEFAULT_IN_MEMORY_TTL
                    };
                    memory.replace(key, cooldown.clone(), ttl, now);
                    Some(cooldown)
                })
            })
            .collect()
    }

    /// Ids among `model_ids` whose cooldown has not run out.
    pub async fn active_cooldowns(&self, model_ids: &[&str]) -> Vec<String> {
        let now = self.now();
        let entries = self.cooldowns(model_ids).await;
        let mut memory = self
            .cooldowns
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        model_ids
            .iter()
            .zip(entries)
            .filter_map(|(id, entry)| {
                let cooldown = entry?;
                if cooldown.remaining(now) > 0.0 {
                    Some((*id).to_owned())
                } else {
                    memory.remove(&Cooldown::key(id));
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use rstest::rstest;

    use super::{Clock, Cooldown, Store};
    use crate::pyrepr::{self, PyNumber};

    struct ManualClock(Mutex<f64>);

    impl Clock for ManualClock {
        fn now(&self) -> f64 {
            *self.0.lock().unwrap()
        }
    }

    fn store_at(clock: &Arc<ManualClock>) -> Store {
        Store::new(clock.clone(), None, 1.0)
    }

    #[rstest]
    fn counter_window_starts_at_the_first_increment() {
        let clock = Arc::new(ManualClock(Mutex::new(100.0)));
        let store = store_at(&clock);
        assert_eq!(store.increment_local("id:fails", 60.0), 1);
        *clock.0.lock().unwrap() = 159.0;
        assert_eq!(store.increment_local("id:fails", 60.0), 2);
        *clock.0.lock().unwrap() = 160.0;
        assert_eq!(store.local_count("id:fails"), 0.0);
        assert_eq!(store.increment_local("id:fails", 60.0), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn cooldown_is_active_until_its_timestamp_plus_cooldown_time() {
        let clock = Arc::new(ManualClock(Mutex::new(1000.0)));
        let store = store_at(&clock);
        store
            .set_cooldown(
                "a",
                Cooldown {
                    exception_received: "boom".into(),
                    status_code: "429".into(),
                    timestamp: 1000.0,
                    cooldown_time: PyNumber::Int(5),
                },
            )
            .await;
        assert_eq!(store.active_cooldowns(&["a", "b"]).await, ["a"]);
        *clock.0.lock().unwrap() = 1005.0;
        assert!(store.active_cooldowns(&["a", "b"]).await.is_empty());
    }

    #[rstest]
    fn cooldown_repr_is_the_python_dict_str_and_reads_back() {
        let cooldown = Cooldown {
            exception_received: "it's".into(),
            status_code: "429".into(),
            timestamp: 1728345600.25,
            cooldown_time: PyNumber::Float(5.0),
        };
        assert_eq!(
            cooldown.repr(),
            "{'exception_received': \"it's\", 'status_code': '429', 'timestamp': 1728345600.25, 'cooldown_time': 5.0}"
        );
        assert_eq!(
            Cooldown::from_literal(pyrepr::parse(&cooldown.repr()).unwrap()),
            Some(cooldown)
        );
    }
}
