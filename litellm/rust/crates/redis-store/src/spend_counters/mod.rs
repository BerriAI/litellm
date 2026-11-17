mod keys;
mod types;

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

use litellm_python_compat::repr::to_str;
use litellm_redis_client::{Declared, Expiry, RedisBatch, commands};

pub use self::keys::{NotASpendCounter, SpendCounter};
pub use self::types::{LostSeed, PostCallSettlement, Reading};
use crate::{
    Error, Number, Pending, RequestRedis, Send,
    dual_cache::{DualCache, MemorySettings},
    memory::{Clock, system_clock},
    remote::{Remote, RemoteSlot},
};

/// `RedisCache.async_set_max`, byte for byte, sent as `EVAL` like it: a counter only ever rises to
/// the authoritative spend.
const SET_MAX: &str = concat!(
    "local cur = redis.call('GET', KEYS[1]) ",
    "if cur == false or tonumber(cur) < tonumber(ARGV[1]) then ",
    "redis.call('SET', KEYS[1], ARGV[1]) ",
    "if tonumber(ARGV[2]) > 0 then redis.call('EXPIRE', KEYS[1], ARGV[2]) end ",
    "return ARGV[1] end ",
    "return cur"
);

/// The proxy's spend counters: Redis shared by every pod, and this pod's memory copy that answers
/// only when Redis cannot. Like `spend_counter_cache`'s callers on `main`, most ops reach past
/// `DualCache` to its Redis and its memory.
pub struct SpendCounters {
    cache: DualCache<Number>,
    committed_ttl_millis: AtomicU64,
}

impl SpendCounters {
    pub fn new(remote: Option<Remote>) -> Self {
        Self::with_clock(remote, system_clock())
    }

    pub fn with_clock(remote: Option<Remote>, now: Clock) -> Self {
        Self {
            cache: DualCache::new(
                Arc::new(RemoteSlot::new(remote)),
                MemorySettings::default(),
                now,
            ),
            committed_ttl_millis: AtomicU64::new(0),
        }
    }

    /// The least TTL a write gives a [`SpendCounter::Committed`] twin or a lifetime counter, sized
    /// to the configured batch flush window so a counter usually outlives the DB write that holds
    /// its charges. A write that lags past it can still leave the counter to reseed short.
    pub fn set_committed_ttl(&self, ttl: Duration) {
        let millis = u64::try_from(ttl.as_millis()).unwrap_or(u64::MAX);
        self.committed_ttl_millis.store(millis, Ordering::Relaxed);
    }

    fn ttl(&self, counter: &SpendCounter, default_ttl: Duration) -> Duration {
        match counter {
            SpendCounter::Committed(_)
            | SpendCounter::CustomerOrg(_)
            | SpendCounter::CustomerTeam(_)
            | SpendCounter::CustomerMember(_) => self.committed_floor(default_ttl),
            counter if is_lifetime(counter) => self.committed_floor(default_ttl),
            _ => default_ttl,
        }
    }

    fn committed_floor(&self, default_ttl: Duration) -> Duration {
        default_ttl.max(Duration::from_millis(
            self.committed_ttl_millis.load(Ordering::Relaxed),
        ))
    }

    /// A committed twin's or lifetime counter's expiry moves out on every charge, on a cluster too,
    /// to the TTL sized to the flush window.
    fn post_call_expiry(&self, counter: &SpendCounter, default_ttl: Duration) -> Expiry {
        let ttl = self.ttl(counter, default_ttl);
        match counter {
            SpendCounter::Committed(_) => Expiry::Refresh(ttl),
            counter if is_lifetime(counter) => Expiry::Refresh(ttl),
            _ => Expiry::KeepExisting(ttl),
        }
    }

    /// `DualCache.attach_redis_cache`: memory stays, and `false` means a Redis was already attached.
    pub fn attach(&self, remote: Remote) -> bool {
        self.cache.redis().attach(remote)
    }

    pub fn has_redis(&self) -> bool {
        self.cache.redis().get().is_some()
    }

    /// The TTL every Redis write gives a counter, `None` without Redis.
    pub fn redis_ttl(&self) -> Option<Duration> {
        self.cache.redis().get().map(|remote| remote.default_ttl)
    }

    /// `read_spend_counter_cache_value`: Redis first. `RedisCache.async_get_cache` swallows an unreachable
    /// `GET` as a miss, so an open breaker, a transport error, or a value `float()` refuses reads memory instead.
    pub async fn read(&self, counter: &SpendCounter) -> Result<Reading, Error> {
        let Some(client) = self.cache.redis().client() else {
            return Ok(Reading::Memory(self.get_local(counter)?));
        };
        match client.send(commands::get::<Vec<u8>>(&counter.key())).await {
            Ok(raw) => match types::decode(raw) {
                Ok(value) => Ok(Reading::Redis(value)),
                Err(error) => {
                    tracing::debug!(%error, "spend counter Redis value did not decode, falling back to memory");
                    Ok(Reading::Memory(self.get_local(counter)?))
                }
            },
            Err(litellm_redis_client::Error::BreakerOpen) => {
                Ok(Reading::Memory(self.get_local(counter)?))
            }
            Err(error) => {
                crate::failure_log::redis_failure!(&error, "spend counter Redis read failed");
                Ok(Reading::Memory(self.get_local(counter)?))
            }
        }
    }

    /// One `MGET` for many counters on the batch `send` picks; `None` without Redis. The read
    /// resolves to `None` when it failed, so each caller reads on its own.
    pub fn read_many(
        &self,
        send: Send<'_>,
        counters: &[SpendCounter],
    ) -> Option<Pending<Option<Vec<Option<Number>>>>> {
        let batch = self.cache.redis().batch(send)?;
        let keys: Vec<String> = counters.iter().map(SpendCounter::key).collect();
        let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
        let read = batch.add(commands::mget::<Vec<u8>>(&keys));
        Some(Box::pin(async move {
            let decoded = read
                .resolve()
                .await
                .map_err(Error::from)
                .and_then(|replies| {
                    replies
                        .into_iter()
                        .map(|raw| Ok(types::decode(raw)?))
                        .collect::<Result<Vec<_>, Error>>()
                });
            Ok(decoded
                .inspect_err(|error| {
                    tracing::debug!(%error, "spend counter batch read failed, falling back to per-key reads");
                })
                .ok())
        }))
    }

    /// `increment_spend_counter_cache`: `INCRBYFLOAT` refreshing the TTL, mirrored into memory. A
    /// failed increment may or may not have landed, so the counter is dropped everywhere before the
    /// error returns. Without Redis it is a memory increment.
    pub async fn increment(&self, counter: &SpendCounter, amount: Number) -> Result<Number, Error> {
        let Some(remote) = self.cache.redis().get() else {
            return self.increment_local(counter, amount);
        };
        let key = counter.key();
        let increment = commands::incr_by_float(
            &key,
            amount.into(),
            Some(Expiry::Refresh(self.ttl(counter, remote.default_ttl))),
        );
        match remote.client.send(increment).await {
            Ok(total) => {
                let total = Number::Float(total);
                self.cache.memory().set(&key, total, None)?;
                Ok(total)
            }
            Err(error) => {
                self.invalidate(std::slice::from_ref(counter)).await;
                Err(error.into())
            }
        }
    }

    /// `run_spend_counter_pipeline`: every increment in one round trip, totals in order. The caller
    /// decides what a failure means for counters whose increment may or may not have landed.
    pub async fn increment_many(
        &self,
        increments: &[(SpendCounter, Number)],
    ) -> Result<Vec<Number>, Error> {
        let Some(remote) = self.cache.redis().get() else {
            return increments
                .iter()
                .map(|(counter, amount)| self.increment_local(counter, *amount))
                .collect();
        };
        let pipeline = RedisBatch::pipelined(remote.client.clone());
        let totals = increments.iter().map(|(counter, amount)| {
            pipeline.add(commands::incr_by_float(
                &counter.key(),
                (*amount).into(),
                Some(Expiry::Refresh(self.ttl(counter, remote.default_ttl))),
            ))
        });
        let totals = Declared::all(totals.collect::<Vec<_>>()).await;
        increments
            .iter()
            .zip(totals)
            .map(|((counter, _), total)| {
                let total = Number::Float(total?);
                self.cache.memory().set(&counter.key(), total, None)?;
                Ok(total)
            })
            .collect()
    }

    /// `increment_many` on the request's post-call batch. Each total reaches memory once the batch
    /// goes out, and a counter whose increment failed is dropped everywhere. `None` without Redis,
    /// else a settlement that reports when every reply is back, whoever flushed the batch.
    pub fn increment_many_post_call(
        self: &Arc<Self>,
        request: &RequestRedis,
        increments: &[(SpendCounter, Number)],
    ) -> Option<PostCallSettlement> {
        let remote = self.cache.redis().get()?;
        let batch = Send::PostCall(request).batch(&remote.client);
        let declared = increments
            .iter()
            .map(|(counter, amount)| {
                let total = batch.add(commands::incr_by_float(
                    &counter.key(),
                    (*amount).into(),
                    Some(self.post_call_expiry(counter, remote.default_ttl)),
                ));
                (counter.clone(), total)
            })
            .collect::<Vec<_>>();
        let store = Arc::clone(self);
        let settled = Arc::new(AtomicBool::new(false));
        let settlement = PostCallSettlement(Arc::clone(&settled));
        request.spawn(async move {
            for (counter, total) in declared {
                match total.settled().await {
                    Ok(total) => {
                        if let Err(error) = store.cache.memory().set(&counter.key(), Number::Float(total), None) {
                            tracing::warn!(%error, "mirroring a spend counter total into memory failed");
                        }
                    }
                    Err(error) => {
                        let key = counter.key();
                        crate::failure_log::redis_failure!(
                            &error,
                            &format!("spend counter {key} increment did not land in the post-call pipeline; invalidating it"),
                        );
                        store.invalidate(std::slice::from_ref(&counter)).await;
                    }
                }
            }
            settled.store(true, Ordering::Release);
        });
        Some(settlement)
    }

    /// `_repair_stale_spend_counter`: raise memory and Redis to `spend`, never lowering either.
    /// `None` without Redis, else what Redis holds afterwards.
    pub async fn set_max(
        &self,
        counter: &SpendCounter,
        spend: Number,
    ) -> Result<Option<Number>, Error> {
        let key = counter.key();
        self.cache
            .memory()
            .set_if(&key, spend, None, |cached| cached.as_f64() < spend.as_f64())?;
        let Some(remote) = self.cache.redis().get() else {
            return Ok(None);
        };
        let args = (
            to_str(&spend.to_value()),
            self.ttl(counter, remote.default_ttl).as_secs().to_string(),
        );
        let stored: Option<f64> = remote
            .client
            .send(commands::eval_source(SET_MAX, &[&key], args))
            .await?;
        Ok(stored.map(Number::Float))
    }

    /// `SpendCounterReseed.coalesced(_window)`: seed a cold counter with `SET NX` so one pod wins,
    /// answer with whatever the counter holds, and mirror it into memory. Without Redis, memory
    /// keeps the larger of the seed and what it held.
    pub async fn seed(
        &self,
        counter: &SpendCounter,
        spend: Number,
        lost: LostSeed,
    ) -> Result<Number, Error> {
        let key = counter.key();
        let Some(Remote {
            client,
            default_ttl,
        }) = self.cache.redis().get()
        else {
            return Ok(self.cache.memory().update(&key, None, |cached| {
                Ok(match cached {
                    Some(cached) if cached.as_f64() > spend.as_f64() => {
                        Number::Float(cached.as_f64())
                    }
                    _ => spend,
                })
            })?);
        };
        let ttl = self.ttl(counter, *default_ttl);
        let current = if client
            .send(commands::set_nx(&key, spend.to_json(), ttl))
            .await?
        {
            spend
        } else {
            match types::decode(client.send(commands::get::<Vec<u8>>(&key)).await?)? {
                Some(winner) => Number::Float(winner.as_f64()),
                None => match lost {
                    LostSeed::Answer => spend,
                    LostSeed::Increment => {
                        let increment = commands::incr_by_float(
                            &key,
                            spend.into(),
                            Some(Expiry::KeepExisting(ttl)),
                        );
                        Number::Float(client.send(increment).await?)
                    }
                },
            }
        };
        self.cache.memory().set(&key, current, None)?;
        Ok(current)
    }

    /// A spend reset: memory, then Redis, each for `ttl` or its own default.
    pub async fn set(
        &self,
        counter: &SpendCounter,
        spend: Number,
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        let key = counter.key();
        self.cache.memory().set(&key, spend, ttl)?;
        let Some(remote) = self.cache.redis().get() else {
            return Ok(());
        };
        let ttl = ttl.unwrap_or_else(|| self.ttl(counter, remote.default_ttl));
        remote
            .client
            .send(commands::set(&key, spend.to_json(), ttl))
            .await?;
        Ok(())
    }

    /// Drop counters from memory, then from Redis in one round trip.
    pub async fn delete(&self, counters: &[SpendCounter]) -> Result<(), Error> {
        let keys: Vec<String> = counters.iter().map(SpendCounter::key).collect();
        self.cache.delete_keys(&keys).await
    }

    /// `refresh_spend_counter_ttl`: `false` without Redis or for a counter Redis no longer holds.
    pub async fn refresh_ttl(&self, counter: &SpendCounter) -> Result<bool, Error> {
        let Some(remote) = self.cache.redis().get() else {
            return Ok(false);
        };
        Ok(remote
            .client
            .send(commands::expire(
                &counter.key(),
                self.ttl(counter, remote.default_ttl),
            ))
            .await?)
    }

    pub fn get_local(&self, counter: &SpendCounter) -> Result<Option<Number>, Error> {
        Ok(self.cache.memory().get(&counter.key())?)
    }

    pub fn set_local(
        &self,
        counter: &SpendCounter,
        spend: Number,
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        Ok(self.cache.memory().set(&counter.key(), spend, ttl)?)
    }

    pub fn delete_local(&self, counter: &SpendCounter) -> Result<(), Error> {
        Ok(self.cache.memory().delete(&counter.key())?)
    }

    /// `DualCache.async_increment_cache(local_only=True)`, which passes `InMemoryCache` no TTL. A
    /// lifetime counter's charge moves its deadline out like a Redis write would, so without Redis
    /// it too outlives the DB write that holds the charge.
    pub fn increment_local(&self, counter: &SpendCounter, amount: Number) -> Result<Number, Error> {
        let memory = self.cache.memory();
        let key = counter.key();
        if is_lifetime(counter) {
            let ttl = self.ttl(counter, memory.default_ttl());
            return Ok(memory.increment_refreshing(&key, amount, ttl)?);
        }
        Ok(memory.increment(&key, amount, None)?)
    }

    pub fn flush_local(&self) -> Result<(), Error> {
        Ok(self.cache.memory().flush()?)
    }

    pub fn snapshot(&self) -> Result<Vec<(String, Number)>, Error> {
        Ok(self.cache.memory().entries()?)
    }

    /// `invalidate_spend_counter`: memory always, Redis best effort.
    pub async fn invalidate(&self, counters: &[SpendCounter]) {
        if let Err(error) = self.delete(counters).await {
            tracing::debug!(%error, "dropping spend counters from Redis failed");
        }
    }
}

/// An entity's lifetime counter, reseeded on a miss from the DB row the batch writer fills late.
fn is_lifetime(counter: &SpendCounter) -> bool {
    matches!(
        counter,
        SpendCounter::Key { window: None, .. }
            | SpendCounter::Team { window: None, .. }
            | SpendCounter::TeamMember { .. }
            | SpendCounter::User(_)
            | SpendCounter::Org(_)
            | SpendCounter::Project(_)
            | SpendCounter::EndUser(_)
    )
}

#[cfg(test)]
mod tests {
    use super::{SpendCounter, SpendCounters};
    use crate::{Number, test_support::clock};
    use rstest::rstest;
    use std::{sync::atomic::Ordering, time::Duration};

    fn key(token: &str) -> SpendCounter {
        SpendCounter::Key {
            token: token.into(),
            window: None,
        }
    }

    #[rstest]
    #[case::key(key("k"))]
    #[case::team(SpendCounter::Team { team_id: "t".into(), window: None })]
    #[case::team_member(SpendCounter::TeamMember { user_id: "u".into(), team_id: "t".into() })]
    #[case::user(SpendCounter::User("u".into()))]
    #[case::org(SpendCounter::Org("o".into()))]
    #[case::project(SpendCounter::Project("p".into()))]
    #[case::end_user(SpendCounter::EndUser("e".into()))]
    fn without_redis_each_charge_keeps_a_lifetime_counter_for_its_lifetime_from_that_charge(
        #[case] counter: SpendCounter,
        #[values(12, 0)] memory_defaults: u64,
    ) {
        let (now, seconds) = clock();
        let store = SpendCounters::with_clock(None, now);
        let memory_default = store.cache.memory().default_ttl().as_secs();
        let committed_ttl = (memory_defaults * memory_default).max(1);
        store.set_committed_ttl(Duration::from_secs(committed_ttl));
        let lifetime = committed_ttl.max(memory_default);

        store
            .increment_local(&counter, Number::Float(0.25))
            .unwrap();
        seconds.fetch_add(lifetime - 1, Ordering::SeqCst);
        store
            .increment_local(&counter, Number::Float(0.25))
            .unwrap();
        seconds.fetch_add(lifetime - 1, Ordering::SeqCst);
        let kept = store.get_local(&counter).unwrap();
        seconds.fetch_add(2, Ordering::SeqCst);

        assert_eq!(kept, Some(Number::Float(0.5)));
        assert_eq!(store.get_local(&counter).unwrap(), None);
    }

    #[rstest]
    #[case::key_window(SpendCounter::Key { token: "k".into(), window: Some("1d".into()) })]
    #[case::team_window(SpendCounter::Team { team_id: "t".into(), window: Some("1d".into()) })]
    #[case::tag(SpendCounter::Tag("tag".into()))]
    #[case::committed_twin(SpendCounter::Committed(Box::new(key("k"))))]
    fn without_redis_a_charge_leaves_other_counters_on_their_first_deadline(
        #[case] counter: SpendCounter,
    ) {
        let (now, seconds) = clock();
        let store = SpendCounters::with_clock(None, now);
        let memory_default = store.cache.memory().default_ttl().as_secs();
        store.set_committed_ttl(Duration::from_secs(12 * memory_default));

        store.increment_local(&counter, Number::Int(1)).unwrap();
        seconds.fetch_add(memory_default - 1, Ordering::SeqCst);
        store.increment_local(&counter, Number::Int(1)).unwrap();
        let kept = store.get_local(&counter).unwrap();
        seconds.fetch_add(2, Ordering::SeqCst);

        assert_eq!(kept, Some(Number::Int(2)));
        assert_eq!(store.get_local(&counter).unwrap(), None);
    }

    #[rstest]
    #[tokio::test]
    async fn without_redis_a_reset_ttl_holds_while_no_charge_follows() {
        let (now, seconds) = clock();
        let store = SpendCounters::with_clock(None, now);
        store.set_committed_ttl(Duration::from_secs(7_200));
        let counter = key("k");
        let reset_ttl = 60;

        store
            .set(
                &counter,
                Number::Float(0.0),
                Some(Duration::from_secs(reset_ttl)),
            )
            .await
            .unwrap();
        seconds.fetch_add(reset_ttl - 1, Ordering::SeqCst);
        let kept = store.get_local(&counter).unwrap();
        seconds.fetch_add(2, Ordering::SeqCst);

        assert_eq!(kept, Some(Number::Float(0.0)));
        assert_eq!(store.get_local(&counter).unwrap(), None);
    }

    #[rstest]
    #[tokio::test]
    async fn without_redis_a_charge_after_a_reset_moves_the_deadline_out_to_the_committed_ttl() {
        let (now, seconds) = clock();
        let store = SpendCounters::with_clock(None, now);
        let committed_ttl = 7_200;
        store.set_committed_ttl(Duration::from_secs(committed_ttl));
        let counter = key("k");

        store
            .set(&counter, Number::Float(0.0), Some(Duration::from_secs(60)))
            .await
            .unwrap();
        seconds.fetch_add(30, Ordering::SeqCst);
        store.increment_local(&counter, Number::Float(1.0)).unwrap();
        seconds.fetch_add(committed_ttl - 1, Ordering::SeqCst);
        let kept = store.get_local(&counter).unwrap();
        seconds.fetch_add(2, Ordering::SeqCst);

        assert_eq!(kept, Some(Number::Float(1.0)));
        assert_eq!(store.get_local(&counter).unwrap(), None);
    }

    #[rstest]
    #[case::organization(SpendCounter::CustomerOrg("tenant:lifetime".into()))]
    #[case::team(SpendCounter::CustomerTeam("team:lifetime".into()))]
    #[case::member(SpendCounter::CustomerMember("member:lifetime".into()))]
    fn customer_counters_outlive_the_pending_database_flush(#[case] counter: SpendCounter) {
        let store = SpendCounters::new(None);
        store.set_committed_ttl(Duration::from_secs(120));
        assert_eq!(
            store.ttl(&counter, Duration::from_secs(1)),
            Duration::from_secs(120)
        );
    }
}
