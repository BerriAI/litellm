//! `get_current_spend` and `ensure_spend_counter_initialized` for a lifetime counter: what a
//! budget is checked against before a call, and the warm a reservation runs before it charges.
//!
//! The counter is read from Redis first, since every pod charges it there, and from this process's
//! memory only when Redis cannot answer, as memory holds only this pod's charges. A clean Redis miss
//! is a cold counter, seeded once from its row under the counter's seed lock (`SET NX`, so one pod
//! wins). For a cold counter with no row, a read answers the spend the caller's plan read and
//! writes nothing, as `get_current_spend` does. A warm raises it to that spend with `set_max`
//! instead, as `ensure_spend_counter_initialized` does, so the charges that follow start from it.
//!
//! A counter Redis reloaded from an older snapshot reads below the spend its caller already knows
//! of, so a read under the budget that is below that spend is held to its row's spend as a floor,
//! and the counter is raised to it with `set_max`, never lowered. The floor is kept in memory for
//! [`FLOOR_TTL`], so a counter that stays stale costs one row read per window.
//!
//! Unlike Python, a kept floor never raises a counter that reads below it. Only a reset lowers a
//! counter once its floor was kept, and Python's budget reset job zeroes the row and deletes the
//! counter without announcing a floor, so a kept floor above the counter is reread from the row.
//!
//! As in Python, an end user's counter is never seeded from its row, only floored by it.

use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_redis_store::{
    Error, Number,
    spend_counters::{FLOOR_TTL, LostSeed, Reading, SpendCounter, SpendCounters},
};

use crate::charge::{Failure, raise_to_cached_spend, store_failure};

/// The spend a counter's row holds, `None` when it has no row or the read failed.
pub trait RowSpend: Send + Sync {
    fn stored<'a>(&'a self, counter: &'a SpendCounter) -> BoxFuture<'a, Option<f64>>;
}

/// Reads a counter's current spend from the store a budget reservation holds against, seeding and
/// repairing it from its row. `rows` is the proxy's database, since without one a counter can be
/// neither seeded nor floored.
pub struct SpendReader {
    store: Arc<SpendCounters>,
    rows: Arc<dyn RowSpend>,
}

/// What a cold counter's seed under its lock found.
enum Seed {
    /// Another task in this process seeded it while this one waited for the lock.
    Warm(f64),
    FromRow {
        spend: f64,
        seeded: Result<Number, Error>,
    },
    NoRow,
}

impl SpendReader {
    pub fn new(store: Arc<SpendCounters>, rows: Arc<dyn RowSpend>) -> Self {
        Self { store, rows }
    }

    /// What `counter` holds, a cold one first seeded, and one under `max_budget` held to its row's
    /// spend when it reads below `fallback_spend`, the spend its caller's row was read with.
    pub async fn current(
        &self,
        counter: &SpendCounter,
        fallback_spend: f64,
        max_budget: Option<f64>,
    ) -> f64 {
        let current = match self.cached(counter).await {
            Some(current) => current,
            None => match self.seed(counter).await {
                Seed::Warm(current) => current,
                Seed::FromRow { spend, seeded } => seeded.map_or(spend, Number::as_f64),
                Seed::NoRow => fallback_spend,
            },
        };
        if max_budget.is_none_or(|max_budget| current >= max_budget) || fallback_spend <= current {
            return current;
        }
        match self.floor(counter, current).await {
            Some(floor) if floor > current => {
                self.repair(counter, floor).await;
                floor
            }
            Some(_) => current,
            None => fallback_spend,
        }
    }

    async fn cached(&self, counter: &SpendCounter) -> Option<f64> {
        let read = self.store.read(counter).await.ok()?;
        read.value().map(Number::as_f64)
    }

    /// `ensure_spend_counter_initialized`: makes sure a cold counter holds the spend already made
    /// before a reservation charges it. One with a row is seeded from it, and one without, or whose
    /// row read failed, is raised to `fallback_spend` when that is above zero, after the seed lock
    /// is released. A failure leaves the counter for the caller to drop.
    pub async fn warm(&self, counter: &SpendCounter, fallback_spend: f64) -> Result<(), Failure> {
        if self.is_warm(counter).await? {
            return Ok(());
        }
        match self.seed(counter).await {
            Seed::Warm(_) => Ok(()),
            Seed::FromRow { seeded, .. } => seeded.map(drop).map_err(store_failure),
            Seed::NoRow if fallback_spend > 0.0 => {
                raise_to_cached_spend(&self.store, counter, fallback_spend).await
            }
            Seed::NoRow => Ok(()),
        }
    }

    /// `_is_spend_counter_cache_warm`: whether the counter holds a value, keeping what Redis holds
    /// in memory for when Redis cannot answer.
    async fn is_warm(&self, counter: &SpendCounter) -> Result<bool, Failure> {
        match self.store.read(counter).await.map_err(store_failure)? {
            Reading::Redis(Some(spend)) => {
                self.store
                    .set_local(counter, spend, None)
                    .map_err(store_failure)?;
                Ok(true)
            }
            reading => Ok(reading.value().is_some()),
        }
    }

    /// `SpendCounterReseed.coalesced`: seeds a cold counter from its row under the counter's seed
    /// lock, so concurrent seeds in this process read the row once.
    async fn seed(&self, counter: &SpendCounter) -> Seed {
        let locks = self.store.lock_seeds(std::slice::from_ref(counter)).await;
        if locks.waited()
            && let Some(current) = self.cached(counter).await
        {
            return Seed::Warm(current);
        }
        let stored = match counter {
            SpendCounter::EndUser(_) => None,
            _ => self.rows.stored(counter).await,
        };
        match stored {
            Some(spend) => Seed::FromRow {
                spend,
                seeded: self
                    .store
                    .seed(counter, Number::Float(spend), LostSeed::Answer)
                    .await,
            },
            None => Seed::NoRow,
        }
    }

    /// `_authoritative_floor_spend`: the row's spend, read at most once per [`FLOOR_TTL`] while the
    /// counter reads at or above it. A reset announced while the row was read wins over what the
    /// read found.
    async fn floor(&self, counter: &SpendCounter, current: f64) -> Option<f64> {
        let kept = || self.store.floor(counter).ok().flatten().map(Number::as_f64);
        match kept() {
            Some(floor) if floor <= current => return Some(floor),
            Some(_) => {
                if let Err(error) = self.store.delete_floor(counter) {
                    tracing::debug!(%error, "dropping a spend counter's kept floor failed");
                }
            }
            None => {}
        }
        let spend = self.rows.stored(counter).await?;
        if let Some(floor) = kept() {
            return Some(floor);
        }
        if let Err(error) = self
            .store
            .set_floor(counter, Number::Float(spend), Some(FLOOR_TTL))
        {
            tracing::debug!(%error, "keeping a spend counter's floor in memory failed");
        }
        Some(spend)
    }

    /// `_repair_stale_spend_counter`: raises the counter to its row's spend, never lowering it.
    async fn repair(&self, counter: &SpendCounter, spend: f64) {
        if let Err(error) = self.store.set_max(counter, Number::Float(spend)).await {
            tracing::debug!(%error, "raising a stale spend counter to its row's spend failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{
            Mutex,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };

    use litellm_redis_client::{
        Credentials, Namespace, RedisClient, RedisConfig, RedisUrl, Target, Timeouts,
    };
    use litellm_redis_store::Remote;
    use litellm_redis_testing::RedisServer;
    use litellm_secret_value::SecretValue;
    use rstest::{fixture, rstest};

    use super::*;

    /// The spend each counter's row holds, by key, counting the reads.
    #[derive(Default)]
    struct Rows {
        spend: Mutex<HashMap<String, f64>>,
        reads: AtomicUsize,
    }

    impl Rows {
        fn holding(counter: &SpendCounter, spend: Option<f64>) -> Arc<Self> {
            Arc::new(Self {
                spend: Mutex::new(
                    spend
                        .map(|spend| (counter.key(), spend))
                        .into_iter()
                        .collect(),
                ),
                reads: AtomicUsize::new(0),
            })
        }

        /// What a budget reset leaves in the row: its spend zeroed.
        fn reset(&self, counter: &SpendCounter) {
            self.spend.lock().unwrap().insert(counter.key(), 0.0);
        }

        fn reads(&self) -> usize {
            self.reads.load(Ordering::SeqCst)
        }
    }

    impl RowSpend for Rows {
        fn stored<'a>(&'a self, counter: &'a SpendCounter) -> BoxFuture<'a, Option<f64>> {
            Box::pin(async move {
                self.reads.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(20)).await;
                self.spend.lock().unwrap().get(&counter.key()).copied()
            })
        }
    }

    fn team() -> SpendCounter {
        SpendCounter::Team {
            team_id: "t".to_owned(),
            window: None,
        }
    }

    fn end_user() -> SpendCounter {
        SpendCounter::EndUser("e".to_owned())
    }

    #[fixture]
    async fn server() -> RedisServer {
        RedisServer::start().await.unwrap()
    }

    fn store(server: &RedisServer) -> Arc<SpendCounters> {
        let client = RedisClient::new(RedisConfig {
            target: Target::Url(RedisUrl {
                url: SecretValue::new(server.url().to_owned()),
                tls: None,
                db: None,
                username: None,
                password: None,
            }),
            credentials: Credentials::NONE,
            namespace: Namespace::default(),
            timeouts: Timeouts::new(Duration::from_millis(500), None),
            breaker: None,
            events: None,
            health_check: None,
            keepalive: true,
        })
        .unwrap();
        Arc::new(SpendCounters::new(Some(Remote {
            client,
            default_ttl: Duration::from_secs(60),
        })))
    }

    /// What Redis holds for `counter`, as another pod reads it.
    async fn in_redis(server: &RedisServer, counter: &SpendCounter) -> Option<f64> {
        store(server)
            .read(counter)
            .await
            .unwrap()
            .value()
            .map(Number::as_f64)
    }

    async fn held(store: &SpendCounters, counter: &SpendCounter, spend: f64) {
        store
            .set(counter, Number::Float(spend), None)
            .await
            .unwrap();
    }

    #[rstest]
    #[tokio::test]
    async fn a_warm_counter_answers_from_redis_without_reading_its_row(
        #[future] server: RedisServer,
    ) {
        let server = server.await;
        let store = store(&server);
        held(&store, &team(), 2.0).await;
        let rows = Rows::holding(&team(), Some(9.0));
        let reader = SpendReader::new(store, rows.clone());

        let current = reader.current(&team(), 0.0, Some(10.0)).await;

        assert_eq!((current, rows.reads()), (2.0, 0));
    }

    #[rstest]
    #[tokio::test]
    async fn a_cold_counter_is_seeded_from_its_row_for_every_pod(#[future] server: RedisServer) {
        let server = server.await;
        let reader = SpendReader::new(store(&server), Rows::holding(&team(), Some(4.0)));

        let read = reader.current(&team(), 1.0, None).await;

        assert_eq!((read, in_redis(&server, &team()).await), (4.0, Some(4.0)));
    }

    /// `get_current_spend` answers the caller's spend for a counter with no row and writes
    /// nothing, as only `ensure_spend_counter_initialized` seeds a counter from it.
    #[rstest]
    #[case::a_team_without_a_row(team(), None, 3.0)]
    #[case::a_team_with_nothing_spent(team(), None, 0.0)]
    #[case::an_end_user_whatever_its_row_holds(end_user(), Some(4.0), 1.0)]
    #[tokio::test]
    async fn a_cold_counter_without_a_row_answers_the_plans_spend_and_writes_nothing(
        #[future] server: RedisServer,
        #[case] counter: SpendCounter,
        #[case] row: Option<f64>,
        #[case] fallback_spend: f64,
    ) {
        let server = server.await;
        let store = store(&server);
        let reader = SpendReader::new(store.clone(), Rows::holding(&counter, row));

        let read = reader.current(&counter, fallback_spend, None).await;

        assert_eq!(
            (
                read,
                in_redis(&server, &counter).await,
                store.get_local(&counter).unwrap()
            ),
            (fallback_spend, None, None)
        );
    }

    #[rstest]
    #[tokio::test]
    async fn concurrent_reads_of_a_cold_counter_read_its_row_once(#[future] server: RedisServer) {
        let server = server.await;
        let rows = Rows::holding(&team(), Some(4.0));
        let reader = SpendReader::new(store(&server), rows.clone());

        let counter = team();
        let (first, second) = tokio::join!(
            reader.current(&counter, 0.0, None),
            reader.current(&counter, 0.0, None)
        );

        assert_eq!((first, second, rows.reads()), (4.0, 4.0, 1));
    }

    #[rstest]
    #[case::below_its_row_is_raised_to_it(1.0, 3.0, Some(10.0), Some(5.0), 5.0, 5.0)]
    #[case::at_its_budget_is_left(12.0, 13.0, Some(10.0), Some(15.0), 12.0, 12.0)]
    #[case::without_a_budget_is_left(1.0, 3.0, None, Some(5.0), 1.0, 1.0)]
    #[case::at_its_plans_spend_is_left(3.0, 3.0, Some(10.0), Some(5.0), 3.0, 3.0)]
    #[case::without_a_row_answers_the_plans_spend(1.0, 3.0, Some(10.0), None, 3.0, 1.0)]
    #[case::above_its_row_is_left(1.0, 3.0, Some(10.0), Some(0.5), 1.0, 1.0)]
    #[tokio::test]
    async fn a_counter_that_reads_below_its_plans_spend_is_held_to_its_row(
        #[future] server: RedisServer,
        #[case] counter: f64,
        #[case] fallback_spend: f64,
        #[case] max_budget: Option<f64>,
        #[case] row: Option<f64>,
        #[case] current: f64,
        #[case] left: f64,
    ) {
        let server = server.await;
        let store = store(&server);
        held(&store, &team(), counter).await;
        let reader = SpendReader::new(store, Rows::holding(&team(), row));

        let read = reader.current(&team(), fallback_spend, max_budget).await;

        assert_eq!(
            (read, in_redis(&server, &team()).await),
            (current, Some(left))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_stale_counter_reads_its_row_once_per_floor_window(#[future] server: RedisServer) {
        let server = server.await;
        let store = store(&server);
        held(&store, &team(), 1.0).await;
        let rows = Rows::holding(&team(), Some(2.0));
        let reader = SpendReader::new(store, rows.clone());

        let first = reader.current(&team(), 3.0, Some(10.0)).await;
        let second = reader.current(&team(), 3.0, Some(10.0)).await;

        assert_eq!((first, second, rows.reads()), (2.0, 2.0, 1));
    }

    #[rstest]
    #[tokio::test]
    async fn a_reset_floor_keeps_a_just_reset_counter_from_being_raised_back(
        #[future] server: RedisServer,
    ) {
        let server = server.await;
        let store = store(&server);
        held(&store, &team(), 0.0).await;
        store
            .set_floor(&team(), Number::Float(0.0), Some(FLOOR_TTL))
            .unwrap();
        let reader = SpendReader::new(store, Rows::holding(&team(), Some(5.0)));

        let read = reader.current(&team(), 3.0, Some(10.0)).await;

        assert_eq!((read, in_redis(&server, &team()).await), (0.0, Some(0.0)));
    }

    /// Python's budget reset job zeroes the row and deletes the counter without announcing a
    /// floor, and another pod may reseed the counter from the zeroed row before this one reads
    /// it. The caller's plan still carries the spend from before the reset, so the floor this
    /// process kept from before it must not raise the counter back.
    #[rstest]
    #[case::deleted(None)]
    #[case::reseeded_by_another_pod(Some(0.0))]
    #[tokio::test]
    async fn a_budget_reset_is_not_undone_by_the_floor_kept_from_before_it(
        #[future] server: RedisServer,
        #[case] after_reset: Option<f64>,
    ) {
        let server = server.await;
        let store = store(&server);
        held(&store, &team(), 1.0).await;
        let rows = Rows::holding(&team(), Some(5.0));
        let reader = SpendReader::new(store.clone(), rows.clone());
        let before = reader.current(&team(), 5.0, Some(10.0)).await;
        rows.reset(&team());
        store.delete(std::slice::from_ref(&team())).await.unwrap();
        if let Some(spend) = after_reset {
            held(&store, &team(), spend).await;
        }

        let after = reader.current(&team(), 5.0, Some(10.0)).await;

        assert_eq!(
            (before, after, in_redis(&server, &team()).await),
            (5.0, 0.0, Some(0.0))
        );
    }

    /// `ensure_spend_counter_initialized` seeds a cold counter from its row, raises one without a
    /// row to the plan's spend when that is above zero, and leaves a warm one as it is.
    #[rstest]
    #[case::cold_with_a_row(team(), None, Some(4.0), 1.0, Some(4.0))]
    #[case::cold_with_a_row_that_holds_nothing(team(), None, Some(0.0), 3.0, Some(0.0))]
    #[case::cold_without_a_row(team(), None, None, 3.0, Some(3.0))]
    #[case::cold_without_a_row_or_spend(team(), None, None, 0.0, None)]
    #[case::cold_end_user_whatever_its_row_holds(end_user(), None, Some(4.0), 1.0, Some(1.0))]
    #[case::warm_below_the_plans_spend(team(), Some(2.0), Some(9.0), 5.0, Some(2.0))]
    #[tokio::test]
    async fn a_warm_leaves_the_counter_holding_the_spend_already_made(
        #[future] server: RedisServer,
        #[case] counter: SpendCounter,
        #[case] holding: Option<f64>,
        #[case] row: Option<f64>,
        #[case] fallback_spend: f64,
        #[case] warmed: Option<f64>,
    ) {
        let server = server.await;
        let store = store(&server);
        if let Some(spend) = holding {
            held(&store, &counter, spend).await;
        }
        let reader = SpendReader::new(store, Rows::holding(&counter, row));

        reader.warm(&counter, fallback_spend).await.unwrap();

        assert_eq!(in_redis(&server, &counter).await, warmed);
    }

    #[rstest]
    #[tokio::test]
    async fn a_warm_counter_is_kept_in_memory_for_when_redis_cannot_answer(
        #[future] server: RedisServer,
    ) {
        let server = server.await;
        held(&store(&server), &team(), 2.0).await;
        let reader = SpendReader::new(store(&server), Rows::holding(&team(), None));
        reader.warm(&team(), 0.0).await.unwrap();
        server.stop().await.unwrap();

        assert_eq!(reader.current(&team(), 0.0, None).await, 2.0);
    }

    #[rstest]
    #[tokio::test]
    async fn a_warm_redis_cannot_take_fails(#[future] server: RedisServer) {
        let server = server.await;
        let reader = SpendReader::new(store(&server), Rows::holding(&team(), None));
        server.stop().await.unwrap();

        assert!(reader.warm(&team(), 3.0).await.is_err());
    }

    #[rstest]
    #[tokio::test]
    async fn an_unreachable_redis_answers_from_this_processs_memory(#[future] server: RedisServer) {
        let server = server.await;
        let store = store(&server);
        store.set_local(&team(), Number::Float(3.0), None).unwrap();
        server.stop().await.unwrap();
        let reader = SpendReader::new(store, Rows::holding(&team(), Some(1.0)));

        assert_eq!(reader.current(&team(), 0.0, None).await, 3.0);
    }
}
