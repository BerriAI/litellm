//! `get_current_spend` for a lifetime counter: what a budget is checked against before a call.
//!
//! The counter is read from Redis first, since every pod charges it there, and from this process's
//! memory only when Redis cannot answer, as memory holds only this pod's charges. A clean Redis miss
//! is a cold counter, seeded once from its row under the counter's seed lock (`SET NX`, so one pod
//! wins), or from the spend the caller's plan read when the row has none, as
//! `ensure_spend_counter_initialized` seeds it before a reservation reads it.
//!
//! A counter Redis reloaded from an older snapshot reads below the spend its caller already knows
//! of, so a read under the budget that is below that spend is held to its row's spend as a floor,
//! and the counter is raised to it with `set_max`, never lowered. The floor is kept in memory for
//! [`FLOOR_TTL`], so a counter that stays stale costs one row read per window.
//!
//! As in Python, an end user's counter is never seeded from its row, only floored by it.

use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_redis_store::{
    Number,
    spend_counters::{FLOOR_TTL, LostSeed, SpendCounter, SpendCounters},
};

/// The spend a counter's row holds, `None` when it has no row or the read failed.
pub trait RowSpend: Send + Sync {
    fn stored<'a>(&'a self, counter: &'a SpendCounter) -> BoxFuture<'a, Option<f64>>;
}

/// Reads a counter's current spend from the store a budget reservation holds against, seeding and
/// repairing it from its row.
pub struct SpendReader {
    store: Arc<SpendCounters>,
    rows: Arc<dyn RowSpend>,
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
            None => self.seeded(counter, fallback_spend).await,
        };
        if max_budget.is_none_or(|max_budget| current >= max_budget) || fallback_spend <= current {
            return current;
        }
        match self.floor(counter).await {
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

    async fn seeded(&self, counter: &SpendCounter, fallback_spend: f64) -> f64 {
        let locks = self.store.lock_seeds(std::slice::from_ref(counter)).await;
        if locks.waited()
            && let Some(current) = self.cached(counter).await
        {
            return current;
        }
        let stored = match counter {
            SpendCounter::EndUser(_) => None,
            _ => self.rows.stored(counter).await,
        };
        match stored {
            Some(spend) => self
                .store
                .seed(counter, Number::Float(spend), LostSeed::Answer)
                .await
                .map_or(spend, Number::as_f64),
            None if fallback_spend > 0.0 => self
                .store
                .set_max(counter, Number::Float(fallback_spend))
                .await
                .ok()
                .flatten()
                .map_or(fallback_spend, Number::as_f64),
            None => fallback_spend,
        }
    }

    /// `_authoritative_floor_spend`: the row's spend, read at most once per [`FLOOR_TTL`]. A reset
    /// announced while the row was read wins over what the read found.
    async fn floor(&self, counter: &SpendCounter) -> Option<f64> {
        let kept = || self.store.floor(counter).ok().flatten().map(Number::as_f64);
        if let Some(floor) = kept() {
            return Some(floor);
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
        sync::atomic::{AtomicUsize, Ordering},
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
        spend: HashMap<String, f64>,
        reads: AtomicUsize,
    }

    impl Rows {
        fn holding(counter: &SpendCounter, spend: Option<f64>) -> Arc<Self> {
            Arc::new(Self {
                spend: spend
                    .map(|spend| (counter.key(), spend))
                    .into_iter()
                    .collect(),
                reads: AtomicUsize::new(0),
            })
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
                self.spend.get(&counter.key()).copied()
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
    #[case::from_its_row(team(), Some(4.0), 1.0, 4.0, Some(4.0))]
    #[case::from_the_plan_without_a_row(team(), None, 3.0, 3.0, Some(3.0))]
    #[case::not_at_all_with_nothing_spent(team(), None, 0.0, 0.0, None)]
    #[case::an_end_user_from_the_plan_whatever_its_row_holds(
        end_user(),
        Some(4.0),
        1.0,
        1.0,
        Some(1.0)
    )]
    #[tokio::test]
    async fn a_cold_counter_is_seeded_for_every_pod(
        #[future] server: RedisServer,
        #[case] counter: SpendCounter,
        #[case] row: Option<f64>,
        #[case] fallback_spend: f64,
        #[case] current: f64,
        #[case] seeded: Option<f64>,
    ) {
        let server = server.await;
        let reader = SpendReader::new(store(&server), Rows::holding(&counter, row));

        let read = reader.current(&counter, fallback_spend, None).await;

        assert_eq!((read, in_redis(&server, &counter).await), (current, seeded));
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
