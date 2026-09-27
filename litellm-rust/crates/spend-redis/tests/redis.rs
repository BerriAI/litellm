//! Runs against a real Redis. Set `LITELLM_TEST_REDIS_URL` (for example `redis://127.0.0.1:6379`)
//! to run them; without it every test returns early.

use std::time::Duration;

use litellm_spend::{Batch, BatchId, Buffer, Cost, CounterKey, Counters, EntityKey, Store, Totals};
use litellm_spend_redis::{
    BatchCodec, BufferSettings, CounterNaming, Error, PythonCounterNaming, PythonFormatError,
    PythonTotalsCodec, RedisBuffer, RedisCounters,
};
use litellm_spend_testing::{self as contract, Ledger};
use redis::{AsyncCommands, aio::MultiplexedConnection};

const CLAIM_TTL: Duration = Duration::from_millis(300);

struct IsolatedList(String);

impl BatchCodec<EntityKey, Cost> for IsolatedList {
    type Error = PythonFormatError;

    fn list(&self) -> &str {
        &self.0
    }

    fn encode(&self, batch: &Totals) -> Result<String, Self::Error> {
        PythonTotalsCodec.encode(batch)
    }

    fn decode(&self, blob: &str) -> Result<Totals, Self::Error> {
        PythonTotalsCodec.decode(blob)
    }
}

struct Shared {
    connection: MultiplexedConnection,
    list: String,
    buffer: RedisBuffer<MultiplexedConnection, IsolatedList>,
}

impl Store<EntityKey, Cost> for Shared {
    type Error = Error<PythonFormatError>;

    async fn commit(
        &self,
        claimed: &litellm_spend::Claimed<EntityKey, Cost>,
    ) -> Result<(), Self::Error> {
        self.buffer.commit(claimed).await
    }
}

impl Ledger for Shared {
    async fn total(&self, entity: &EntityKey) -> Option<Cost> {
        let blobs: Vec<String> = self
            .connection
            .clone()
            .lrange(&self.list, 0, -1)
            .await
            .unwrap();
        blobs
            .iter()
            .map(|blob| PythonTotalsCodec.decode(blob).unwrap())
            .fold(Batch::default(), Batch::merge)
            .get(entity)
            .copied()
    }
}

async fn connection() -> Option<MultiplexedConnection> {
    let url = std::env::var("LITELLM_TEST_REDIS_URL").ok()?;
    Some(
        redis::Client::open(url)
            .unwrap()
            .get_multiplexed_async_connection()
            .await
            .unwrap(),
    )
}

async fn shared() -> Option<Shared> {
    let connection = connection().await?;
    let list = format!("test_spend_buffer_{}", BatchId::random().as_uuid());
    Some(Shared {
        buffer: RedisBuffer::new(
            connection.clone(),
            IsolatedList(list.clone()),
            BufferSettings {
                claim_ttl: CLAIM_TTL,
                remember_applied_for: Duration::from_secs(60),
                blobs_per_claim: 100,
            },
        ),
        connection,
        list,
    })
}

async fn expire() {
    tokio::time::sleep(CLAIM_TTL + Duration::from_millis(100)).await;
}

fn team(name: &str) -> EntityKey {
    EntityKey::Team(name.to_owned())
}

#[tokio::test]
async fn redis_buffer_hides_a_claimed_batch_from_other_claimers() {
    let Some(shared) = shared().await else { return };
    contract::a_claimed_batch_is_hidden_from_other_claimers(&shared.buffer, "").await;
}

#[tokio::test]
async fn redis_buffer_returns_an_unacked_batch_unchanged_once_its_claim_expires() {
    let Some(shared) = shared().await else { return };
    contract::an_unacked_batch_returns_unchanged_once_its_claim_expires(&shared.buffer, "", expire)
        .await;
}

#[tokio::test]
async fn redis_buffer_returns_a_released_batch_at_once_with_its_id() {
    let Some(shared) = shared().await else { return };
    contract::a_released_batch_returns_at_once_with_its_id(&shared.buffer, "").await;
}

#[tokio::test]
async fn redis_buffer_forgets_an_acked_batch_and_tolerates_a_second_ack() {
    let Some(shared) = shared().await else { return };
    contract::an_acked_batch_is_gone_and_a_second_ack_is_harmless(&shared.buffer, "", expire).await;
}

#[tokio::test]
async fn redis_buffer_as_a_store_counts_a_batch_committed_twice_once() {
    let Some(shared) = shared().await else { return };
    contract::a_batch_committed_twice_counts_once(&shared, "").await;
}

#[tokio::test]
async fn redis_buffer_as_a_store_counts_equal_batches_with_different_ids_twice() {
    let Some(shared) = shared().await else { return };
    contract::equal_batches_with_different_ids_both_count(&shared, "").await;
}

#[tokio::test]
async fn a_claim_over_max_leaves_the_rest_for_the_next_claim_and_keeps_its_own_contents_on_redelivery()
 {
    let Some(shared) = shared().await else { return };
    let everything = Totals::from_entries([
        (team("a"), Cost(0.25)),
        (team("b"), Cost(0.5)),
        (team("c"), Cost(1.0)),
    ]);
    shared.buffer.push(everything.clone()).await.unwrap();

    let first = shared.buffer.claim(2).await.unwrap().unwrap();
    let rest = shared.buffer.claim(usize::MAX).await.unwrap().unwrap();
    expire().await;
    let redelivered = shared.buffer.claim(usize::MAX).await.unwrap().unwrap();

    assert_eq!(first.batch().len(), 2);
    assert_eq!(
        first.batch().clone().merge(rest.batch().clone()),
        everything
    );
    assert!(
        redelivered == first || redelivered == rest,
        "an expired claim must come back with the id and contents it was handed out with"
    );
}

#[tokio::test]
async fn a_blob_python_pushed_is_claimed_as_typed_spend() {
    let Some(mut shared) = shared().await else {
        return;
    };
    let _: () = shared
        .connection
        .rpush(
            &shared.list,
            vec![
                r#"{"team_list_transactions": {"t": 0.25}, "key_list_transactions": {"k": 0.5}}"#,
                r#"{"team_list_transactions": {"t": 1.0}}"#,
            ],
        )
        .await
        .unwrap();

    let claimed = shared.buffer.claim(usize::MAX).await.unwrap().unwrap();

    assert_eq!(
        claimed.batch(),
        &Totals::from_entries([
            (team("t"), Cost(1.25)),
            (EntityKey::Key("k".to_owned()), Cost(0.5)),
        ])
    );
}

fn counter(entity: EntityKey) -> CounterKey {
    CounterKey::lifetime(entity)
}

#[tokio::test]
async fn counters_accumulate_across_calls_and_expire_after_their_ttl() {
    let Some(connection) = connection().await else {
        return;
    };
    let counters = RedisCounters::new(connection, PythonCounterNaming, Some(CLAIM_TTL));
    let key = counter(team(&format!("counter-{}", BatchId::random().as_uuid())));
    let other = counter(EntityKey::Key(format!(
        "counter-{}",
        BatchId::random().as_uuid()
    )));

    assert_eq!(counters.current(&key).await.unwrap(), None);
    assert_eq!(
        counters
            .add(&[(key.clone(), Cost(0.25)), (other.clone(), Cost(2.0))])
            .await
            .unwrap(),
        vec![Cost(0.25), Cost(2.0)]
    );
    assert_eq!(
        counters.add(&[(key.clone(), Cost(0.5))]).await.unwrap(),
        vec![Cost(0.75)]
    );
    assert_eq!(counters.current(&key).await.unwrap(), Some(Cost(0.75)));

    expire().await;
    assert_eq!(counters.current(&key).await.unwrap(), None);
}

#[tokio::test]
async fn an_increment_holding_an_unnamed_counter_changes_nothing() {
    let Some(connection) = connection().await else {
        return;
    };
    let counters = RedisCounters::new(connection, PythonCounterNaming, None);
    let named = counter(team(&format!("counter-{}", BatchId::random().as_uuid())));
    let unnamed = counter(EntityKey::Agent("a".to_owned()));
    assert_eq!(PythonCounterNaming.counter(&unnamed), None);

    let refused = counters
        .add(&[(named.clone(), Cost(1.0)), (unnamed, Cost(1.0))])
        .await;

    assert!(matches!(refused, Err(Error::Unnamed(_))));
    assert_eq!(counters.current(&named).await.unwrap(), None);
}
