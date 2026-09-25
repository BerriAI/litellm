use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_cache::{
    BaseCache, BatchCache, BatchEntry, BulkDeleteCache, ClaimCache, CounterCache, DeleteCache,
    DualCache, Error, ExactCacheContext, FlushCache, IncrementOperation, ReadPolicy,
    RemoteFailurePolicy, SetCache, TtlCache, WritePolicy,
};
use rstest::{fixture, rstest};

struct TestCache<V> {
    value: Mutex<Option<V>>,
    fail: bool,
}

impl<V> TestCache<V> {
    fn new(value: Option<V>, fail: bool) -> Self {
        Self {
            value: Mutex::new(value),
            fail,
        }
    }
}

impl<V> BaseCache for TestCache<V>
where
    V: Clone + Send + Sync + 'static,
{
    type Value = V;
    type Context = ExactCacheContext;

    fn get_ttl(&self, context: &Self::Context) -> Option<Duration> {
        context.ttl.or(Some(Duration::from_secs(60)))
    }

    fn set_cache(&self, _: &str, value: V, _: &ExactCacheContext) -> Result<(), Error> {
        *self.value.lock().unwrap() = Some(value);
        Ok(())
    }

    fn get_cache(&self, _: &str, _: &ExactCacheContext) -> Result<Option<V>, Error> {
        Ok(self.value.lock().unwrap().clone())
    }
}

impl<V> BatchCache for TestCache<V> where V: Clone + Send + Sync + 'static {}

impl<V> DeleteCache for TestCache<V>
where
    V: Clone + Send + Sync + 'static,
{
    fn delete_cache(&self, _: &str) -> Result<(), Error> {
        *self.value.lock().unwrap() = None;
        Ok(())
    }
}

impl<V> FlushCache for TestCache<V>
where
    V: Clone + Send + Sync + 'static,
{
    fn flush_cache(&self) -> Result<(), Error> {
        *self.value.lock().unwrap() = None;
        Ok(())
    }
}

impl CounterCache for TestCache<f64> {
    fn increment_cache(&self, _: &str, amount: f64, _: ExactCacheContext) -> Result<f64, Error> {
        if self.fail {
            return Err(Error::Unavailable);
        }
        let mut value = self.value.lock().unwrap();
        let incremented = value.unwrap_or_default() + amount;
        *value = Some(incremented);
        Ok(incremented)
    }
}

impl<V> ClaimCache for TestCache<V>
where
    V: Clone + PartialEq + Send + Sync + 'static,
{
    fn claim_cache(
        &self,
        _: &str,
        candidate: V,
        eligible: &[V],
        _: ExactCacheContext,
    ) -> Result<V, Error> {
        if self.fail {
            return Err(Error::Unavailable);
        }
        let mut value = self.value.lock().unwrap();
        let winner = match value.as_ref() {
            Some(existing) if eligible.is_empty() || eligible.contains(existing) => {
                existing.clone()
            }
            _ => candidate,
        };
        *value = Some(winner.clone());
        Ok(winner)
    }
}

#[rstest]
fn failed_l2_increment_leaves_l1_unchanged() {
    let l1 = Arc::new(TestCache::new(Some(10.0), false));
    let cache = DualCache::new(l1.clone(), Arc::new(TestCache::new(Some(20.0), true)));

    assert_eq!(
        cache.increment_cache("counter", 2.0, ExactCacheContext::default()),
        Err(Error::Unavailable)
    );
    assert_eq!(
        l1.get_cache("counter", &ExactCacheContext::default())
            .unwrap(),
        Some(10.0)
    );
}

#[rstest]
fn claim_uses_l1_fallback_without_overwriting_an_eligible_winner() {
    let l1 = Arc::new(TestCache::new(Some("first".to_string()), false));
    let cache = DualCache::new(l1, Arc::new(TestCache::new(None, true)))
        .with_remote_failure_policy(RemoteFailurePolicy::UseLocal);

    assert_eq!(
        cache
            .claim_cache(
                "affinity",
                "second".into(),
                &["first".into(), "second".into()],
                ExactCacheContext {
                    ttl: Some(Duration::from_secs(60)),
                },
            )
            .unwrap(),
        "first"
    );
}

struct SyncPanics(TestCache<String>);

impl BaseCache for SyncPanics {
    type Value = String;
    type Context = ExactCacheContext;

    fn get_ttl(&self, context: &Self::Context) -> Option<Duration> {
        self.0.get_ttl(context)
    }

    fn set_cache(&self, _: &str, _: String, _: &ExactCacheContext) -> Result<(), Error> {
        panic!("sync L2 write on an async path")
    }

    fn get_cache(&self, _: &str, _: &ExactCacheContext) -> Result<Option<String>, Error> {
        panic!("sync L2 read on an async path")
    }

    async fn async_set_cache(
        &self,
        key: &str,
        value: String,
        context: ExactCacheContext,
    ) -> Result<(), Error> {
        self.0.set_cache(key, value, &context)
    }

    async fn async_get_cache(
        &self,
        key: &str,
        context: &ExactCacheContext,
    ) -> Result<Option<String>, Error> {
        self.0.get_cache(key, context)
    }

    async fn async_set_cache_pipeline(
        &self,
        cache_list: Vec<(String, String)>,
        context: ExactCacheContext,
    ) -> Result<(), Error> {
        for (key, value) in cache_list {
            self.0.set_cache(&key, value, &context)?;
        }
        Ok(())
    }
}

impl BatchCache for SyncPanics {
    async fn async_batch_get_cache(
        &self,
        keys: Vec<String>,
        context: ExactCacheContext,
    ) -> Result<Vec<BatchEntry<String>>, Error> {
        assert_eq!(keys, ["missing"]);
        Ok(vec![match self.0.get_cache("missing", &context)? {
            Some(value) => BatchEntry::Hit(value),
            None => BatchEntry::Miss,
        }])
    }
}

impl DeleteCache for SyncPanics {
    fn delete_cache(&self, _: &str) -> Result<(), Error> {
        panic!("sync L2 delete on an async path")
    }

    async fn async_delete_cache(&self, key: &str) -> Result<(), Error> {
        self.0.delete_cache(key)
    }
}

impl FlushCache for SyncPanics {
    fn flush_cache(&self) -> Result<(), Error> {
        panic!("sync L2 flush on an async path")
    }
}

#[rstest]
#[tokio::test]
async fn async_operations_use_the_async_l2_methods() {
    let l1 = Arc::new(TestCache::new(None, false));
    let cache = DualCache::new(
        l1.clone(),
        Arc::new(SyncPanics(TestCache::new(
            Some("remote".to_string()),
            false,
        ))),
    );
    let context = ExactCacheContext::default();

    assert_eq!(
        cache.async_get_cache("missing", &context).await.unwrap(),
        Some("remote".into())
    );
    assert_eq!(
        l1.get_cache("missing", &context).unwrap(),
        Some("remote".into())
    );

    l1.delete_cache("missing").unwrap();
    assert_eq!(
        cache
            .async_batch_get_cache(vec!["missing".into()], context.clone())
            .await
            .unwrap(),
        [BatchEntry::Hit("remote".to_string())]
    );
    cache
        .async_set_cache("missing", "written".into(), context.clone())
        .await
        .unwrap();
    cache
        .async_set_cache_pipeline(vec![("missing".into(), "piped".into())], context.clone())
        .await
        .unwrap();
    cache.async_delete_cache("missing").await.unwrap();
    assert_eq!(
        cache.async_get_cache("missing", &context).await.unwrap(),
        None
    );
}

struct Unavailable;

impl BaseCache for Unavailable {
    type Value = String;
    type Context = ExactCacheContext;

    fn get_ttl(&self, context: &Self::Context) -> Option<Duration> {
        context.ttl
    }

    fn set_cache(&self, _: &str, _: String, _: &ExactCacheContext) -> Result<(), Error> {
        Err(Error::Unavailable)
    }

    fn get_cache(&self, _: &str, _: &ExactCacheContext) -> Result<Option<String>, Error> {
        Err(Error::Unavailable)
    }
}

impl BatchCache for Unavailable {}

impl DeleteCache for Unavailable {
    fn delete_cache(&self, _: &str) -> Result<(), Error> {
        Err(Error::Unavailable)
    }
}

impl FlushCache for Unavailable {
    fn flush_cache(&self) -> Result<(), Error> {
        Err(Error::Unavailable)
    }
}

impl ClaimCache for Unavailable {
    fn claim_cache(
        &self,
        _: &str,
        _: String,
        _: &[String],
        _: ExactCacheContext,
    ) -> Result<String, Error> {
        Err(Error::InvalidEntry)
    }
}

#[rstest]
fn remote_failure_policy_selects_propagation_or_the_local_tier() {
    let context = ExactCacheContext::default();
    let strict = DualCache::new(Arc::new(TestCache::new(None, false)), Arc::new(Unavailable));
    assert_eq!(
        strict.set_cache("key", "value".into(), &context),
        Err(Error::Unavailable)
    );
    assert_eq!(strict.get_cache("key", &context), Err(Error::Unavailable));

    let l1 = Arc::new(TestCache::new(None, false));
    let degraded = DualCache::new(l1.clone(), Arc::new(Unavailable))
        .with_remote_failure_policy(RemoteFailurePolicy::UseLocal);
    assert_eq!(degraded.get_cache("key", &context), Ok(None));
    degraded.set_cache("key", "value".into(), &context).unwrap();
    assert_eq!(
        degraded.get_cache("key", &context),
        Ok(Some("value".into()))
    );
    degraded.delete_cache("key").unwrap();
    assert_eq!(l1.get_cache("key", &context), Ok(None));
}

#[rstest]
fn claim_fallback_does_not_hide_non_availability_errors() {
    let cache = DualCache::new(
        Arc::new(TestCache::new(Some("first".to_string()), false)),
        Arc::new(Unavailable),
    )
    .with_remote_failure_policy(RemoteFailurePolicy::UseLocal);
    assert_eq!(
        cache.claim_cache(
            "affinity",
            "second".into(),
            &[],
            ExactCacheContext::default()
        ),
        Err(Error::InvalidEntry)
    );
}

#[rstest]
fn local_only_policies_never_touch_l2() {
    let l2 = Arc::new(TestCache::new(Some("remote".to_string()), false));
    let cache = DualCache::new(Arc::new(TestCache::new(None, false)), l2.clone())
        .with_read_policy(ReadPolicy::LocalOnly)
        .with_write_policy(WritePolicy::LocalOnly);
    let context = ExactCacheContext::default();

    assert_eq!(cache.get_cache("key", &context), Ok(None));
    cache.set_cache("key", "local".into(), &context).unwrap();
    assert_eq!(l2.get_cache("key", &context), Ok(Some("remote".into())));
}

type Log = Arc<Mutex<Vec<String>>>;
type StoredSet = (Vec<String>, Option<Duration>);

/// A keyed tier that logs every call, so tests can assert which tier ran and in what order.
struct Tier {
    name: &'static str,
    log: Log,
    fail: bool,
    counters: Mutex<HashMap<String, (f64, ExactCacheContext)>>,
    sets: Mutex<HashMap<String, StoredSet>>,
    ttls: HashMap<String, Duration>,
}

impl Tier {
    fn new(name: &'static str, log: &Log) -> Self {
        Self {
            name,
            log: log.clone(),
            fail: false,
            counters: Mutex::default(),
            sets: Mutex::default(),
            ttls: HashMap::new(),
        }
    }

    fn failing(self) -> Self {
        Self { fail: true, ..self }
    }

    fn with_counter(self, key: &str, value: f64) -> Self {
        self.counters
            .lock()
            .unwrap()
            .insert(key.into(), (value, ExactCacheContext::default()));
        self
    }

    fn with_ttl(mut self, key: &str, seconds: u64) -> Self {
        self.ttls.insert(key.into(), Duration::from_secs(seconds));
        self
    }

    fn record(&self, event: String) {
        self.log
            .lock()
            .unwrap()
            .push(format!("{} {event}", self.name));
    }

    fn counter(&self, key: &str) -> Option<(f64, ExactCacheContext)> {
        self.counters.lock().unwrap().get(key).cloned()
    }

    fn check(&self) -> Result<(), Error> {
        if self.fail {
            return Err(Error::Unavailable);
        }
        Ok(())
    }
}

impl BaseCache for Tier {
    type Value = f64;
    type Context = ExactCacheContext;

    fn get_ttl(&self, context: &Self::Context) -> Option<Duration> {
        context.ttl
    }

    fn set_cache(&self, key: &str, value: f64, context: &ExactCacheContext) -> Result<(), Error> {
        self.check()?;
        self.record(format!("set {key}={value}"));
        self.counters
            .lock()
            .unwrap()
            .insert(key.into(), (value, context.clone()));
        Ok(())
    }

    fn get_cache(&self, key: &str, _: &ExactCacheContext) -> Result<Option<f64>, Error> {
        Ok(self.counter(key).map(|(value, _)| value))
    }
}

impl CounterCache for Tier {
    fn increment_cache(
        &self,
        key: &str,
        amount: f64,
        context: ExactCacheContext,
    ) -> Result<f64, Error> {
        self.check()?;
        let mut counters = self.counters.lock().unwrap();
        let value = counters.get(key).map_or(0.0, |(value, _)| *value) + amount;
        counters.insert(key.into(), (value, context));
        Ok(value)
    }

    async fn async_increment(
        &self,
        key: &str,
        amount: f64,
        context: ExactCacheContext,
        refresh_ttl: bool,
    ) -> Result<f64, Error> {
        self.record(format!(
            "increment {key}+{amount} refresh_ttl={refresh_ttl}"
        ));
        self.increment_cache(key, amount, context)
    }

    async fn async_increment_pipeline(
        &self,
        operations: Vec<IncrementOperation>,
    ) -> Result<Vec<f64>, Error> {
        let keys = operations
            .iter()
            .map(|operation| operation.key.as_str())
            .collect::<Vec<_>>();
        self.record(format!("pipeline {}", keys.join(",")));
        operations
            .into_iter()
            .map(|operation| {
                self.increment_cache(
                    &operation.key,
                    operation.amount,
                    ExactCacheContext { ttl: operation.ttl },
                )
            })
            .collect()
    }
}

impl DeleteCache for Tier {
    fn delete_cache(&self, key: &str) -> Result<(), Error> {
        self.check()?;
        self.record(format!("delete {key}"));
        self.counters.lock().unwrap().remove(key);
        Ok(())
    }
}

impl BulkDeleteCache for Tier {
    async fn delete_cache_keys(&self, keys: Vec<String>) -> Result<usize, Error> {
        self.check()?;
        self.record(format!("delete_keys {}", keys.join(",")));
        let mut counters = self.counters.lock().unwrap();
        Ok(keys
            .iter()
            .filter(|key| counters.remove(key.as_str()).is_some())
            .count())
    }
}

impl SetCache for Tier {
    type SetValue = String;
    type SetResult = ();

    async fn async_set_cache_sadd(
        &self,
        key: &str,
        values: Vec<String>,
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        self.check()?;
        self.record(format!("sadd {key} {}", values.join(",")));
        self.sets.lock().unwrap().insert(key.into(), (values, ttl));
        Ok(())
    }
}

impl TtlCache for Tier {
    async fn async_get_ttl(&self, key: &str) -> Result<Option<Duration>, Error> {
        self.check()?;
        self.record(format!("ttl {key}"));
        Ok(self.ttls.get(key).copied())
    }
}

#[fixture]
fn log() -> Log {
    Log::default()
}

fn events(log: &Log) -> Vec<String> {
    log.lock().unwrap().clone()
}

fn seconds(ttl: u64) -> ExactCacheContext {
    ExactCacheContext {
        ttl: Some(Duration::from_secs(ttl)),
    }
}

#[rstest]
#[case::window_semantics(false)]
#[case::refresh_on_every_write(true)]
#[tokio::test]
async fn async_increment_passes_refresh_ttl_to_l2_and_stores_its_result_locally(
    log: Log,
    #[case] refresh_ttl: bool,
) {
    let l1 = Arc::new(Tier::new("l1", &log).with_counter("counter", 1.0));
    let l2 = Arc::new(Tier::new("l2", &log).with_counter("counter", 10.0));
    let cache = DualCache::new(l1.clone(), l2.clone());

    assert_eq!(
        cache
            .async_increment("counter", 2.0, seconds(30), refresh_ttl)
            .await,
        Ok(12.0)
    );
    assert_eq!(
        events(&log),
        [
            format!("l2 increment counter+2 refresh_ttl={refresh_ttl}"),
            "l1 set counter=12".into(),
        ]
    );
    assert_eq!(l1.counter("counter"), Some((12.0, seconds(30))));
    assert_eq!(l2.counter("counter"), Some((12.0, seconds(30))));
}

#[rstest]
#[tokio::test]
async fn failed_async_l2_increment_leaves_l1_unchanged(log: Log) {
    let l1 = Arc::new(Tier::new("l1", &log).with_counter("counter", 1.0));
    let cache = DualCache::new(l1.clone(), Arc::new(Tier::new("l2", &log).failing()))
        .with_remote_failure_policy(RemoteFailurePolicy::UseLocal);

    assert_eq!(
        cache
            .async_increment("counter", 2.0, seconds(30), true)
            .await,
        Err(Error::Unavailable)
    );
    assert_eq!(
        l1.counter("counter"),
        Some((1.0, ExactCacheContext::default()))
    );
}

#[rstest]
#[case::empty(Vec::new(), Vec::new())]
#[case::one_key(vec![("a", 1.0, Some(10))], vec![6.0])]
#[case::repeated_and_mixed_ttls(
    vec![("a", 1.0, Some(10)), ("b", 2.0, None), ("a", 3.0, Some(20))],
    vec![6.0, 2.0, 9.0],
)]
#[tokio::test]
async fn async_increment_pipeline_runs_l2_first_and_l1_takes_each_remote_result(
    log: Log,
    #[case] operations: Vec<(&str, f64, Option<u64>)>,
    #[case] expected: Vec<f64>,
) {
    let operations = operations
        .into_iter()
        .map(|(key, amount, ttl)| IncrementOperation {
            key: key.into(),
            amount,
            ttl: ttl.map(Duration::from_secs),
        })
        .collect::<Vec<_>>();
    let l1 = Arc::new(Tier::new("l1", &log).with_counter("a", 100.0));
    let l2 = Arc::new(Tier::new("l2", &log).with_counter("a", 5.0));
    let cache = DualCache::new(l1.clone(), l2);

    assert_eq!(
        cache.async_increment_pipeline(operations.clone()).await,
        Ok(expected.clone())
    );
    let keys = operations
        .iter()
        .map(|operation| operation.key.as_str())
        .collect::<Vec<_>>();
    let mut expected_events = vec![format!("l2 pipeline {}", keys.join(","))];
    expected_events.extend(
        operations
            .iter()
            .zip(&expected)
            .map(|(operation, value)| format!("l1 set {}={value}", operation.key)),
    );
    assert_eq!(events(&log), expected_events);
    if let Some((operation, value)) = operations.iter().zip(&expected).next_back() {
        assert_eq!(
            l1.counter(&operation.key),
            Some((*value, ExactCacheContext { ttl: operation.ttl }))
        );
    }
}

#[rstest]
#[tokio::test]
async fn failed_l2_increment_pipeline_leaves_l1_unchanged(log: Log) {
    let l1 = Arc::new(Tier::new("l1", &log).with_counter("a", 1.0));
    let cache = DualCache::new(l1.clone(), Arc::new(Tier::new("l2", &log).failing()));

    assert_eq!(
        cache
            .async_increment_pipeline(vec![IncrementOperation {
                key: "a".into(),
                amount: 1.0,
                ttl: None,
            }])
            .await,
        Err(Error::Unavailable)
    );
    assert_eq!(events(&log), ["l2 pipeline a"]);
    assert_eq!(l1.counter("a"), Some((1.0, ExactCacheContext::default())));
}

#[rstest]
#[case::both_tiers(WritePolicy::Both, &["l1 sadd members a,b", "l2 sadd members a,b"])]
#[case::local_only(WritePolicy::LocalOnly, &["l1 sadd members a,b"])]
#[tokio::test]
async fn set_add_writes_locally_then_remotely_unless_local_only(
    log: Log,
    #[case] write_policy: WritePolicy,
    #[case] expected: &[&str],
) {
    let l1 = Arc::new(Tier::new("l1", &log));
    let l2 = Arc::new(Tier::new("l2", &log));
    let cache = DualCache::new(l1.clone(), l2.clone()).with_write_policy(write_policy);
    let ttl = Some(Duration::from_secs(45));

    cache
        .async_set_cache_sadd("members", vec!["a".into(), "b".into()], ttl)
        .await
        .unwrap();
    assert_eq!(events(&log), expected);
    let stored = Some((vec!["a".to_string(), "b".to_string()], ttl));
    assert_eq!(l1.sets.lock().unwrap().get("members").cloned(), stored);
    assert_eq!(
        l2.sets.lock().unwrap().get("members").cloned(),
        stored.filter(|_| write_policy == WritePolicy::Both)
    );
}

#[rstest]
#[tokio::test]
async fn failed_local_set_add_never_reaches_l2(log: Log) {
    let cache = DualCache::new(
        Arc::new(Tier::new("l1", &log).failing()),
        Arc::new(Tier::new("l2", &log)),
    );
    assert_eq!(
        cache
            .async_set_cache_sadd("members", vec!["a".into()], None)
            .await,
        Err(Error::Unavailable)
    );
    assert!(events(&log).is_empty());
}

#[rstest]
#[case::empty(None, &[], &[])]
#[case::default_batch_size(None, &["a", "b", "c"], &["a,b,c"])]
#[case::chunked(Some(2), &["a", "b", "c", "d", "e"], &["a,b", "c,d", "e"])]
#[case::exact_chunks(Some(2), &["a", "b", "c", "d"], &["a,b", "c,d"])]
#[case::zero_means_one_per_chunk(Some(0), &["a", "b"], &["a", "b"])]
#[tokio::test]
async fn bulk_delete_removes_every_key_locally_then_remotely_in_chunks(
    log: Log,
    #[case] batch_size: Option<usize>,
    #[case] keys: &[&str],
    #[case] chunks: &[&str],
) {
    let l1 = Tier::new("l1", &log);
    let l2 = Tier::new("l2", &log);
    for key in keys.iter().step_by(2) {
        l2.counters
            .lock()
            .unwrap()
            .insert((*key).into(), (1.0, ExactCacheContext::default()));
    }
    let l1 = Arc::new(l1);
    let mut cache = DualCache::new(l1, Arc::new(l2));
    if let Some(batch_size) = batch_size {
        cache = cache.with_delete_batch_size(batch_size);
    }

    assert_eq!(
        cache
            .delete_cache_keys(keys.iter().map(|key| (*key).into()).collect())
            .await,
        Ok(keys.len().div_ceil(2))
    );
    let expected = keys
        .iter()
        .map(|key| format!("l1 delete {key}"))
        .chain(chunks.iter().map(|chunk| format!("l2 delete_keys {chunk}")))
        .collect::<Vec<_>>();
    assert_eq!(events(&log), expected);
}

#[rstest]
#[tokio::test]
async fn bulk_delete_stops_before_l2_when_the_local_delete_fails(log: Log) {
    let cache = DualCache::new(
        Arc::new(Tier::new("l1", &log).failing()),
        Arc::new(Tier::new("l2", &log)),
    );
    assert_eq!(
        cache.delete_cache_keys(vec!["a".into()]).await,
        Err(Error::Unavailable)
    );
    assert!(events(&log).is_empty());
}

#[rstest]
#[case::local_hit("both", Some(10), &["l1 ttl both"])]
#[case::remote_fallback("remote", Some(20), &["l1 ttl remote", "l2 ttl remote"])]
#[case::missing_everywhere("missing", None, &["l1 ttl missing", "l2 ttl missing"])]
#[tokio::test]
async fn ttl_reads_local_then_remote(
    log: Log,
    #[case] key: &str,
    #[case] expected: Option<u64>,
    #[case] expected_events: &[&str],
) {
    let cache = DualCache::new(
        Arc::new(Tier::new("l1", &log).with_ttl("both", 10)),
        Arc::new(
            Tier::new("l2", &log)
                .with_ttl("both", 99)
                .with_ttl("remote", 20),
        ),
    );
    assert_eq!(
        cache.async_get_ttl(key).await,
        Ok(expected.map(Duration::from_secs))
    );
    assert_eq!(events(&log), expected_events);
}

/// Python `local_only=True`: the increment and the pipeline stay on the local tier.
#[rstest]
#[tokio::test]
async fn local_only_writes_increment_the_local_tier_alone(log: Log) {
    let l1 = Arc::new(Tier::new("l1", &log).with_counter("a", 1.0));
    let l2 = Arc::new(Tier::new("l2", &log).with_counter("a", 10.0));
    let cache = DualCache::new(l1.clone(), l2.clone()).with_write_policy(WritePolicy::LocalOnly);

    assert_eq!(cache.increment_cache("a", 2.0, seconds(30)), Ok(3.0));
    assert_eq!(
        cache.async_increment("a", 1.0, seconds(30), true).await,
        Ok(4.0)
    );
    let operations = vec![
        IncrementOperation {
            key: "a".into(),
            amount: 1.0,
            ttl: Some(Duration::from_secs(10)),
        },
        IncrementOperation {
            key: "b".into(),
            amount: 2.0,
            ttl: None,
        },
    ];
    assert_eq!(
        cache.async_increment_pipeline(operations).await,
        Ok(vec![5.0, 2.0])
    );
    assert_eq!(
        events(&log),
        ["l1 set a=3", "l1 set a=4", "l1 set a=5", "l1 set b=2"]
    );
    assert_eq!(l2.counter("a"), Some((10.0, ExactCacheContext::default())));
}
