mod support;

use std::time::{Duration, Instant};

use futures_util::{StreamExt, stream::BoxStream};
use litellm_circuit_breaker::{Breaker, Config, Failure, State};
use litellm_python_compat::Value;
use litellm_redis_client::Operation;
use litellm_redis_store::{
    Number,
    auth_cache::{AuthCache, AuthObject, Settings},
    auth_cache_invalidation::{self, AuthCacheInvalidation, Invalidation, channel},
    spend_counters::{SpendCounter, SpendCounters},
};
use litellm_redis_testing::{RedisDeployment, RedisServer, TcpProxy, Topology, eventually};
use rstest::rstest;
use support::{connect, connect_guarded, connect_observed, connect_port};
use tokio::runtime::Handle;

fn set(cache_key: &str, value: Number, ttl: Option<Number>) -> Invalidation {
    Invalidation::Set {
        cache_key: cache_key.to_owned(),
        value,
        ttl,
    }
}

fn evict(cache_key: &str) -> Invalidation {
    Invalidation::Evict(cache_key.to_owned())
}

/// The payloads `tests/unit/redis/stores/test_auth_cache_invalidation.py` checks against Python's
/// own `json.dumps`.
#[rstest]
#[case(evict("team_id:t1"), r#"{"cache_key": "team_id:t1"}"#)]
#[case(evict("t\u{e9}"), r#"{"cache_key": "t\u00e9"}"#)]
#[case(
    set("spend:key:abc", Number::Float(1.5), Some(Number::Int(60))),
    r#"{"cache_key": "spend:key:abc", "new_value": 1.5, "ttl": 60}"#
)]
#[case(
    set(
        "spend_db_floor:spend:team:t",
        Number::Float(0.30000000000000004),
        Some(Number::Int(5))
    ),
    r#"{"cache_key": "spend_db_floor:spend:team:t", "new_value": 0.30000000000000004, "ttl": 5}"#
)]
#[case(
    set("spend:user:u", Number::Float(0.0), None),
    r#"{"cache_key": "spend:user:u", "new_value": 0.0}"#
)]
fn a_message_is_mains_json(#[case] invalidation: Invalidation, #[case] payload: &str) {
    assert_eq!(invalidation.encode(), payload);
}

#[rstest]
#[case(br#"{"cache_key": "team_id:t1"}"#.as_slice(), Some(evict("team_id:t1")))]
#[case(br#"{"cache_key": "k", "ttl": 60}"#.as_slice(), Some(evict("k")))]
#[case(
    br#"{"cache_key": "k", "new_value": 2, "ttl": 60}"#.as_slice(),
    Some(set("k", Number::Float(2.0), Some(Number::Float(60.0))))
)]
#[case(br#"{"cache_key": "k", "new_value": true}"#.as_slice(), Some(evict("k")))]
#[case(br#"{"cache_key": "k", "new_value": "1.5"}"#.as_slice(), Some(evict("k")))]
#[case(
    br#"{"cache_key": "k", "new_value": 1.5, "ttl": "60"}"#.as_slice(),
    Some(set("k", Number::Float(1.5), None))
)]
#[case(b"{\"cache_key\": \"k\xff\"}".as_slice(), Some(evict("k\u{fffd}")))]
#[case(br#"{"cache_key": 7}"#.as_slice(), None)]
#[case(br#"["team_id:t1"]"#.as_slice(), None)]
#[case(b"not json".as_slice(), None)]
fn a_message_reads_as_mains_subscriber_reads_it(
    #[case] payload: &[u8],
    #[case] expected: Option<Invalidation>,
) {
    assert_eq!(Invalidation::decode(payload), expected);
}

#[rstest]
#[case(None, "litellm_proxy.auth_cache_invalidation")]
#[case(Some("prod"), "prod:litellm_proxy.auth_cache_invalidation")]
#[case(Some(""), ":litellm_proxy.auth_cache_invalidation")]
fn the_channel_carries_the_namespace(#[case] namespace: Option<&str>, #[case] expected: &str) {
    assert_eq!(channel(namespace), expected);
}

#[rstest]
#[tokio::test]
async fn every_worker_receives_what_one_publishes(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let publisher = AuthCacheInvalidation::new(connect(&deployment), None, Handle::current());
    let subscriber = AuthCacheInvalidation::new(connect(&deployment), None, Handle::current());
    let mut received = subscriber.subscribe();
    let sent = set("spend:key:abc", Number::Float(1.5), Some(Number::Int(60)));

    let first = eventually(async || {
        assert!(publisher.publish(&sent));
        tokio::time::timeout(Duration::from_millis(200), received.next())
            .await
            .ok()
            .flatten()
    })
    .await;
    publisher
        .broadcast_evictions(&["team_id:t1".to_owned(), "a".repeat(64)])
        .await;

    assert_eq!(
        first,
        set(
            "spend:key:abc",
            Number::Float(1.5),
            Some(Number::Float(60.0))
        )
    );
    let mut evicted = Vec::new();
    while evicted.len() < 2 {
        match received.next().await.unwrap() {
            Invalidation::Evict(cache_key) => evicted.push(cache_key),
            Invalidation::Set { .. } => {}
        }
    }
    assert_eq!(evicted, ["team_id:t1".to_owned(), "a".repeat(64)]);
}

async fn arrived<T>(received: &mut BoxStream<'static, T>) -> Option<T> {
    tokio::time::timeout(Duration::from_millis(200), received.next())
        .await
        .ok()
        .flatten()
}

// `Cache` assigns `cache_params.namespace` to the `RedisCache` after building it, so the native
// client carries none, while main read the channel's prefix off the `RedisCache`
#[tokio::test]
async fn a_namespace_the_client_does_not_carry_prefixes_the_channel_on_the_wire() {
    const PREFIXED: &str = "prod:litellm_proxy.auth_cache_invalidation";
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let client = connect(&deployment);
    let invalidation = AuthCacheInvalidation::new(client.clone(), Some("prod"), Handle::current());
    let mut on_the_wire = client.subscribe(PREFIXED);
    let mut decoded = invalidation.subscribe();

    let published = eventually(async || {
        assert!(invalidation.publish(&evict("team_id:t1")));
        arrived(&mut on_the_wire).await
    })
    .await;
    invalidation
        .broadcast_evictions(&["user-1".to_owned()])
        .await;
    let broadcast = eventually(async || {
        arrived(&mut on_the_wire)
            .await
            .filter(|payload| payload != &published)
    })
    .await;
    let from_another_worker = eventually(async || {
        client
            .publish(PREFIXED, &[br#"{"cache_key": "k"}"#.to_vec()])
            .await
            .ok()?;
        arrived(&mut decoded)
            .await
            .filter(|invalidation| *invalidation == evict("k"))
    })
    .await;

    assert_eq!(published, br#"{"cache_key": "team_id:t1"}"#);
    assert_eq!(broadcast, br#"{"cache_key": "user-1"}"#);
    assert_eq!(from_another_worker, evict("k"));
}

// The breaker opens during a Redis outage and stays open for its recovery timeout. main published
// through redis-py's pub/sub client, outside the breaker, so a block reached the other workers as
// soon as Redis was back
#[tokio::test]
async fn an_open_breaker_does_not_hold_back_an_invalidation() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let breaker = Breaker::new(
        Config {
            failure_threshold: 1,
            recovery_timeout: Duration::from_secs(60),
            ..Config::default()
        },
        Box::new(|_| {}),
    );
    let publisher = AuthCacheInvalidation::new(
        connect_guarded(&deployment, breaker.clone()),
        None,
        Handle::current(),
    );
    let subscriber = AuthCacheInvalidation::new(connect(&deployment), None, Handle::current());
    let mut received = subscriber.subscribe();
    eventually(async || {
        assert!(publisher.publish(&evict("subscribed")));
        arrived(&mut received).await
    })
    .await;
    breaker.failed(Failure::Hard, Instant::now());

    assert!(publisher.publish(&evict("team_id:t1")));
    let single = eventually(async || {
        arrived(&mut received)
            .await
            .filter(|invalidation| *invalidation == evict("team_id:t1"))
    })
    .await;
    publisher
        .broadcast_evictions(&["user-1".to_owned(), "user-2".to_owned()])
        .await;
    let bulk = [arrived(&mut received).await, arrived(&mut received).await];

    assert_eq!(single, evict("team_id:t1"));
    assert_eq!(bulk, [Some(evict("user-1")), Some(evict("user-2"))]);
    assert!(
        matches!(breaker.state(), State::Open { .. }),
        "{:?}",
        breaker.state()
    );
}

#[tokio::test]
async fn a_bulk_eviction_is_one_pipeline() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let (client, trips) = connect_observed(&deployment);
    let invalidation = AuthCacheInvalidation::new(client, None, Handle::current());

    invalidation
        .broadcast_evictions(&["a".to_owned(), "b".to_owned(), "c".to_owned()])
        .await;
    invalidation.broadcast_evictions(&[]).await;

    assert_eq!(trips.seen(), [Operation::Pipeline { ops: 3 }]);
}

#[tokio::test]
async fn a_redis_that_never_answers_drops_publishes_past_the_waiting_cap_and_bounds_a_bulk_wait() {
    let server = RedisServer::start().await.unwrap();
    let proxy = TcpProxy::start(server.port()).await.unwrap();
    let invalidation =
        AuthCacheInvalidation::new(connect_port(proxy.port()), None, Handle::current());
    proxy.blackhole_open_connections();

    let accepted = (0..1100)
        .filter(|_| invalidation.publish(&evict("team_id:t1")))
        .count();
    let started = tokio::time::Instant::now();
    invalidation
        .broadcast_evictions(&["team_id:t2".to_owned()])
        .await;

    assert_eq!(accepted, 1024);
    assert!(started.elapsed() < Duration::from_secs(3));
}

#[tokio::test]
async fn an_eviction_drops_this_workers_copies_and_a_set_writes_only_spend_counters() {
    let auth_cache = AuthCache::new(None, Settings::default());
    let spend_counters = SpendCounters::new(None);
    let team = AuthObject::Team("t1".to_owned());
    let counter = SpendCounter::Key {
        token: "abc".to_owned(),
        window: None,
    };
    auth_cache
        .set_local(&team, Value::from("cached"), None)
        .unwrap();
    auth_cache
        .set_local(&AuthObject::User("a".repeat(64)), Value::from("user"), None)
        .unwrap();

    auth_cache_invalidation::apply(
        &set("spend:key:abc", Number::Float(2.5), Some(Number::Int(60))),
        &auth_cache,
        &spend_counters,
    );
    for outside in [
        "team_id:t1",
        "spend_db_floor:spend:key:abc",
        "spend:unknown:x",
    ] {
        auth_cache_invalidation::apply(
            &set(outside, Number::Float(9.0), None),
            &auth_cache,
            &spend_counters,
        );
    }

    assert_eq!(
        spend_counters.snapshot().unwrap(),
        [("spend:key:abc".to_owned(), Number::Float(2.5))]
    );
    assert_eq!(
        auth_cache.get_local(&team).unwrap(),
        Some(Value::from("cached"))
    );

    auth_cache_invalidation::apply(&evict("team_id:t1"), &auth_cache, &spend_counters);
    auth_cache_invalidation::apply(&evict(&"a".repeat(64)), &auth_cache, &spend_counters);
    auth_cache_invalidation::apply(&evict("spend:key:abc"), &auth_cache, &spend_counters);

    assert_eq!(auth_cache.get_local(&team).unwrap(), None);
    assert_eq!(auth_cache.len_local().unwrap(), (0, 0));
    assert_eq!(spend_counters.get_local(&counter).unwrap(), None);
}
