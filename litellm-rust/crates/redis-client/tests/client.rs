mod support;

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use litellm_redis_testing::{RedisCluster, RedisDeployment, RedisServer, Topology, eventually};
use rstest::rstest;

use litellm_circuit_breaker::{Breaker, Config, State};

use litellm_redis_client::commands;
use litellm_redis_client::{
    Amount, CallEvent, Credentials, DEFAULT_TIMEOUT, Error, Expiry, FailureKind, Namespace,
    Operation, Outcome, RedisClient, RedisConfig, RedisEvents, RedisUrl, Script, Standalone,
    Target, Timeouts, Ttl, Value,
};
use litellm_secret_value::SecretValue;
use support::{KEEP, connect, connect_guarded, connections, endpoint};

async fn persist(client: &RedisClient, key: &str) {
    let set = Script::new("return redis.call('SET', KEYS[1], '0')");
    client
        .send(commands::eval::<()>(&set, &[key], Vec::<String>::new()))
        .await
        .unwrap();
}

#[derive(Clone, Copy, Debug)]
enum Counter {
    IncrBy,
    IncrByFloat,
}

const MINUTE: Duration = Duration::from_secs(60);

async fn write(client: &RedisClient, counter: Counter, key: &str, expiry: Expiry) {
    match (counter, expiry) {
        (Counter::IncrBy, Expiry::KeepExisting(ttl)) => {
            client.send(commands::incr_by(key, 1, ttl)).await.unwrap();
        }
        (Counter::IncrBy, Expiry::Refresh(_)) => unreachable!("increment_cache never refreshes"),
        (Counter::IncrByFloat, expiry) => {
            client
                .send(commands::incr_by_float(
                    key,
                    Amount::Float(0.5),
                    Some(expiry),
                ))
                .await
                .unwrap();
        }
    }
}

async fn persist_as(client: &RedisClient, key: &str) {
    let source = "return redis.call('SET', KEYS[1], '1')";
    client
        .send(commands::eval::<redis::Value>(
            &Script::new(source),
            &[key],
            Vec::<String>::new(),
        ))
        .await
        .unwrap();
}

fn remaining(ttl: Ttl) -> Duration {
    match ttl {
        Ttl::Expires(left) => left,
        other => panic!("expected an expiring key, got {other:?}"),
    }
}

#[rstest]
#[case::increment_cache(Counter::IncrBy, Expiry::KeepExisting(MINUTE))]
#[case::async_increment(Counter::IncrByFloat, Expiry::KeepExisting(MINUTE))]
#[case::async_increment_refreshing(Counter::IncrByFloat, Expiry::Refresh(MINUTE))]
#[tokio::test]
async fn a_counter_write_expires_a_new_key(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[case] counter: Counter,
    #[case] expiry: Expiry,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);

    write(&client, counter, "fresh", expiry).await;

    let left = remaining(client.send(commands::ttl("fresh")).await.unwrap());
    assert!(left > Duration::from_secs(50) && left <= MINUTE, "{left:?}");
}

#[rstest]
#[tokio::test]
async fn keep_existing_leaves_a_running_ttl_and_refresh_resets_it(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    let short = Expiry::Refresh(Duration::from_secs(30));
    let long = Duration::from_secs(600);
    write(&client, Counter::IncrByFloat, "kept", short).await;
    write(&client, Counter::IncrByFloat, "refreshed", short).await;
    write(
        &client,
        Counter::IncrBy,
        "kept_int",
        Expiry::KeepExisting(Duration::from_secs(30)),
    )
    .await;

    write(
        &client,
        Counter::IncrByFloat,
        "kept",
        Expiry::KeepExisting(long),
    )
    .await;
    write(
        &client,
        Counter::IncrByFloat,
        "refreshed",
        Expiry::Refresh(long),
    )
    .await;
    write(
        &client,
        Counter::IncrBy,
        "kept_int",
        Expiry::KeepExisting(long),
    )
    .await;

    assert!(
        remaining(client.send(commands::ttl("kept")).await.unwrap()) <= Duration::from_secs(30)
    );
    assert!(
        remaining(client.send(commands::ttl("kept_int")).await.unwrap()) <= Duration::from_secs(30)
    );
    assert!(
        remaining(client.send(commands::ttl("refreshed")).await.unwrap())
            > Duration::from_secs(500)
    );
}

#[rstest]
#[tokio::test]
async fn keep_existing_expires_a_key_that_had_no_ttl(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[values(Counter::IncrBy, Counter::IncrByFloat)] counter: Counter,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "ns", DEFAULT_TIMEOUT);
    persist_as(&client, "persistent").await;

    write(&client, counter, "persistent", KEEP).await;

    let left = remaining(client.send(commands::ttl("persistent")).await.unwrap());
    assert!(left > Duration::from_secs(50), "{left:?}");
}

#[rstest]
#[tokio::test]
async fn counters_send_the_commands_redis_cache_sends(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    server
        .cli_on_every_node(&["CONFIG", "RESETSTAT"])
        .await
        .unwrap();

    client
        .send(commands::incr_by("int", 1, MINUTE))
        .await
        .unwrap();
    client
        .send(commands::incr_by_float("kept", Amount::Int(1), Some(KEEP)))
        .await
        .unwrap();
    client
        .send(commands::incr_by_float(
            "refreshed",
            Amount::Float(0.5),
            Some(Expiry::Refresh(MINUTE)),
        ))
        .await
        .unwrap();
    client
        .run(
            &client
                .pipeline()
                .with(commands::incr_by_float(
                    "piped",
                    Amount::Float(0.25),
                    Some(Expiry::Refresh(MINUTE)),
                ))
                .with(commands::incr_by_float("bare", Amount::Float(0.25), None)),
        )
        .await;

    let stats = support::command_stats(&server).await;
    let calls = |command: &str| support::calls(&stats, command);
    assert_eq!(calls("incrby"), 1);
    assert_eq!(calls("incrbyfloat"), 4);
    assert_eq!(calls("ttl"), 2);
    assert_eq!(calls("expire"), 4);
    assert_eq!(
        calls("eval") + calls("evalsha") + calls("pttl") + calls("pexpire"),
        0
    );
}

#[derive(Default)]
struct Recorded(Mutex<Vec<CallEvent>>);

#[rstest]
#[case::beyond_double_precision(9_007_199_254_740_992_i64, 1_i64)]
#[case::negative_beyond_double_precision(-9_007_199_254_740_992_i64, -1_i64)]
#[case::maximum_integer(i64::MAX - 1, 1_i64)]
#[case::minimum_integer(i64::MIN + 1, -1_i64)]
#[tokio::test]
async fn integer_increments_preserve_every_bit(
    #[case] initial: i64,
    #[case] delta: i64,
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    client
        .send(commands::set("integer", initial, Duration::from_secs(60)))
        .await
        .unwrap();
    let expected = initial + delta;
    let result = client
        .send(commands::incr_by("integer", delta, MINUTE))
        .await
        .unwrap();
    assert_eq!(result, expected);
    assert_eq!(
        client.send(commands::get::<i64>("integer")).await.unwrap(),
        Some(expected)
    );
}

#[rstest]
#[tokio::test]
async fn outage_events_report_the_failure_kind(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[values(false, true)] timeout: bool,
    #[values(false, true)] pipeline: bool,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let recorded = Arc::new(Recorded::default());
    let client = RedisClient::new(RedisConfig {
        target: endpoint(&server),
        namespace: Namespace::default(),
        timeouts: Timeouts::new(Duration::from_millis(300), None),
        breaker: None,
        events: Some(recorded.clone()),
        credentials: Credentials::NONE,
        health_check: None,
        keepalive: true,
    })
    .unwrap();
    client.ping().await.unwrap();
    if timeout {
        server.freeze().await.unwrap();
    } else {
        server.stop().await.unwrap();
    }
    let result = if pipeline {
        client
            .run(&client.pipeline().with(commands::get::<Value>("missing")))
            .await
            .remove(0)
            .map(drop)
    } else {
        client
            .send(commands::get::<String>("missing"))
            .await
            .map(drop)
    };
    if timeout {
        server.thaw().await.unwrap();
    } else {
        server.resume().await.unwrap();
    }
    assert!(matches!(result, Err(Error::Transport(_))));
    let expected = if timeout {
        FailureKind::Timeout
    } else {
        FailureKind::Transport
    };
    assert_eq!(
        recorded.0.lock().unwrap().last().unwrap().outcome,
        Outcome::Failed(expected)
    );
}

impl RedisEvents for Recorded {
    fn call(&self, event: &CallEvent) {
        self.0.lock().unwrap().push(*event);
    }

    fn breaker(&self, _transition: &litellm_circuit_breaker::Transition<State>) {}
}

#[rstest]
#[tokio::test]
async fn every_call_reports_its_operation_and_outcome(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let recorded = Arc::new(Recorded::default());
    let client = RedisClient::new(RedisConfig {
        target: endpoint(&server),
        credentials: Credentials::NONE,
        namespace: Namespace::default(),
        timeouts: Timeouts::default(),
        breaker: None,
        events: Some(recorded.clone()),
        health_check: None,
        keepalive: true,
    })
    .unwrap();
    client
        .send(commands::set("list", "item", Duration::from_secs(60)))
        .await
        .unwrap();

    let _ = client
        .send(commands::incr_by("list", 1, Duration::from_secs(60)))
        .await;
    let _ = client.send(commands::get::<String>("missing")).await;
    let _ = client
        .run(
            &client
                .pipeline()
                .with(commands::get::<Value>("a"))
                .with(commands::get::<Value>("b")),
        )
        .await;
    let failures = client
        .run(&client.pipeline().with(commands::incr_by_float(
            "list",
            Amount::Int(1),
            Some(Expiry::Refresh(Duration::from_secs(60))),
        )))
        .await;
    assert!(matches!(failures[0], Err(Error::Server(_))));

    let seen: Vec<(Operation, Outcome)> = recorded
        .0
        .lock()
        .unwrap()
        .iter()
        .map(|event| (event.operation, event.outcome))
        .collect();
    assert_eq!(
        seen,
        vec![
            (Operation::Command("SET"), Outcome::Succeeded),
            (
                Operation::Command("INCRBY"),
                Outcome::Failed(FailureKind::Server)
            ),
            (Operation::Command("GET"), Outcome::Succeeded),
            (Operation::Pipeline { ops: 2 }, Outcome::Succeeded),
            (
                Operation::Pipeline { ops: 1 },
                Outcome::Failed(FailureKind::Server)
            ),
        ]
    );
}

#[rstest]
#[tokio::test]
async fn namespaces_isolate_keys_on_the_server(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let a = connect(&server, "a", DEFAULT_TIMEOUT);
    let b = connect(&server, "b", DEFAULT_TIMEOUT);
    let bare = connect(&server, "", DEFAULT_TIMEOUT);

    a.send(commands::set("x", "1", Duration::from_secs(60)))
        .await
        .unwrap();

    assert_eq!(
        a.send(commands::get::<String>("x"))
            .await
            .unwrap()
            .as_deref(),
        Some("1")
    );
    assert_eq!(
        a.send(commands::get::<String>("a:x"))
            .await
            .unwrap()
            .as_deref(),
        Some("1")
    );
    assert_eq!(
        bare.send(commands::get::<String>("a:x"))
            .await
            .unwrap()
            .as_deref(),
        Some("1")
    );
    assert_eq!(b.send(commands::get::<String>("x")).await.unwrap(), None);
}

#[rstest]
#[tokio::test]
async fn a_rejected_command_keeps_the_connection(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    client
        .send(commands::set("list", "item", Duration::from_secs(60)))
        .await
        .unwrap();
    let before = connections(&server).await;

    let rejected = client
        .send(commands::incr_by("list", 1, Duration::from_secs(60)))
        .await;

    assert!(matches!(rejected, Err(Error::Server(_))), "{rejected:?}");
    assert_eq!(
        client
            .send(commands::get::<String>("list"))
            .await
            .unwrap()
            .as_deref(),
        Some("item")
    );
    assert_eq!(connections(&server).await, before);
}

#[rstest]
#[tokio::test]
async fn a_reply_of_another_type_is_a_decode_error_and_a_missing_key_is_none(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    client
        .send(commands::set("float", "15.0", Duration::from_secs(60)))
        .await
        .unwrap();

    let decoded = client.send(commands::get::<i64>("float")).await;

    assert!(matches!(decoded, Err(Error::Decode(_))), "{decoded:?}");
    assert_eq!(
        client.send(commands::get::<i64>("missing")).await.unwrap(),
        None
    );
}

#[rstest]
#[tokio::test]
async fn a_timed_out_command_does_not_leak_its_reply_into_the_next_one(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", Duration::from_millis(500));
    client
        .send(commands::set("first", "1", Duration::from_secs(60)))
        .await
        .unwrap();
    client
        .send(commands::set("second", "2", Duration::from_secs(60)))
        .await
        .unwrap();

    server.freeze().await.unwrap();
    let timed_out = client.send(commands::get::<String>("first")).await;
    server.thaw().await.unwrap();

    assert!(
        matches!(timed_out, Err(Error::Transport(_))),
        "{timed_out:?}"
    );
    assert_eq!(
        client
            .send(commands::get::<String>("second"))
            .await
            .unwrap()
            .as_deref(),
        Some("2")
    );
}

#[rstest]
#[tokio::test]
async fn the_same_client_recovers_after_redis_restarts(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", Duration::from_millis(500));

    server.restart().await.unwrap();

    let recovered = eventually(async || {
        client
            .send(commands::incr_by(
                "after_restart",
                1,
                Duration::from_secs(60),
            ))
            .await
            .ok()
    })
    .await;
    assert_eq!(recovered, 1);
}

#[rstest]
#[tokio::test]
async fn a_client_created_while_redis_is_down_works_once_redis_is_up(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    server.stop().await.unwrap();

    let client = connect(&server, "", Duration::from_millis(500));
    let started = Instant::now();
    let while_down = client.ping().await;
    let failed_in = started.elapsed();
    server.resume().await.unwrap();

    assert!(failed_in < Duration::from_secs(2), "{failed_in:?}");

    assert!(
        matches!(while_down, Err(Error::Transport(_))),
        "{while_down:?}"
    );
    let recovered = eventually(async || client.ping().await.ok()).await;
    assert_eq!(recovered, ());
}

#[rstest]
#[tokio::test]
async fn empty_batches_succeed_without_asking_redis(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);

    assert_eq!(
        client.send(commands::mget::<String>(&[])).await.unwrap(),
        Vec::<Option<String>>::new()
    );
    assert_eq!(client.send(commands::del(&[])).await.unwrap(), 0);
}

#[rstest]
#[tokio::test]
async fn mget_and_del_span_slots_in_input_order(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    client
        .send(commands::set("a", "1", Duration::from_secs(60)))
        .await
        .unwrap();
    client
        .send(commands::set("c", "3", Duration::from_secs(60)))
        .await
        .unwrap();

    let values = client
        .send(commands::mget::<String>(&["c", "b", "a"]))
        .await
        .unwrap();
    let deleted = client.send(commands::del(&["a", "b", "c"])).await.unwrap();

    assert_eq!(
        values,
        vec![Some("3".to_owned()), None, Some("1".to_owned())]
    );
    assert_eq!(deleted, 2);
    assert_eq!(
        client
            .send(commands::mget::<String>(&["a", "c"]))
            .await
            .unwrap(),
        vec![None, None]
    );
}

#[rstest]
#[tokio::test]
async fn set_nx_only_writes_a_missing_key(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);

    let first = client
        .send(commands::set_nx("lock", "owner-1", Duration::from_secs(60)))
        .await
        .unwrap();
    let second = client
        .send(commands::set_nx("lock", "owner-2", Duration::from_secs(60)))
        .await
        .unwrap();

    assert!(first);
    assert!(!second);
    assert_eq!(
        client
            .send(commands::get::<String>("lock"))
            .await
            .unwrap()
            .as_deref(),
        Some("owner-1")
    );
}

#[rstest]
#[tokio::test]
async fn ttl_reports_missing_persistent_and_remaining_time(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    persist(&client, "persistent").await;
    client
        .send(commands::set("expiring", "1", Duration::from_secs(60)))
        .await
        .unwrap();
    client
        .send(commands::incr_by_float(
            "expired_later",
            Amount::Float(0.5),
            Some(KEEP),
        ))
        .await
        .unwrap();
    assert!(
        client
            .send(commands::expire("expired_later", Duration::from_secs(30)))
            .await
            .unwrap()
    );

    assert_eq!(
        client.send(commands::ttl("missing")).await.unwrap(),
        Ttl::Missing
    );
    assert_eq!(
        client.send(commands::ttl("persistent")).await.unwrap(),
        Ttl::Persistent
    );
    assert!(matches!(
        client.send(commands::ttl("expiring")).await.unwrap(),
        Ttl::Expires(left) if left > Duration::from_secs(50) && left <= Duration::from_secs(60)
    ));
    assert!(matches!(
        client.send(commands::ttl("expired_later")).await.unwrap(),
        Ttl::Expires(left) if left > Duration::from_secs(20) && left <= Duration::from_secs(30)
    ));
    assert!(
        !client
            .send(commands::expire("missing", Duration::from_secs(30)))
            .await
            .unwrap()
    );
}

#[rstest]
#[tokio::test]
async fn incr_by_float_accumulates_fractions(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);

    client
        .send(commands::incr_by_float(
            "spend",
            Amount::Float(0.25),
            Some(KEEP),
        ))
        .await
        .unwrap();

    assert_eq!(
        client
            .send(commands::incr_by_float(
                "spend",
                Amount::Float(0.5),
                Some(KEEP)
            ))
            .await
            .unwrap(),
        0.75
    );
}

#[rstest]
#[tokio::test]
async fn concurrent_commands_on_one_cloned_client_all_apply(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);

    let tasks: Vec<_> = (0..100)
        .map(|_| {
            let client = client.clone();
            tokio::spawn(async move {
                for _ in 0..10 {
                    client
                        .send(commands::incr_by("shared", 1, Duration::from_secs(60)))
                        .await
                        .unwrap();
                }
            })
        })
        .collect();
    for task in tasks {
        task.await.unwrap();
    }

    assert_eq!(
        client.send(commands::get::<i64>("shared")).await.unwrap(),
        Some(1000)
    );
}

fn breaker(failure_threshold: u32) -> Breaker {
    Breaker::new(
        Config {
            failure_threshold,
            recovery_timeout: Duration::from_millis(200),
            timeout_min_duration: Duration::ZERO,
        },
        Box::new(|_| {}),
    )
}

#[rstest]
#[tokio::test]
async fn an_open_breaker_rejects_without_waiting_on_redis_and_a_probe_closes_it(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let breaker = breaker(2);
    let client = connect_guarded(
        &server,
        "",
        Duration::from_millis(300),
        Some(breaker.clone()),
    );

    server.freeze().await.unwrap();
    let first = client.send(commands::get::<String>("k")).await;
    let second = client.send(commands::get::<String>("k")).await;
    let started = Instant::now();
    let third = client.send(commands::get::<String>("k")).await;
    let rejected_in = started.elapsed();
    server.thaw().await.unwrap();

    assert!(matches!(first, Err(Error::Transport(_))), "{first:?}");
    assert!(matches!(second, Err(Error::Transport(_))), "{second:?}");
    assert!(matches!(third, Err(Error::BreakerOpen)), "{third:?}");
    assert!(rejected_in < Duration::from_millis(50), "{rejected_in:?}");

    let recovered = eventually(async || client.send(commands::get::<String>("k")).await.ok()).await;
    assert_eq!(recovered, None);
    assert!(
        matches!(breaker.state(), State::Closed { .. }),
        "{:?}",
        breaker.state()
    );
}

#[rstest]
#[tokio::test]
async fn rejected_commands_never_open_the_breaker(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let breaker = breaker(2);
    let client = connect_guarded(&server, "", DEFAULT_TIMEOUT, Some(breaker.clone()));
    client
        .send(commands::set("list", "item", Duration::from_secs(60)))
        .await
        .unwrap();

    let rejected = [
        client
            .send(commands::incr_by("list", 1, Duration::from_secs(60)))
            .await,
        client
            .send(commands::incr_by("list", 1, Duration::from_secs(60)))
            .await,
        client
            .send(commands::incr_by("list", 1, Duration::from_secs(60)))
            .await,
    ];

    assert!(
        matches!(rejected[2], Err(Error::Server(_))),
        "{:?}",
        rejected[2]
    );
    assert!(
        matches!(breaker.state(), State::Closed { hard: 0, .. }),
        "{:?}",
        breaker.state()
    );
}

#[rstest]
#[tokio::test]
async fn an_open_breaker_fails_every_pipeline_op_without_sending(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect_guarded(&server, "", Duration::from_millis(300), Some(breaker(1)));
    let pipeline = client
        .pipeline()
        .with(commands::incr_by_float(
            "a",
            Amount::Int(1),
            Some(Expiry::Refresh(Duration::from_secs(60))),
        ))
        .with(commands::incr_by_float(
            "c",
            Amount::Int(1),
            Some(Expiry::Refresh(Duration::from_secs(60))),
        ));

    server.freeze().await.unwrap();
    let failed = client.run(&pipeline).await;
    let rejected = client.run(&pipeline).await;
    server.thaw().await.unwrap();

    assert!(
        matches!(failed[0], Err(Error::Transport(_))),
        "{:?}",
        failed[0]
    );
    assert!(
        matches!(rejected[0], Err(Error::BreakerOpen)),
        "{:?}",
        rejected[0]
    );
    assert!(
        matches!(rejected[1], Err(Error::BreakerOpen)),
        "{:?}",
        rejected[1]
    );
}

#[rstest]
#[tokio::test]
async fn a_host_endpoint_connects_to_its_database() {
    let server = RedisServer::start().await.unwrap();
    let port = server.url().rsplit(':').next().unwrap().parse().unwrap();
    let connect = |target| {
        RedisClient::new(RedisConfig {
            target,
            credentials: Credentials::NONE,
            namespace: Namespace::default(),
            timeouts: Timeouts::default(),
            breaker: None,
            events: None,
            health_check: None,
            keepalive: true,
        })
    };
    let url = |url: String| {
        Target::Url(RedisUrl {
            url: SecretValue::new(url),
            tls: None,
            db: None,
            username: None,
            password: None,
        })
    };
    let in_db_3 = connect(Target::Standalone(Standalone {
        host: "127.0.0.1".to_owned(),
        port,
        tls: None,
        db: 3,
    }))
    .unwrap();
    let in_db_0 = connect(url(server.url().to_owned())).unwrap();
    let via_url = connect(url(format!("{}/3", server.url()))).unwrap();

    in_db_3
        .send(commands::set("k", "3", Duration::from_secs(60)))
        .await
        .unwrap();

    assert_eq!(
        in_db_0.send(commands::get::<String>("k")).await.unwrap(),
        None
    );
    assert_eq!(
        via_url
            .send(commands::get::<String>("k"))
            .await
            .unwrap()
            .as_deref(),
        Some("3")
    );
}

async fn ping_calls(cluster: &RedisCluster) -> Vec<u64> {
    cluster
        .cli_outputs(&["INFO", "commandstats"])
        .await
        .unwrap()
        .iter()
        .map(|info| {
            info.lines()
                .find_map(|line| line.strip_prefix("cmdstat_ping:calls="))
                .and_then(|rest| rest.split(',').next())
                .and_then(|calls| calls.trim().parse().ok())
                .unwrap_or(0)
        })
        .collect()
}

fn health_checked(
    server: &RedisDeployment,
    interval: Duration,
    timeout: Duration,
    breaker: Option<Breaker>,
    events: Option<Arc<Recorded>>,
) -> RedisClient {
    RedisClient::new(RedisConfig {
        target: endpoint(server),
        credentials: Credentials::NONE,
        namespace: Namespace::default(),
        timeouts: Timeouts::new(timeout, None),
        breaker,
        events: events.map(|events| events as Arc<dyn RedisEvents>),
        health_check: Some(interval),
        keepalive: true,
    })
    .unwrap()
}

// redis-py 5.3.1 redis/asyncio/connection.py check_health: a connection idle longer than
// health_check_interval gets a PING before the next command, a busy one does not
#[rstest]
#[tokio::test]
async fn an_idle_client_pings_every_node_before_the_next_command_like_redis_py() {
    let server = RedisDeployment::start(Topology::Cluster).await.unwrap();
    let RedisDeployment::Cluster(cluster) = &server else {
        unreachable!("started a cluster")
    };
    let client = health_checked(&server, Duration::from_secs(3), DEFAULT_TIMEOUT, None, None);
    client
        .send(commands::set("k", "1", Duration::from_secs(60)))
        .await
        .unwrap();
    let before = ping_calls(cluster).await;

    client
        .send(commands::set("k", "2", Duration::from_secs(60)))
        .await
        .unwrap();
    let busy = ping_calls(cluster).await;
    tokio::time::sleep(Duration::from_millis(3500)).await;
    client
        .send(commands::set("k", "3", Duration::from_secs(60)))
        .await
        .unwrap();
    let idle = ping_calls(cluster).await;

    assert_eq!(busy, before);
    assert_eq!(
        idle,
        before.iter().map(|calls| calls + 1).collect::<Vec<_>>(),
        "before {before:?}"
    );
}

#[rstest]
#[tokio::test]
async fn a_failed_health_check_is_neither_a_call_nor_a_breaker_failure() {
    let server = RedisDeployment::start(Topology::Cluster).await.unwrap();
    let recorded = Arc::new(Recorded::default());
    let breaker = breaker(2);
    let client = health_checked(
        &server,
        Duration::from_millis(200),
        Duration::from_millis(300),
        Some(breaker.clone()),
        Some(recorded.clone()),
    );
    client
        .send(commands::set("k", "1", Duration::from_secs(60)))
        .await
        .unwrap();

    server.freeze().await.unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    let failed = client.send(commands::get::<String>("k")).await;
    server.thaw().await.unwrap();

    assert!(matches!(failed, Err(Error::Transport(_))), "{failed:?}");
    assert!(
        matches!(breaker.state(), State::Closed { timeouts: 1, .. }),
        "{:?}",
        breaker.state()
    );
    assert_eq!(recorded.0.lock().unwrap().len(), 2);
}

#[rstest]
#[case::int(Amount::Int(1), "1")]
#[case::float_keeps_its_point(Amount::Float(1.0), "1.0")]
#[case::small_float(Amount::Float(1e-5), "1e-05")]
#[case::large_float(Amount::Float(1e16), "1e+16")]
#[case::fraction(Amount::Float(0.1 + 0.2), "0.30000000000000004")]
fn amounts_are_written_like_redis_py_encodes_them(#[case] amount: Amount, #[case] expected: &str) {
    assert_eq!(
        redis::ToRedisArgs::to_redis_args(&amount),
        vec![expected.as_bytes().to_vec()]
    );
}
