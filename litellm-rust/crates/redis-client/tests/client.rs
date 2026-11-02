mod support;

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use litellm_redis_testing::{RedisDeployment, RedisServer, Topology, eventually};
use rstest::rstest;

use litellm_circuit_breaker::{Breaker, Config, State};

use litellm_redis_client::{
    CallEvent, Credentials, DEFAULT_TIMEOUT, Endpoint, Error, Expiry, FailureKind, HostEndpoint,
    Namespace, Operation, Outcome, RedisClient, RedisConfig, RedisEvents, Script, Timeouts,
    TlsOptions, Ttl,
};
use litellm_secret_value::SecretValue;
use support::{KEEP, connect, connect_guarded, connections, endpoint};

async fn persist(client: &RedisClient, key: &str) {
    let set = Script::new("return redis.call('SET', KEYS[1], '0')");
    client
        .eval::<()>(&set, &[key], Vec::<String>::new())
        .await
        .unwrap();
}

#[derive(Clone, Copy, Debug)]
enum Counter {
    IncrBy,
    IncrByFloat,
    Rpush,
}

async fn write(client: &RedisClient, counter: Counter, key: &str, expiry: Expiry) {
    match counter {
        Counter::IncrBy => drop(client.incr_by(key, 1, expiry).await.unwrap()),
        Counter::IncrByFloat => drop(client.incr_by_float(key, 0.5, expiry).await.unwrap()),
        Counter::Rpush => drop(client.rpush(key, "item", expiry).await.unwrap()),
    }
}

async fn persist_as(client: &RedisClient, counter: Counter, key: &str) {
    let source = match counter {
        Counter::IncrBy | Counter::IncrByFloat => "return redis.call('SET', KEYS[1], '1')",
        Counter::Rpush => "return redis.call('RPUSH', KEYS[1], 'first')",
    };
    client
        .eval::<redis::Value>(&Script::new(source), &[key], Vec::<String>::new())
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
#[tokio::test]
async fn a_counter_write_expires_a_new_key_in_the_same_command(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[values(Counter::IncrBy, Counter::IncrByFloat, Counter::Rpush)] counter: Counter,
    #[values(
        Expiry::KeepExisting(Duration::from_secs(60)),
        Expiry::Refresh(Duration::from_secs(60))
    )]
    expiry: Expiry,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);

    write(&client, counter, "fresh", expiry).await;

    let left = remaining(client.ttl("fresh").await.unwrap());
    assert!(
        left > Duration::from_secs(50) && left <= Duration::from_secs(60),
        "{left:?}"
    );
}

#[rstest]
#[tokio::test]
async fn keep_existing_leaves_a_running_ttl_and_refresh_resets_it(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[values(Counter::IncrBy, Counter::IncrByFloat, Counter::Rpush)] counter: Counter,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    let long = Duration::from_secs(600);
    write(
        &client,
        counter,
        "kept",
        Expiry::Refresh(Duration::from_secs(30)),
    )
    .await;
    write(
        &client,
        counter,
        "refreshed",
        Expiry::Refresh(Duration::from_secs(30)),
    )
    .await;

    write(&client, counter, "kept", Expiry::KeepExisting(long)).await;
    write(&client, counter, "refreshed", Expiry::Refresh(long)).await;

    assert!(remaining(client.ttl("kept").await.unwrap()) <= Duration::from_secs(30));
    assert!(remaining(client.ttl("refreshed").await.unwrap()) > Duration::from_secs(500));
}

#[rstest]
#[tokio::test]
async fn keep_existing_expires_a_key_that_had_no_ttl(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[values(Counter::IncrBy, Counter::IncrByFloat, Counter::Rpush)] counter: Counter,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "ns", DEFAULT_TIMEOUT);
    persist_as(&client, counter, "persistent").await;

    write(&client, counter, "persistent", KEEP).await;

    let left = remaining(client.ttl("persistent").await.unwrap());
    assert!(left > Duration::from_secs(50), "{left:?}");
}

#[derive(Default)]
struct Recorded(Mutex<Vec<CallEvent>>);

#[rstest]
#[tokio::test]
async fn atomic_writes_require_scripting_permissions(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[values(Counter::IncrBy, Counter::IncrByFloat, Counter::Rpush)] counter: Counter,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    server
        .cli_on_every_node(&["ACL", "SETUSER", "default", "-@scripting"])
        .await
        .unwrap();
    client
        .set("ordinary", "allowed", Duration::from_secs(60))
        .await
        .unwrap();
    assert_eq!(
        client.get::<String>("ordinary").await.unwrap().as_deref(),
        Some("allowed")
    );
    let denied = match counter {
        Counter::IncrBy => client.incr_by("atomic", 1, KEEP).await.map(drop),
        Counter::IncrByFloat => client.incr_by_float("atomic", 0.5, KEEP).await.map(drop),
        Counter::Rpush => client.rpush("atomic", "item", KEEP).await.map(drop),
    };
    assert!(matches!(denied, Err(Error::Server(_))));
    assert_eq!(client.get::<String>("atomic").await.unwrap(), None);
    server
        .cli_on_every_node(&["ACL", "SETUSER", "default", "+@scripting"])
        .await
        .unwrap();
    write(&client, counter, "atomic", KEEP).await;
    assert!(remaining(client.ttl("atomic").await.unwrap()) > Duration::ZERO);
}

#[rstest]
#[tokio::test]
async fn an_integer_reply_acl_failure_does_not_change_the_counter(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[values(false, true)] pipelined: bool,
    #[values(None, Some(5))] initial: Option<i64>,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    if let Some(value) = initial {
        client
            .set("atomic", value, Duration::from_secs(60))
            .await
            .unwrap();
    }
    server
        .cli_on_every_node(&["ACL", "SETUSER", "default", "-get"])
        .await
        .unwrap();
    let result = if pipelined {
        client
            .run(&client.pipeline().incr_by("atomic", 1, KEEP))
            .await
            .remove(0)
            .map(drop)
    } else {
        client.incr_by("atomic", 1, KEEP).await.map(drop)
    };
    assert!(matches!(result, Err(Error::Server(_))));
    server
        .cli_on_every_node(&["ACL", "SETUSER", "default", "+get"])
        .await
        .unwrap();
    assert_eq!(client.get::<i64>("atomic").await.unwrap(), initial);
    if initial.is_some() {
        assert!(remaining(client.ttl("atomic").await.unwrap()) > Duration::ZERO);
    }
}

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
    #[values(false, true)] pipelined: bool,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    client
        .set("integer", initial, Duration::from_secs(60))
        .await
        .unwrap();
    let expected = initial + delta;
    let result = if pipelined {
        client
            .run(&client.pipeline().incr_by("integer", delta, KEEP))
            .await
            .remove(0)
            .unwrap()
    } else {
        litellm_redis_client::Value::Int(client.incr_by("integer", delta, KEEP).await.unwrap())
    };
    assert_eq!(result, litellm_redis_client::Value::Int(expected));
    assert_eq!(client.get::<i64>("integer").await.unwrap(), Some(expected));
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
        endpoint: endpoint(&server),
        namespace: Namespace::default(),
        timeouts: Timeouts::new(Duration::from_millis(300), None),
        breaker: None,
        events: Some(recorded.clone()),
        credentials: Credentials::NONE,
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
            .run(&client.pipeline().get("missing"))
            .await
            .remove(0)
            .map(drop)
    } else {
        client.get::<String>("missing").await.map(drop)
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
        endpoint: endpoint(&server),
        credentials: Credentials::NONE,
        namespace: Namespace::default(),
        timeouts: Timeouts::default(),
        breaker: None,
        events: Some(recorded.clone()),
    })
    .unwrap();
    client.rpush("list", "item", KEEP).await.unwrap();

    let _ = client.incr_by("list", 1, KEEP).await;
    let _ = client.get::<String>("missing").await;
    let _ = client.run(&client.pipeline().get("a").get("b")).await;
    let failures = client
        .run(&client.pipeline().incr_by("list", 1, KEEP))
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
            (Operation::Command("RPUSH"), Outcome::Succeeded),
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

    a.set("x", "1", Duration::from_secs(60)).await.unwrap();

    assert_eq!(a.get::<String>("x").await.unwrap().as_deref(), Some("1"));
    assert_eq!(a.get::<String>("a:x").await.unwrap().as_deref(), Some("1"));
    assert_eq!(
        bare.get::<String>("a:x").await.unwrap().as_deref(),
        Some("1")
    );
    assert_eq!(b.get::<String>("x").await.unwrap(), None);
}

#[rstest]
#[tokio::test]
async fn a_rejected_command_keeps_the_connection(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    client.rpush("list", "item", KEEP).await.unwrap();
    let before = connections(&server).await;

    let rejected = client.incr_by("list", 1, KEEP).await;

    assert!(matches!(rejected, Err(Error::Server(_))), "{rejected:?}");
    assert_eq!(client.rpush("list", "next", KEEP).await.unwrap(), 2);
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
        .set("float", "15.0", Duration::from_secs(60))
        .await
        .unwrap();

    let decoded = client.get::<i64>("float").await;

    assert!(matches!(decoded, Err(Error::Decode(_))), "{decoded:?}");
    assert_eq!(client.get::<i64>("missing").await.unwrap(), None);
}

#[rstest]
#[tokio::test]
async fn a_timed_out_command_does_not_leak_its_reply_into_the_next_one(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", Duration::from_millis(500));
    client
        .set("first", "1", Duration::from_secs(60))
        .await
        .unwrap();
    client
        .set("second", "2", Duration::from_secs(60))
        .await
        .unwrap();

    server.freeze().await.unwrap();
    let timed_out = client.get::<String>("first").await;
    server.thaw().await.unwrap();

    assert!(
        matches!(timed_out, Err(Error::Transport(_))),
        "{timed_out:?}"
    );
    assert_eq!(
        client.get::<String>("second").await.unwrap().as_deref(),
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

    let recovered = eventually(async || client.incr_by("after_restart", 1, KEEP).await.ok()).await;
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
        client.mget::<String>(&[]).await.unwrap(),
        Vec::<Option<String>>::new()
    );
    assert_eq!(client.del(&[]).await.unwrap(), 0);
}

#[rstest]
#[tokio::test]
async fn mget_and_del_span_slots_in_input_order(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let server = RedisDeployment::start(topology).await.unwrap();
    let client = connect(&server, "", DEFAULT_TIMEOUT);
    client.set("a", "1", Duration::from_secs(60)).await.unwrap();
    client.set("c", "3", Duration::from_secs(60)).await.unwrap();

    let values = client.mget::<String>(&["c", "b", "a"]).await.unwrap();
    let deleted = client.del(&["a", "b", "c"]).await.unwrap();

    assert_eq!(
        values,
        vec![Some("3".to_owned()), None, Some("1".to_owned())]
    );
    assert_eq!(deleted, 2);
    assert_eq!(
        client.mget::<String>(&["a", "c"]).await.unwrap(),
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
        .set_nx("lock", "owner-1", Duration::from_secs(60))
        .await
        .unwrap();
    let second = client
        .set_nx("lock", "owner-2", Duration::from_secs(60))
        .await
        .unwrap();

    assert!(first);
    assert!(!second);
    assert_eq!(
        client.get::<String>("lock").await.unwrap().as_deref(),
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
        .set("expiring", "1", Duration::from_secs(60))
        .await
        .unwrap();
    client
        .incr_by_float("expired_later", 0.5, KEEP)
        .await
        .unwrap();
    assert!(
        client
            .expire("expired_later", Duration::from_secs(30))
            .await
            .unwrap()
    );

    assert_eq!(client.ttl("missing").await.unwrap(), Ttl::Missing);
    assert_eq!(client.ttl("persistent").await.unwrap(), Ttl::Persistent);
    assert!(matches!(
        client.ttl("expiring").await.unwrap(),
        Ttl::Expires(left) if left > Duration::from_secs(50) && left <= Duration::from_secs(60)
    ));
    assert!(matches!(
        client.ttl("expired_later").await.unwrap(),
        Ttl::Expires(left) if left > Duration::from_secs(20) && left <= Duration::from_secs(30)
    ));
    assert!(
        !client
            .expire("missing", Duration::from_secs(30))
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

    client.incr_by_float("spend", 0.25, KEEP).await.unwrap();

    assert_eq!(
        client.incr_by_float("spend", 0.5, KEEP).await.unwrap(),
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
                    client.incr_by("shared", 1, KEEP).await.unwrap();
                }
            })
        })
        .collect();
    for task in tasks {
        task.await.unwrap();
    }

    assert_eq!(client.get::<i64>("shared").await.unwrap(), Some(1000));
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
    let first = client.get::<String>("k").await;
    let second = client.get::<String>("k").await;
    let started = Instant::now();
    let third = client.get::<String>("k").await;
    let rejected_in = started.elapsed();
    server.thaw().await.unwrap();

    assert!(matches!(first, Err(Error::Transport(_))), "{first:?}");
    assert!(matches!(second, Err(Error::Transport(_))), "{second:?}");
    assert!(matches!(third, Err(Error::BreakerOpen)), "{third:?}");
    assert!(rejected_in < Duration::from_millis(50), "{rejected_in:?}");

    let recovered = eventually(async || client.get::<String>("k").await.ok()).await;
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
    client.rpush("list", "item", KEEP).await.unwrap();

    let rejected = [
        client.incr_by("list", 1, KEEP).await,
        client.incr_by("list", 1, KEEP).await,
        client.incr_by("list", 1, KEEP).await,
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
        .incr_by("a", 1, KEEP)
        .incr_by("c", 1, KEEP);

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
    let connect = |endpoint| {
        RedisClient::new(RedisConfig {
            endpoint,
            credentials: Credentials::NONE,
            namespace: Namespace::default(),
            timeouts: Timeouts::default(),
            breaker: None,
            events: None,
        })
    };
    let url = |url: String| Endpoint::Url {
        url: SecretValue::new(url),
        tls: TlsOptions::verified(),
    };
    let in_db_3 = connect(Endpoint::Host(HostEndpoint {
        host: "127.0.0.1".to_owned(),
        port,
        tls: None,
        db: 3,
    }))
    .unwrap();
    let in_db_0 = connect(url(server.url().to_owned())).unwrap();
    let via_url = connect(url(format!("{}/3", server.url()))).unwrap();

    in_db_3
        .set("k", "3", Duration::from_secs(60))
        .await
        .unwrap();

    assert_eq!(in_db_0.get::<String>("k").await.unwrap(), None);
    assert_eq!(
        via_url.get::<String>("k").await.unwrap().as_deref(),
        Some("3")
    );
}
