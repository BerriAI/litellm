mod support;

use std::{sync::Arc, time::Duration};

use litellm_redis_client::{Operation, RedisClient, Script, Ttl};
use litellm_redis_store::{
    Number, RequestRedis,
    rate_limits::{
        AcquireReply, Applied, AtomicCounter, AtomicReply, Counter, Entry, Gauge, GuardedIncrement,
        GuardedReply, Increment, Kind, Local, RateLimits, Scope, WindowCount,
    },
    router_cooldowns::{POLICY as COOLDOWN_POLICY, RouterCooldowns},
};
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::{connect, connect_observed, within};
use tokio::runtime::Handle;

const SLOT_TTL: Duration = Duration::from_secs(3600);

struct WireScope {
    descriptor: &'static str,
    value: &'static str,
    window: &'static str,
    requests: &'static str,
    tokens: &'static str,
    gauge: &'static str,
}

const SCOPES: [WireScope; 4] = [
    WireScope {
        descriptor: "api_key",
        value: "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
        window: "{api_key:88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b}:window",
        requests: "{api_key:88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b}:requests",
        tokens: "{api_key:88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b}:tokens",
        gauge: "{api_key:88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b}:max_parallel_requests",
    },
    WireScope {
        descriptor: "team_member",
        value: "team-1:user@example.com",
        window: "{team_member:team-1:user@example.com}:window",
        requests: "{team_member:team-1:user@example.com}:requests",
        tokens: "{team_member:team-1:user@example.com}:tokens",
        gauge: "{team_member:team-1:user@example.com}:max_parallel_requests",
    },
    WireScope {
        descriptor: "model_per_key",
        value: "sk-hash:anthropic/claude-sonnet-5-5",
        window: "{model_per_key:sk-hash:anthropic/claude-sonnet-5-5}:window",
        requests: "{model_per_key:sk-hash:anthropic/claude-sonnet-5-5}:requests",
        tokens: "{model_per_key:sk-hash:anthropic/claude-sonnet-5-5}:tokens",
        gauge: "{model_per_key:sk-hash:anthropic/claude-sonnet-5-5}:max_parallel_requests",
    },
    WireScope {
        descriptor: "tag",
        value: "tier}é",
        window: "{tag:tier}é}:window",
        requests: "{tag:tier}é}:requests",
        tokens: "{tag:tier}é}:tokens",
        gauge: "{tag:tier}é}:max_parallel_requests",
    },
];

const SCRIPTS: [(&str, &str); 8] = [
    (
        "batch_rate_limiter",
        "1e1d6e80e9488215da3f8d4aefae5ee3fbe71663",
    ),
    (
        "batch_counter_read",
        "b272298fae5e83182e7719c6be8d1ef76a62465f",
    ),
    (
        "check_and_increment_by_n",
        "cbf6fb1b9184f1834b44b05bdeda3806fbc2da41",
    ),
    (
        "window_guarded_token_increment",
        "fec058661fefb81df4e5da0cc045df5f4a3f0a00",
    ),
    (
        "parallel_acquire",
        "86c5a792cfc23df2842c413908b3eb3f2b811ac3",
    ),
    (
        "parallel_release",
        "cb232fec078c0b66691670a16b84079139035b4e",
    ),
    ("parallel_count", "9ae6ae0ca54e4c1930356965275e1d442b2f25ba"),
    (
        "token_increment",
        "bacdccafbf0cb85105a45154294f340c6275fe3f",
    ),
];

fn scope(descriptor: &str, value: &str) -> Scope {
    Scope {
        descriptor: descriptor.to_owned(),
        value: value.to_owned(),
    }
}

fn counter(scope: &Scope, kind: Kind) -> Counter {
    Counter {
        scope: scope.clone(),
        kind,
    }
}

fn atomic(scope: &Scope, kind: Kind, limit: i64, increment: i64) -> AtomicCounter {
    AtomicCounter {
        counter: counter(scope, kind),
        limit,
        increment,
        ttl_seconds: 60,
        window_size_seconds: 60,
    }
}

/// `BaseCache.default_ttl`, what `RedisCache` gives an increment that names no TTL.
const REDIS_TTL: Duration = Duration::from_secs(60);

fn store(deployment: &RedisDeployment) -> Arc<RateLimits> {
    Arc::new(RateLimits::new(Some(connect(deployment)), None, REDIS_TTL))
}

async fn raw_text(raw: &RedisClient, key: &str) -> Option<String> {
    raw.get(key).await.unwrap()
}

async fn zcard(raw: &RedisClient, key: &str) -> i64 {
    raw.eval(
        &Script::new("return redis.call('ZCARD', KEYS[1])"),
        &[key],
        Vec::<String>::new(),
    )
    .await
    .unwrap()
}

async fn redis_now(raw: &RedisClient) -> i64 {
    let time: Vec<String> = raw
        .eval(
            &Script::new("return redis.call('TIME')"),
            &["any"],
            Vec::<String>::new(),
        )
        .await
        .unwrap();
    time[0].parse().unwrap()
}

async fn cli(deployment: &RedisDeployment, args: &[&str]) -> Vec<String> {
    match deployment {
        RedisDeployment::Standalone(server) => vec![server.cli(args).await.unwrap()],
        RedisDeployment::Cluster(cluster) => cluster.cli_outputs(args).await.unwrap(),
    }
}

#[rstest]
#[tokio::test]
async fn every_script_runs_mains_bytes_under_mains_keys(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let limits = store(&deployment);
    let now = redis_now(&raw).await;

    for wire in &SCOPES {
        let key = scope(wire.descriptor, wire.value);
        let requests = counter(&key, Kind::Requests);
        let tokens = counter(&key, Kind::Tokens);
        limits
            .sliding_window(None, std::slice::from_ref(&requests), now, 60)
            .unwrap()
            .resolve()
            .await
            .unwrap();
        limits
            .check_and_increment(None, &[atomic(&key, Kind::Tokens, 1000, 40)])
            .unwrap()
            .resolve()
            .await
            .unwrap();
        limits
            .token_increment(
                None,
                &[Increment {
                    counter: tokens.clone(),
                    amount: Number::Float(2.5),
                    ttl: Some(Duration::from_secs(60)),
                }],
            )
            .unwrap()
            .resolve()
            .await
            .unwrap();
        limits
            .parallel_acquire(
                None,
                &[Gauge {
                    scope: key.clone(),
                    limit: 5,
                }],
                SLOT_TTL,
                "slot-1",
            )
            .unwrap()
            .resolve()
            .await
            .unwrap();

        assert_eq!(raw_text(&raw, wire.window).await, Some(now.to_string()));
        assert_eq!(raw_text(&raw, wire.requests).await, Some("1".to_owned()));
        assert_eq!(raw_text(&raw, wire.tokens).await, Some("42.5".to_owned()));
        assert_eq!(zcard(&raw, wire.gauge).await, 1);
        assert_eq!(
            limits
                .read_counters(None, &[requests, tokens])
                .unwrap()
                .resolve()
                .await
                .unwrap(),
            vec![
                Some(now.to_string()),
                Some("1".to_owned()),
                Some(now.to_string()),
                Some("42.5".to_owned()),
            ]
        );
    }

    let guarded = scope("api_key", "sha-check");
    limits
        .guarded_increment(
            None,
            &GuardedIncrement {
                counter: counter(&guarded, Kind::Tokens),
                expected_window_start: "0".to_owned(),
                amount: Number::Int(1),
                ttl: None,
            },
        )
        .unwrap()
        .resolve()
        .await
        .unwrap();
    for pending in [
        limits.parallel_release(None, std::slice::from_ref(&guarded), "slot-1"),
        limits.parallel_count(None, std::slice::from_ref(&guarded), SLOT_TTL),
    ] {
        pending.unwrap().resolve().await.unwrap();
    }
    for (name, sha) in SCRIPTS {
        let cached = cli(&deployment, &["SCRIPT", "EXISTS", sha]).await;
        assert!(
            cached.iter().any(|reply| reply.trim() == "1"),
            "{name} ran under a different SHA than main's {sha}"
        );
    }
}

#[rstest]
#[case::epoch("1730000000", Number::Int(1730000000))]
#[case::int("7", Number::Int(7))]
#[case::float("7.5", Number::Float(7.5))]
#[case::negative("-3", Number::Int(-3))]
#[tokio::test]
async fn reads_values_python_wrote_and_backfills_memory(
    #[case] raw_value: &str,
    #[case] expected: Number,
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let limits = store(&deployment);
    let wire = &SCOPES[0];
    raw.set(wire.tokens, raw_value, Duration::from_secs(60))
        .await
        .unwrap();
    let entry = Entry::Counter(counter(&scope(wire.descriptor, wire.value), Kind::Tokens));

    assert_eq!(
        limits
            .get(&entry, Some(Duration::from_secs(30)))
            .await
            .unwrap(),
        Some(Local::Number(expected))
    );
    assert_eq!(limits.local(&entry).unwrap(), Some(Local::Number(expected)));
}

#[rstest]
#[tokio::test]
async fn sliding_window_opens_counts_and_resets_with_its_sibling(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let limits = store(&deployment);
    let key = scope("api_key", "k1");
    let requests = counter(&key, Kind::Requests);
    let run = |now: i64| {
        let pending = limits
            .sliding_window(None, std::slice::from_ref(&requests), now, 60)
            .unwrap();
        async move { pending.resolve().await.unwrap() }
    };

    assert_eq!(
        run(1_000).await,
        vec![WindowCount {
            window_start: "1000".to_owned(),
            count: 1
        }]
    );
    assert_eq!(run(1_059).await[0].count, 2);
    raw.set("{api_key:k1}:tokens", "500", Duration::from_secs(60))
        .await
        .unwrap();
    let Ttl::Expires(ttl) = raw.ttl("{api_key:k1}:requests").await.unwrap() else {
        panic!("the counter has no TTL");
    };
    assert!(within(ttl, 60), "{ttl:?}");

    assert_eq!(
        run(1_060).await,
        vec![WindowCount {
            window_start: "1060".to_owned(),
            count: 1
        }]
    );
    assert_eq!(raw_text(&raw, "{api_key:k1}:tokens").await, None);
}

#[rstest]
#[tokio::test]
async fn check_and_increment_is_all_or_nothing(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let limits = store(&deployment);
    let key = scope("model_per_key", "k1:gpt");
    let now = redis_now(&raw).await;
    let both = |requests: i64, tokens: i64| {
        [
            atomic(&key, Kind::Requests, 2, requests),
            atomic(&key, Kind::Tokens, 100, tokens),
        ]
    };

    let first = limits
        .check_and_increment(None, &both(1, 60))
        .unwrap()
        .resolve()
        .await
        .unwrap();
    let AtomicReply::Applied(applied) = first else {
        panic!("{first:?}");
    };
    assert_eq!(applied.iter().map(|a| a.count).collect::<Vec<_>>(), [1, 60]);
    assert!(applied.iter().all(|a| (a.window_start - now).abs() <= 2));

    assert_eq!(
        limits
            .check_and_increment(None, &both(1, 41))
            .unwrap()
            .resolve()
            .await
            .unwrap(),
        AtomicReply::OverLimit {
            index: 1,
            current: 60,
            limit: 100
        }
    );
    assert_eq!(
        raw_text(&raw, "{model_per_key:k1:gpt}:requests").await,
        Some("1".to_owned())
    );
    assert_eq!(
        raw_text(&raw, "{model_per_key:k1:gpt}:tokens").await,
        Some("60".to_owned())
    );

    let at_limit = limits
        .check_and_increment(None, &both(1, 40))
        .unwrap()
        .resolve()
        .await
        .unwrap();
    assert!(
        matches!(at_limit, AtomicReply::Applied(ref applied) if applied[0].count == 2 && applied[1].count == 100)
    );
    assert_eq!(
        limits
            .check_and_increment(None, &both(0, 0))
            .unwrap()
            .resolve()
            .await
            .unwrap(),
        AtomicReply::OverLimit {
            index: 0,
            current: 2,
            limit: 2
        }
    );
}

#[rstest]
#[tokio::test]
async fn guarded_increment_only_settles_the_window_it_was_reserved_in(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let limits = store(&deployment);
    let key = scope("team", "t1");
    let tokens = counter(&key, Kind::Tokens);
    let reply = limits
        .check_and_increment(None, &[atomic(&key, Kind::Tokens, 1000, 300)])
        .unwrap()
        .resolve()
        .await
        .unwrap();
    let AtomicReply::Applied(applied) = reply else {
        panic!("{reply:?}");
    };
    let Applied { window_start, .. } = applied[0];
    let guarded = |expected_window_start: String, amount: i64| {
        let pending = limits
            .guarded_increment(
                None,
                &GuardedIncrement {
                    counter: tokens.clone(),
                    expected_window_start,
                    amount: Number::Int(amount),
                    ttl: Some(Duration::from_secs(60)),
                },
            )
            .unwrap();
        async move { pending.resolve().await.unwrap() }
    };

    assert_eq!(
        guarded(window_start.to_string(), -120).await,
        GuardedReply {
            applied: true,
            count: 180
        }
    );
    assert_eq!(
        guarded((window_start - 60).to_string(), -180).await,
        GuardedReply {
            applied: false,
            count: 180
        }
    );
    assert_eq!(
        raw_text(&raw, "{team:t1}:tokens").await,
        Some("180".to_owned())
    );
}

#[rstest]
#[tokio::test]
async fn gauge_slots_are_acquired_released_by_id_and_expire(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let limits = store(&deployment);
    let key = scope("api_key", "k1");
    let gauges = [Gauge {
        scope: key.clone(),
        limit: 2,
    }];
    let acquire = |slot_id: &'static str, slot_ttl: Duration| {
        let pending = limits
            .parallel_acquire(None, &gauges, slot_ttl, slot_id)
            .unwrap();
        async move { pending.resolve().await.unwrap() }
    };
    let release = |slot_id: &'static str| {
        let pending = limits
            .parallel_release(None, std::slice::from_ref(&key), slot_id)
            .unwrap();
        async move { pending.resolve().await.unwrap() }
    };
    let count = |slot_ttl: Duration| {
        let pending = limits
            .parallel_count(None, std::slice::from_ref(&key), slot_ttl)
            .unwrap();
        async move { pending.resolve().await.unwrap() }
    };

    assert_eq!(
        acquire("a", SLOT_TTL).await,
        AcquireReply::Acquired(vec![1])
    );
    assert_eq!(
        acquire("b", SLOT_TTL).await,
        AcquireReply::Acquired(vec![2])
    );
    assert_eq!(
        acquire("c", SLOT_TTL).await,
        AcquireReply::OverLimit {
            index: 0,
            in_flight: 2,
            limit: 2
        }
    );
    assert_eq!(release("never-acquired").await, vec![2]);
    assert_eq!(release("a").await, vec![1]);
    assert_eq!(release("a").await, vec![1]);
    assert_eq!(count(SLOT_TTL).await, vec![1]);

    tokio::time::sleep(Duration::from_millis(2_100)).await;
    assert_eq!(count(Duration::from_secs(1)).await, vec![0]);
    assert_eq!(
        acquire("d", SLOT_TTL).await,
        AcquireReply::Acquired(vec![1])
    );
}

#[rstest]
#[tokio::test]
async fn token_increment_sets_a_ttl_only_on_a_key_without_one(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let limits = store(&deployment);
    let key = scope("user", "u1");
    let increment = |amount: Number| Increment {
        counter: counter(&key, Kind::Tokens),
        amount,
        ttl: Some(Duration::from_secs(60)),
    };
    raw.set("{user:u1}:tokens", "10", Duration::from_secs(600))
        .await
        .unwrap();

    let totals = limits
        .token_increment(
            None,
            &[increment(Number::Float(0.5)), increment(Number::Int(-3))],
        )
        .unwrap()
        .resolve()
        .await
        .unwrap();

    assert_eq!(totals, vec![10.5, 7.5]);
    let Ttl::Expires(ttl) = raw.ttl("{user:u1}:tokens").await.unwrap() else {
        panic!("the counter lost its TTL");
    };
    assert!(within(ttl, 600), "{ttl:?}");
}

#[rstest]
#[tokio::test]
async fn slot_groups_follow_the_cluster_slot_of_each_hash_tag(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let limits = store(&deployment);
    let scopes = [
        scope("api_key", "a"),
        scope("team", "t"),
        scope("api_key", "a"),
        scope("user", "u"),
    ];
    let slot = |key: String| {
        let deployment = &deployment;
        async move {
            match deployment {
                RedisDeployment::Cluster(cluster) => cluster
                    .cli_outputs(&["CLUSTER", "KEYSLOT", &key])
                    .await
                    .unwrap()[0]
                    .trim()
                    .to_owned(),
                RedisDeployment::Standalone(_) => key,
            }
        }
    };
    let mut expected: Vec<(String, Vec<usize>)> = Vec::new();
    for (index, scope) in scopes.iter().enumerate() {
        let slot = slot(format!("{{{}:{}}}:window", scope.descriptor, scope.value)).await;
        match expected.iter_mut().find(|(seen, _)| *seen == slot) {
            Some((_, members)) => members.push(index),
            None => expected.push((slot, vec![index])),
        }
    }
    let by_slot = expected
        .into_iter()
        .map(|(_, members)| members)
        .collect::<Vec<_>>();

    let grouped = limits.slot_groups(&scopes, false);
    match deployment {
        RedisDeployment::Cluster(_) => assert_eq!(grouped, by_slot),
        RedisDeployment::Standalone(_) => assert_eq!(grouped, vec![vec![0, 1, 2, 3]]),
    }
    assert_eq!(
        limits.slot_groups(&scopes, true),
        vec![vec![0, 2], vec![1], vec![3]]
    );
}

#[rstest]
#[tokio::test]
async fn gauges_of_different_slots_in_one_call_fail_only_on_a_cluster(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let limits = store(&deployment);
    let gauges = ["a", "b", "c", "d"].map(|value| Gauge {
        scope: scope("api_key", value),
        limit: 5,
    });

    let acquired = limits
        .parallel_acquire(None, &gauges, SLOT_TTL, "slot")
        .unwrap()
        .resolve()
        .await;

    match deployment {
        RedisDeployment::Standalone(_) => {
            assert_eq!(acquired.unwrap(), AcquireReply::Acquired(vec![1, 1, 1, 1]))
        }
        RedisDeployment::Cluster(_) => assert!(acquired.is_err(), "{acquired:?}"),
    }
}

#[tokio::test]
async fn a_cross_slot_token_script_on_the_post_call_batch_falls_back_to_plain_increments() {
    let deployment = RedisDeployment::start(Topology::Cluster).await.unwrap();
    let raw = connect(&deployment);
    let limits = store(&deployment);
    let request = RequestRedis::new(Handle::current(), Duration::from_secs(60));
    let increments = ["a", "b", "c", "d"].map(|value| Increment {
        counter: counter(&scope("api_key", value), Kind::Tokens),
        amount: Number::Int(25),
        ttl: Some(Duration::from_secs(60)),
    });

    limits
        .token_increment_post_call(&request, increments.to_vec())
        .unwrap();
    assert_eq!(raw_text(&raw, "{api_key:a}:tokens").await, None);
    request.flush_post().await;

    for value in ["a", "b", "c", "d"] {
        let key = format!("{{api_key:{value}}}:tokens");
        let written = litellm_redis_testing::eventually(async || raw_text(&raw, &key).await).await;
        assert_eq!(written, "25");
    }
    let local = limits
        .local(&Entry::Counter(counter(
            &scope("api_key", "a"),
            Kind::Tokens,
        )))
        .unwrap();
    assert_eq!(local, Some(Local::Number(Number::Int(25))));
}

#[rstest]
#[tokio::test]
async fn a_post_call_increment_from_a_thread_without_a_runtime_reaches_redis(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let limits = store(&deployment);
    let request = RequestRedis::new(Handle::current(), Duration::from_secs(60));
    let increment = Increment {
        counter: counter(&scope("end_user", "e1"), Kind::Tokens),
        amount: Number::Int(40),
        ttl: Some(Duration::from_secs(60)),
    };

    std::thread::spawn({
        let (limits, request, increment) =
            (Arc::clone(&limits), request.clone(), increment.clone());
        move || {
            limits
                .increment_many_post_call(&request, std::slice::from_ref(&increment))
                .unwrap()
        }
    })
    .join()
    .expect("a Python thread has no Tokio context, and declaring there must not panic");
    request.flush_post().await;

    let written =
        litellm_redis_testing::eventually(async || raw_text(&raw, "{end_user:e1}:tokens").await)
            .await;
    assert_eq!(written, "40");
}

#[rstest]
#[tokio::test]
async fn post_call_increments_count_in_memory_now_and_reach_redis_with_the_batch(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let limits = store(&deployment);
    let request = RequestRedis::new(Handle::current(), Duration::from_secs(60));
    let increment = Increment {
        counter: counter(&scope("end_user", "e1"), Kind::Tokens),
        amount: Number::Int(-40),
        ttl: Some(Duration::from_secs(60)),
    };
    raw.set("{end_user:e1}:tokens", "100", Duration::from_secs(600))
        .await
        .unwrap();

    limits
        .increment_many_post_call(&request, std::slice::from_ref(&increment))
        .unwrap();
    assert_eq!(
        limits
            .local(&Entry::Counter(increment.counter.clone()))
            .unwrap(),
        Some(Local::Number(Number::Int(-40)))
    );
    assert_eq!(
        raw_text(&raw, "{end_user:e1}:tokens").await,
        Some("100".to_owned())
    );
    request.flush_post().await;

    let written = litellm_redis_testing::eventually(async || {
        raw_text(&raw, "{end_user:e1}:tokens")
            .await
            .filter(|value| value != "100")
    })
    .await;
    assert_eq!(written, "60");
    let Ttl::Expires(ttl) = raw.ttl("{end_user:e1}:tokens").await.unwrap() else {
        panic!("the counter lost its TTL");
    };
    // RedisBatch refreshes the TTL in a pipeline, and on a cluster runs async_increment, which
    // keeps the TTL the counter already had
    match topology {
        Topology::Standalone => assert!(within(ttl, 60), "{ttl:?}"),
        Topology::Cluster => assert!(ttl > Duration::from_secs(60), "{ttl:?}"),
    }
}

#[rstest]
#[tokio::test]
async fn increment_keeps_an_existing_ttl_and_counts_in_memory(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let limits = store(&deployment);
    let increment = Increment {
        counter: counter(&scope("organization", "o1"), Kind::Tokens),
        amount: Number::Int(5),
        ttl: Some(Duration::from_secs(60)),
    };
    raw.set("{organization:o1}:tokens", "10", Duration::from_secs(600))
        .await
        .unwrap();

    assert_eq!(
        limits.increment(&increment).await.unwrap(),
        Number::Float(15.0)
    );
    let Ttl::Expires(ttl) = raw.ttl("{organization:o1}:tokens").await.unwrap() else {
        panic!("the counter lost its TTL");
    };
    assert!(within(ttl, 600), "{ttl:?}");
    assert_eq!(
        limits
            .local(&Entry::Counter(increment.counter.clone()))
            .unwrap(),
        Some(Local::Number(Number::Int(5)))
    );
}

#[tokio::test]
async fn increment_answers_with_memory_while_redis_is_down() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let limits = store(&deployment);
    let increment = Increment {
        counter: counter(&scope("user", "u1"), Kind::Tokens),
        amount: Number::Int(7),
        ttl: Some(Duration::from_secs(60)),
    };
    deployment.stop().await.unwrap();

    assert_eq!(limits.increment(&increment).await.unwrap(), Number::Int(7));
    assert_eq!(limits.increment(&increment).await.unwrap(), Number::Int(14));
    assert!(
        limits
            .sliding_window(None, std::slice::from_ref(&increment.counter), 1_000, 60)
            .unwrap()
            .resolve()
            .await
            .is_err()
    );
}

#[tokio::test]
async fn scripts_need_redis() {
    let limits = RateLimits::new(None, None, REDIS_TTL);
    assert!(
        limits
            .sliding_window(
                None,
                &[counter(&scope("api_key", "k"), Kind::Requests)],
                0,
                60
            )
            .is_err()
    );
    assert_eq!(
        limits.slot_groups(&[scope("a", "1"), scope("b", "2")], false),
        vec![vec![0, 1]]
    );
}

#[rstest]
#[tokio::test]
async fn the_limiter_script_and_the_armed_cooldown_read_share_one_pipeline(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let (client, trips) = connect_observed(&deployment);
    let cooldowns = RouterCooldowns::new(Some(client.clone()), COOLDOWN_POLICY);
    let limits = RateLimits::new(Some(client), None, REDIS_TTL);
    let request = RequestRedis::new(Handle::current(), Duration::from_secs(60));

    let armed = cooldowns
        .declare_active(&request, &["dep-a".to_owned()])
        .unwrap();
    let admission = limits
        .sliding_window(
            Some(&request),
            &[counter(&scope("api_key", "k1"), Kind::Requests)],
            1_000,
            60,
        )
        .unwrap();
    let counts = admission.resolve().await.unwrap();
    let active = cooldowns.resolve_active(armed).await.unwrap();

    assert_eq!(counts[0].count, 1);
    assert!(active.is_empty());
    let mut seen = trips.seen();
    seen.sort_by_key(|operation| format!("{operation:?}"));
    let expected = match topology {
        Topology::Standalone => vec![Operation::Pipeline { ops: 2 }],
        Topology::Cluster => vec![Operation::Command("MGET"), Operation::Script],
    };
    assert_eq!(seen, expected);
}

// DualCache.async_increment_cache without a TTL still passes RedisCache.default_ttl, so
// async_increment sends INCRBYFLOAT and TTL, then EXPIRE on a key that has none, never a script
#[rstest]
#[tokio::test]
async fn an_increment_without_a_ttl_takes_the_redis_default_like_async_increment(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let limits = store(&deployment);
    deployment
        .cli_on_every_node(&["CONFIG", "RESETSTAT"])
        .await
        .unwrap();

    limits
        .increment(&Increment {
            counter: counter(&scope("api_key", "k"), Kind::Tokens),
            amount: Number::Int(3),
            ttl: None,
        })
        .await
        .unwrap();

    let Ttl::Expires(ttl) = raw.ttl("{api_key:k}:tokens").await.unwrap() else {
        panic!("the counter has no TTL");
    };
    assert!(within(ttl, 60), "{ttl:?}");
    assert_eq!(support::calls(&deployment, "incrbyfloat").await, 1);
    assert_eq!(support::calls(&deployment, "expire").await, 1);
    assert_eq!(
        support::calls(&deployment, "eval").await + support::calls(&deployment, "evalsha").await,
        0
    );
}
