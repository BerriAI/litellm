mod support;

use std::time::Duration;

use litellm_python_compat::Value;
use litellm_redis_client::commands;
use litellm_redis_client::{Operation, RedisClient, Ttl};
use litellm_redis_store::Send as RedisSend;
use litellm_redis_store::{
    Number, Remote, RequestRedis,
    auth_cache::{self, AuthCache, Settings},
    router_cooldowns::RouterCooldowns,
};
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::{COOLDOWN_POLICY, connect, connect_observed, remote, within};
use tokio::runtime::Handle;

const LONG_DEADLINE: Duration = Duration::from_secs(60);
const KEY_OBJECT: &str = "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b";
const TEAM_JSON: &str = concat!(
    r#"{"team_id": "team-\u00e9", "models": ["gpt-5", "claude-opus-5"], "max_budget": null, "#,
    r#""spend": 0.30000000000000004, "blocked": false, "tpm_limit": 10000000000, "#,
    r#""metadata": {"tags": [], "quoted": "it's \"x\""}}"#,
);
const TEAM_STR: &str = concat!(
    "{'team_id': 'team-é', 'models': ['gpt-5', 'claude-opus-5'], 'max_budget': None, ",
    "'spend': 0.30000000000000004, 'blocked': False, 'tpm_limit': 10000000000, ",
    r#"'metadata': {'tags': [], 'quoted': 'it\'s "x"'}}"#,
);

fn dict<const N: usize>(entries: [(&str, Value); N]) -> Value {
    Value::Dict(
        entries
            .into_iter()
            .map(|(key, value)| (Value::from(key), value))
            .collect(),
    )
}

fn strings(items: &[&str]) -> Value {
    Value::List(items.iter().copied().map(Value::from).collect())
}

fn team_record() -> Value {
    dict([
        ("team_id", Value::from("team-é")),
        ("models", strings(&["gpt-5", "claude-opus-5"])),
        ("max_budget", Value::None),
        ("spend", Value::Float(0.30000000000000004)),
        ("blocked", Value::Bool(false)),
        ("tpm_limit", Value::from(10000000000)),
        (
            "metadata",
            dict([
                ("tags", strings(&[])),
                ("quoted", Value::from("it's \"x\"")),
            ]),
        ),
    ])
}

/// The writes `tests/unit/redis/stores/test_auth_cache.py` checks against Python's own
/// `json.dumps` and `str`.
fn written() -> [(Value, &'static str, &'static str); 7] {
    [
        (team_record(), TEAM_JSON, TEAM_STR),
        (
            dict([
                ("token", Value::from(KEY_OBJECT)),
                ("spend", Value::Float(1e-05)),
                ("rpm_limit", Value::from(5)),
                ("expires", Value::from("2026-11-03T00:00:00Z")),
            ]),
            concat!(
                r#"{"token": "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b", "spend": 1e-05, "#,
                r#""rpm_limit": 5, "expires": "2026-11-03T00:00:00Z"}"#,
            ),
            concat!(
                "{'token': '88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b', 'spend': 1e-05, ",
                "'rpm_limit': 5, 'expires': '2026-11-03T00:00:00Z'}",
            ),
        ),
        (
            Value::from("__no_team_membership__"),
            "\"__no_team_membership__\"",
            "__no_team_membership__",
        ),
        (
            strings(&["mcp-a", "mcp-b"]),
            r#"["mcp-a", "mcp-b"]"#,
            "['mcp-a', 'mcp-b']",
        ),
        (Value::Float(12.5), "12.5", "12.5"),
        (Value::from(7), "7", "7"),
        (Value::Bool(true), "true", "True"),
    ]
}

/// The reads `tests/unit/redis/stores/test_auth_cache.py` checks against Python's own
/// `RedisCache._get_cache_logic`.
fn read_back() -> [(&'static str, Value); 5] {
    [
        (TEAM_JSON, team_record()),
        (TEAM_STR, team_record()),
        ("['mcp-a', 'mcp-b']", strings(&["mcp-a", "mcp-b"])),
        (
            "{'user_id': 'u1', 'teams': ['t1'], 'active': True, 'budget': None}",
            dict([
                ("user_id", Value::from("u1")),
                ("teams", strings(&["t1"])),
                ("active", Value::Bool(true)),
                ("budget", Value::None),
            ]),
        ),
        ("\"__NO_MAPPING__\"", Value::from("__NO_MAPPING__")),
    ]
}

fn settings(memory_ttl: Option<u64>) -> Settings {
    Settings {
        memory_ttl: memory_ttl.map(Duration::from_secs),
        ..Settings::default()
    }
}

fn store(client: Option<RedisClient>) -> AuthCache {
    AuthCache::new(client.map(remote), settings(Some(60)))
}

fn team(id: &str) -> Value {
    Value::Dict(vec![(Value::from("team_id"), Value::from(id))])
}

async fn redis_ttl(raw: &RedisClient, key: &str) -> Duration {
    let Ttl::Expires(remaining) = raw.send(commands::ttl(key)).await.unwrap() else {
        panic!("{key} has no TTL");
    };
    remaining
}

#[rstest]
#[case(KEY_OBJECT, true)]
#[case(
    "88DC28D0F030C55ED4AB77ED8FAF098196CB1C05DF778539800C9F1243FE6B4B",
    false
)]
#[case(
    "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4",
    false
)]
#[case(
    "team_id:88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
    false
)]
#[case("team_membership:user-1:team-1", false)]
#[case("end_user_restricted_registry", false)]
fn only_a_bare_lowercase_sha256_digest_is_a_user_key_object(
    #[case] key: &str,
    #[case] key_object: bool,
) {
    assert_eq!(auth_cache::is_key_object(key), key_object, "{key}");
}

#[rstest]
#[tokio::test]
async fn writes_mains_json_and_str_bytes(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = store(Some(connect(&deployment)));

    for (index, (value, json, printed)) in written().into_iter().enumerate() {
        let key = format!("team_id:t{index}");
        store.set(&key, value.clone(), None).await.unwrap();
        assert_eq!(
            raw.send(commands::get::<String>(&key))
                .await
                .unwrap()
                .as_deref(),
            Some(json)
        );
        store.set_str(&key, value, None).await.unwrap();
        assert_eq!(
            raw.send(commands::get::<String>(&key))
                .await
                .unwrap()
                .as_deref(),
            Some(printed)
        );
    }
}

#[rstest]
#[tokio::test]
async fn reads_every_encoding_main_writes_and_backfills_memory(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = store(Some(connect(&deployment)));

    for (index, (bytes, expected)) in read_back().into_iter().enumerate() {
        let key = format!("user_id:u{index}");
        raw.send(commands::set(&key, bytes, Duration::from_secs(60)))
            .await
            .unwrap();

        assert_eq!(store.get(None, &key).await.unwrap(), Some(expected.clone()));
        raw.send(commands::del(&[&key])).await.unwrap();
        assert_eq!(
            store.get(None, &key).await.unwrap(),
            Some(expected),
            "{key}"
        );
    }
    assert_eq!(store.get(None, "user_id:absent").await.unwrap(), None);
}

#[rstest]
#[tokio::test]
async fn ttls_follow_dual_cache_and_the_write_back_rule(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = AuthCache::new(
        Some(Remote {
            client: connect(&deployment),
            default_ttl: Duration::from_secs(45),
        }),
        Settings {
            memory_ttl: Some(Duration::from_secs(20)),
            ..Settings::default()
        },
    );

    store
        .set(
            "team_id:given",
            team("a"),
            Some(Duration::from_secs_f64(30.9)),
        )
        .await
        .unwrap();
    store.set("team_id:default", team("b"), None).await.unwrap();
    store
        .write_back(
            RedisSend::Now,
            vec![
                ("user-1".to_owned(), team("c"), None),
                ("user-2".to_owned(), team("d"), Some(Duration::from_secs(5))),
            ],
        )
        .unwrap()
        .await
        .unwrap();

    assert!(within(redis_ttl(&raw, "team_id:given").await, 30));
    assert!(within(redis_ttl(&raw, "team_id:default").await, 20));
    assert!(within(redis_ttl(&raw, "user-1").await, 45));
    assert!(within(redis_ttl(&raw, "user-2").await, 5));
    assert_eq!(store.get_local("user-1").unwrap(), Some(team("c")));
}

#[tokio::test]
async fn user_key_objects_survive_churn_in_the_management_object_partition() {
    let store = AuthCache::new(
        None,
        Settings {
            memory_entries: 2,
            ..Settings::default()
        },
    );
    store.set(KEY_OBJECT, team("key"), None).await.unwrap();

    for index in 0..5 {
        store
            .set(&format!("team_id:t{index}"), team("t"), None)
            .await
            .unwrap();
    }

    assert_eq!(
        store.get(None, KEY_OBJECT).await.unwrap(),
        Some(team("key"))
    );
    assert_eq!(store.len_local().unwrap(), (2, 1));
}

#[rstest]
#[tokio::test]
async fn a_value_json_refuses_stays_in_memory_and_never_reaches_redis(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = store(Some(connect(&deployment)));
    let members = Value::Set(vec![Value::from("a")]);

    store
        .set("team_id:set", members.clone(), None)
        .await
        .unwrap();

    assert_eq!(store.get_local("team_id:set").unwrap(), Some(members));
    assert_eq!(
        raw.send(commands::get::<String>("team_id:set"))
            .await
            .unwrap(),
        None
    );
}

// RedisCache._pipeline_helper runs json.dumps inside the pipeline's try, so one value it refuses
// aborts every SET of that pipeline, while DualCache has already written them all to memory
#[tokio::test]
async fn a_value_json_refuses_keeps_its_whole_pipeline_out_of_redis() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let raw = connect(&deployment);
    let store = store(Some(connect(&deployment)));
    let members = Value::Set(vec![Value::from("a")]);

    store
        .set_many(
            vec![
                ("team_id:a".to_owned(), team("a")),
                ("team_id:set".to_owned(), members.clone()),
            ],
            None,
        )
        .await
        .unwrap();

    assert_eq!(store.get_local("team_id:a").unwrap(), Some(team("a")));
    assert_eq!(store.get_local("team_id:set").unwrap(), Some(members));
    assert_eq!(
        raw.send(commands::get::<String>("team_id:a"))
            .await
            .unwrap(),
        None
    );
}

// auth_object_prefetch writes every entry to memory before
// RedisCache.async_set_cache_pipeline_with_ttls encodes them, and that encodes all before sending
#[tokio::test]
async fn a_write_back_value_json_refuses_fails_after_memory_and_before_redis() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let raw = connect(&deployment);
    let store = store(Some(connect(&deployment)));
    let members = Value::Set(vec![Value::from("a")]);

    let written = store.write_back(
        RedisSend::Now,
        vec![
            ("team_id:set".to_owned(), members.clone(), None),
            ("team_id:a".to_owned(), team("a"), None),
        ],
    );

    assert!(written.is_err());
    assert_eq!(store.get_local("team_id:set").unwrap(), Some(members));
    assert_eq!(store.get_local("team_id:a").unwrap(), Some(team("a")));
    assert_eq!(
        raw.send(commands::get::<String>("team_id:a"))
            .await
            .unwrap(),
        None
    );
}

#[rstest]
#[tokio::test]
async fn pre_call_writes_and_reads_ride_one_pipeline_with_the_cooldown_read(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let (client, trips) = connect_observed(&deployment);
    let cooldowns = std::sync::Arc::new(RouterCooldowns::new(
        Some(remote(client.clone())),
        COOLDOWN_POLICY,
    ));
    let store = store(Some(client));
    let request = RequestRedis::new(Handle::current(), LONG_DEADLINE);
    raw.send(commands::set(
        KEY_OBJECT,
        "{\"token\": \"t\"}",
        Duration::from_secs(60),
    ))
    .await
    .unwrap();
    raw.send(commands::set(
        "team_alias:alpha",
        "\"stale\"",
        Duration::from_secs(60),
    ))
    .await
    .unwrap();

    let reads = store
        .get_many(
            RedisSend::PreCall(&request),
            &[KEY_OBJECT.to_owned(), "end_user_id:absent".to_owned()],
        )
        .unwrap();
    let active = cooldowns
        .active(RedisSend::PreCall(&request), &["dep-a".to_owned()])
        .unwrap();
    let written = store
        .declare_set(RedisSend::PreCall(&request), "team_id:t1", team("t1"), None)
        .unwrap();
    let deleted = store
        .declare_delete(RedisSend::PreCall(&request), "team_alias:alpha")
        .unwrap();

    assert!(trips.seen().is_empty());
    assert_eq!(store.get_local("team_id:t1").unwrap(), Some(team("t1")));
    request.flush_pre().await;

    let mut seen = trips.seen();
    seen.sort_by_key(|operation| format!("{operation:?}"));
    let expected = match topology {
        Topology::Standalone => vec![Operation::Pipeline { ops: 4 }],
        Topology::Cluster => vec![
            Operation::Command("DEL"),
            Operation::Command("MGET"),
            Operation::Command("MGET"),
            Operation::Command("SET"),
        ],
    };
    assert_eq!(seen, expected);
    assert_eq!(
        reads.await.unwrap(),
        vec![
            Some(Value::Dict(vec![(Value::from("token"), Value::from("t"))])),
            None
        ]
    );
    assert_eq!(active.await.unwrap(), vec![]);
    written.await.unwrap();
    deleted.await.unwrap();
    assert_eq!(
        raw.send(commands::get::<String>("team_id:t1"))
            .await
            .unwrap(),
        Some("{\"team_id\": \"t1\"}".to_owned())
    );
    assert!(within(redis_ttl(&raw, "team_id:t1").await, 60));
    assert_eq!(
        raw.send(commands::get::<String>("team_alias:alpha"))
            .await
            .unwrap(),
        None
    );
    assert_eq!(trips.seen().len(), expected.len());
}

#[rstest]
#[tokio::test]
async fn a_write_back_on_the_request_waits_for_the_next_flush(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let (client, trips) = connect_observed(&deployment);
    let store = store(Some(client));
    let request = RequestRedis::new(Handle::current(), LONG_DEADLINE);

    store
        .write_back(
            RedisSend::PreCall(&request),
            vec![(
                "user-1".to_owned(),
                team("u"),
                Some(Duration::from_secs(42)),
            )],
        )
        .unwrap()
        .await
        .unwrap();

    assert!(trips.seen().is_empty());
    assert_eq!(store.get_local("user-1").unwrap(), Some(team("u")));
    request.flush_pre().await;
    let write = match topology {
        Topology::Standalone => Operation::Pipeline { ops: 1 },
        Topology::Cluster => Operation::Command("SET"),
    };
    assert_eq!(trips.seen(), [write]);
    assert!(within(redis_ttl(&raw, "user-1").await, 42));
}

#[rstest]
#[tokio::test]
async fn an_eviction_broadcast_drops_only_this_pods_copy_and_the_next_read_refills_from_redis(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let this_pod = store(Some(connect(&deployment)));
    let other_pod = store(Some(connect(&deployment)));
    this_pod.set("team_id:t1", team("old"), None).await.unwrap();
    other_pod
        .set("team_id:t1", team("new"), None)
        .await
        .unwrap();

    assert_eq!(
        this_pod.get(None, "team_id:t1").await.unwrap(),
        Some(team("old"))
    );
    this_pod.delete_local("team_id:t1").unwrap();

    assert_eq!(
        this_pod.get(None, "team_id:t1").await.unwrap(),
        Some(team("new"))
    );
    assert!(
        raw.send(commands::get::<String>("team_id:t1"))
            .await
            .unwrap()
            .is_some()
    );
}

// RedisBatch.read_as_missing: once the request's MGET found a key absent, or the request deleted it,
// a GET of it is answered without a round trip until the request sets it again
#[rstest]
#[tokio::test]
async fn a_key_the_request_read_as_missing_is_not_read_again(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = store(Some(connect(&deployment)));
    let request = RequestRedis::new(Handle::current(), LONG_DEADLINE);
    let other_pod_writes = |key: &'static str| {
        raw.send(commands::set(
            key,
            TEAM_JSON.to_owned(),
            Duration::from_secs(60),
        ))
    };
    let read = store
        .get_many(RedisSend::PreCall(&request), &["team_id:read".to_owned()])
        .unwrap();
    assert_eq!(read.await.unwrap(), vec![None]);
    store
        .declare_delete(RedisSend::PreCall(&request), "team_id:deleted")
        .unwrap()
        .await
        .unwrap();
    other_pod_writes("team_id:read").await.unwrap();
    other_pod_writes("team_id:deleted").await.unwrap();

    assert_eq!(
        store.get(Some(&request), "team_id:read").await.unwrap(),
        None
    );
    assert_eq!(
        store.get(Some(&request), "team_id:deleted").await.unwrap(),
        None
    );
    assert_eq!(
        store.get(None, "team_id:read").await.unwrap(),
        Some(team_record())
    );

    store
        .declare_set(
            RedisSend::PreCall(&request),
            "team_id:deleted",
            team("t"),
            None,
        )
        .unwrap()
        .await
        .unwrap();
    store.delete_local("team_id:deleted").unwrap();
    assert_eq!(
        store.get(Some(&request), "team_id:deleted").await.unwrap(),
        Some(team("t"))
    );
}

#[rstest]
#[tokio::test]
async fn delete_drops_memory_and_redis(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = store(Some(connect(&deployment)));
    let keys = [KEY_OBJECT.to_owned(), "team_id:t1".to_owned()];
    store
        .set_many(
            keys.iter().map(|key| (key.clone(), team("t"))).collect(),
            None,
        )
        .await
        .unwrap();
    assert!(
        raw.send(commands::get::<String>("team_id:t1"))
            .await
            .unwrap()
            .is_some()
    );

    store.delete(&keys).await.unwrap();

    for key in &keys {
        assert_eq!(store.get_local(key).unwrap(), None);
        assert_eq!(raw.send(commands::get::<String>(key)).await.unwrap(), None);
    }
}

#[rstest]
#[tokio::test]
async fn increment_if_present_only_raises_an_entry_redis_holds(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = store(Some(connect(&deployment)));

    assert_eq!(
        store
            .increment_if_present("litellm-proxy-budget:spend", Number::Float(2.5))
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        raw.send(commands::get::<String>("litellm-proxy-budget:spend"))
            .await
            .unwrap(),
        None
    );
    store
        .set("litellm-proxy-budget:spend", Value::Float(1.5), None)
        .await
        .unwrap();

    assert_eq!(
        store
            .increment_if_present("litellm-proxy-budget:spend", Number::Float(2.5))
            .await
            .unwrap(),
        Some(4.0)
    );
}

#[tokio::test]
async fn without_redis_every_operation_is_memory_only() {
    let store = store(None);
    let request = RequestRedis::new(Handle::current(), LONG_DEADLINE);

    store.set("team_id:t1", team("t"), None).await.unwrap();

    store
        .declare_set(RedisSend::PreCall(&request), "team_id:t2", team("t"), None)
        .unwrap()
        .await
        .unwrap();
    assert_eq!(store.get_local("team_id:t2").unwrap(), Some(team("t")));
    store
        .declare_delete(RedisSend::PreCall(&request), "team_id:t2")
        .unwrap()
        .await
        .unwrap();
    assert_eq!(store.get_local("team_id:t2").unwrap(), None);
    assert!(
        store
            .get_many(RedisSend::PreCall(&request), &["team_id:t1".to_owned()])
            .is_none()
    );
    assert!(
        store
            .get_many(RedisSend::Now, &["team_id:t1".to_owned()])
            .is_none()
    );
    assert_eq!(
        store
            .increment_if_present("k", Number::Float(1.0))
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        store.get(None, "team_id:t1").await.unwrap(),
        Some(team("t"))
    );
    assert_eq!(store.get_local("team_id:t2").unwrap(), None);
}

// UserApiKeyCache writes the key-object partition, then the rest, and deletes both partitions
// concurrently, while async_increment_if_present sends its script as a plain EVAL
#[rstest]
#[tokio::test]
async fn the_auth_cache_sends_the_commands_user_api_key_cache_sends(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let (client, trips) = connect_observed(&deployment);
    let store = store(Some(client));
    deployment
        .cli_on_every_node(&["CONFIG", "RESETSTAT"])
        .await
        .unwrap();

    store
        .set_many(
            vec![
                (KEY_OBJECT.to_owned(), team("t0")),
                ("team_id:t1".to_owned(), team("t1")),
            ],
            None,
        )
        .await
        .unwrap();
    let written = trips.seen();
    store
        .delete(&[KEY_OBJECT.to_owned(), "team_id:t1".to_owned()])
        .await
        .unwrap();
    store
        .increment_if_present("litellm-proxy-budget:spend", Number::Int(5))
        .await
        .unwrap();

    assert_eq!(written, vec![Operation::Pipeline { ops: 1 }; 2]);
    assert_eq!(
        trips.seen()[written.len()..written.len() + 2],
        [Operation::Command("DEL"), Operation::Command("DEL")]
    );
    assert_eq!(support::calls(&deployment, "eval").await, 1);
    assert_eq!(support::calls(&deployment, "evalsha").await, 0);
}
