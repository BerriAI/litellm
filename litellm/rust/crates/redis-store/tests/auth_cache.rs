mod support;

use std::time::Duration;

use litellm_python_compat::Value;
use litellm_redis_client::commands;
use litellm_redis_client::{Operation, RedisClient, Ttl};
use litellm_redis_store::Send as RedisSend;
use litellm_redis_store::{
    Number, Remote, RequestRedis,
    auth_cache::{AuthCache, AuthObject, Settings},
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

fn team_id(id: &str) -> AuthObject {
    AuthObject::Team(id.to_owned())
}

fn user(id: &str) -> AuthObject {
    AuthObject::User(id.to_owned())
}

fn key_object() -> AuthObject {
    AuthObject::Key(KEY_OBJECT.to_owned())
}

fn owned(ids: &[&str]) -> Vec<String> {
    ids.iter().map(|id| (*id).to_owned()).collect()
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

/// The spellings `tests/unit/redis/stores/test_auth_cache.py` checks against `main`'s f-strings.
#[rstest]
#[case(AuthObject::Key(KEY_OBJECT.to_owned()), KEY_OBJECT)]
#[case(AuthObject::User("user-1".to_owned()), "user-1")]
#[case(AuthObject::Team("team-1".to_owned()), "team_id:team-1")]
#[case(AuthObject::TeamAlias("alpha".to_owned()), "team_alias:alpha")]
#[case(
    AuthObject::TeamMembership { team_id: "team_1".to_owned(), user_id: "user_1".to_owned() },
    "team_1_user_1"
)]
#[case(
    AuthObject::TeamMembershipReservation { user_id: "user-1".to_owned(), team_id: "team-1".to_owned() },
    "team_membership:user-1:team-1"
)]
#[case(AuthObject::Org("org-1".to_owned()), "org_id:org-1")]
#[case(AuthObject::OrgWithBudget("org-1".to_owned()), "org_id:org-1:with_budget")]
#[case(
    AuthObject::OrgWithBudgetLastKnown("org-1".to_owned()),
    "org_id:org-1:with_budget:last_known"
)]
#[case(AuthObject::OrgAlias("acme".to_owned()), "org_alias:acme")]
#[case(AuthObject::Project("p-1".to_owned()), "project_id:p-1")]
#[case(AuthObject::EndUser("eu-1".to_owned()), "end_user_id:eu-1")]
#[case(AuthObject::EndUserRestrictedRegistry, "end_user_restricted_registry")]
#[case(AuthObject::EndUserValidation("eu 1".to_owned()), "end_user_validation:eu 1")]
#[case(AuthObject::CustomerMember("m-1".to_owned()), "customer_member_id:m-1")]
#[case(AuthObject::CustomerOrgAlias("acme".to_owned()), "customer_org_alias:acme")]
#[case(AuthObject::DefaultEndUserBudget("b-1".to_owned()), "default_end_user_budget:b-1")]
#[case(
    AuthObject::TeamMemberDefaultBudget("b-1".to_owned()),
    "team_member_default_budget:b-1"
)]
#[case(AuthObject::Tag("prod".to_owned()), "tag:prod")]
#[case(AuthObject::TagRegistry, "tag_registry")]
#[case(AuthObject::ModelAccessGroup("beta".to_owned()), "model_access_group:beta")]
#[case(AuthObject::ModelAccessGroupRegistry, "model_access_group_registry")]
#[case(AuthObject::AccessGroup("ag-1".to_owned()), "access_group_id:ag-1")]
#[case(AuthObject::AccessGroupRuleName("rule".to_owned()), "access_group_rule_name:rule")]
#[case(AuthObject::ObjectPermission("op-1".to_owned()), "object_permission_id:op-1")]
#[case(
    AuthObject::UserObjectPermission("user-1".to_owned()),
    "user_object_permission_id:user-1"
)]
#[case(
    AuthObject::AgentObjectPermission("agent-1".to_owned()),
    "agent_object_permission_id:agent-1"
)]
#[case(AuthObject::ManagedVectorStore("vs-1".to_owned()), "managed_vector_store_id:vs-1")]
#[case(
    AuthObject::JwtKeyMapping { issuer: None, claim_name: "sub".to_owned(), claim_value: "a:b".to_owned() },
    "jwt_key_mapping:sub:a:b"
)]
#[case(
    AuthObject::JwtKeyMapping {
        issuer: Some("https://idp".to_owned()),
        claim_name: "sub".to_owned(),
        claim_value: "a".to_owned(),
    },
    "jwt_key_mapping:https://idp:sub:a"
)]
#[case(
    AuthObject::AgentIdentity { tenant_id: "t\u{e9}".to_owned(), client_id: "c\"1".to_owned() },
    r#"agent_identity:["t\u00e9", "c\"1"]"#
)]
#[case(AuthObject::GlobalProxySpend, "default_user_id:spend")]
fn each_object_is_cached_under_mains_key(#[case] object: AuthObject, #[case] key: &str) {
    assert_eq!(object.key(), key);
}

#[rstest]
#[case(AuthObject::Key(KEY_OBJECT.to_owned()), true)]
#[case(AuthObject::User(KEY_OBJECT.to_owned()), false)]
#[case(AuthObject::Team(KEY_OBJECT.to_owned()), false)]
fn only_a_virtual_key_is_a_user_key_object(#[case] object: AuthObject, #[case] key_object: bool) {
    assert_eq!(object.is_key_object(), key_object);
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
        let object = team_id(&format!("t{index}"));
        let key = object.key();
        store.set(None, &object, value.clone(), None).await.unwrap();
        assert_eq!(
            raw.send(commands::get::<String>(&key))
                .await
                .unwrap()
                .as_deref(),
            Some(json)
        );
        store.set_str(None, &object, value, None).await.unwrap();
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
        let object = user(&format!("u{index}"));
        let key = object.key();
        raw.send(commands::set(&key, bytes, Duration::from_secs(60)))
            .await
            .unwrap();

        assert_eq!(
            store.get(None, &object).await.unwrap(),
            Some(expected.clone())
        );
        raw.send(commands::del(&[&key])).await.unwrap();
        assert_eq!(
            store.get(None, &object).await.unwrap(),
            Some(expected),
            "{key}"
        );
    }
    assert_eq!(store.get(None, &user("absent")).await.unwrap(), None);
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
            None,
            &team_id("given"),
            team("a"),
            Some(Duration::from_secs_f64(30.9)),
        )
        .await
        .unwrap();
    store
        .set(None, &team_id("default"), team("b"), None)
        .await
        .unwrap();
    store
        .write_back(
            RedisSend::Now,
            vec![
                (user("user-1"), team("c"), None),
                (user("user-2"), team("d"), Some(Duration::from_secs(5))),
            ],
        )
        .unwrap()
        .await
        .unwrap();

    assert!(within(redis_ttl(&raw, "team_id:given").await, 30));
    assert!(within(redis_ttl(&raw, "team_id:default").await, 20));
    assert!(within(redis_ttl(&raw, "user-1").await, 45));
    assert!(within(redis_ttl(&raw, "user-2").await, 5));
    assert_eq!(store.get_local(&user("user-1")).unwrap(), Some(team("c")));
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
    store
        .set(None, &key_object(), team("key"), None)
        .await
        .unwrap();

    for index in 0..5 {
        store
            .set(None, &team_id(&format!("t{index}")), team("t"), None)
            .await
            .unwrap();
    }

    assert_eq!(
        store.get(None, &key_object()).await.unwrap(),
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
        .set(None, &team_id("set"), members.clone(), None)
        .await
        .unwrap();

    assert_eq!(store.get_local(&team_id("set")).unwrap(), Some(members));
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
            None,
            vec![(team_id("a"), team("a")), (team_id("set"), members.clone())],
            None,
        )
        .await
        .unwrap();

    assert_eq!(store.get_local(&team_id("a")).unwrap(), Some(team("a")));
    assert_eq!(store.get_local(&team_id("set")).unwrap(), Some(members));
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
            (team_id("set"), members.clone(), None),
            (team_id("a"), team("a"), None),
        ],
    );

    assert!(written.is_err());
    assert_eq!(store.get_local(&team_id("set")).unwrap(), Some(members));
    assert_eq!(store.get_local(&team_id("a")).unwrap(), Some(team("a")));
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
            &[key_object(), AuthObject::EndUser("absent".to_owned())],
        )
        .unwrap();
    let active = cooldowns
        .active(RedisSend::PreCall(&request), &["dep-a".to_owned()])
        .unwrap();
    let written = store
        .declare_set(
            RedisSend::PreCall(&request),
            &team_id("t1"),
            team("t1"),
            None,
        )
        .unwrap();
    let deleted = store
        .declare_delete(RedisSend::PreCall(&request), "team_alias:alpha")
        .unwrap();

    assert!(trips.seen().is_empty());
    assert_eq!(store.get_local(&team_id("t1")).unwrap(), Some(team("t1")));
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
            vec![(user("user-1"), team("u"), Some(Duration::from_secs(42)))],
        )
        .unwrap()
        .await
        .unwrap();

    assert!(trips.seen().is_empty());
    assert_eq!(store.get_local(&user("user-1")).unwrap(), Some(team("u")));
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
    this_pod
        .set(None, &team_id("t1"), team("old"), None)
        .await
        .unwrap();
    other_pod
        .set(None, &team_id("t1"), team("new"), None)
        .await
        .unwrap();

    assert_eq!(
        this_pod.get(None, &team_id("t1")).await.unwrap(),
        Some(team("old"))
    );
    this_pod.delete_local("team_id:t1").unwrap();

    assert_eq!(
        this_pod.get(None, &team_id("t1")).await.unwrap(),
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
        .get_many(RedisSend::PreCall(&request), &[team_id("read")])
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
        store.get(Some(&request), &team_id("read")).await.unwrap(),
        None
    );
    assert_eq!(
        store
            .get(Some(&request), &team_id("deleted"))
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        store.get(None, &team_id("read")).await.unwrap(),
        Some(team_record())
    );

    store
        .declare_set(
            RedisSend::PreCall(&request),
            &team_id("deleted"),
            team("t"),
            None,
        )
        .unwrap()
        .await
        .unwrap();
    store.delete_local("team_id:deleted").unwrap();
    assert_eq!(
        store
            .get(Some(&request), &team_id("deleted"))
            .await
            .unwrap(),
        Some(team("t"))
    );
}

enum RequestWrite {
    Set,
    SetStr,
    SetMany,
    WriteBack,
}

#[rstest]
#[case::set(RequestWrite::Set, team("t"))]
#[case::set_str(RequestWrite::SetStr, Value::Float(1.5))]
#[case::set_many(RequestWrite::SetMany, team("t"))]
#[case::write_back(RequestWrite::WriteBack, team("t"))]
#[tokio::test]
async fn a_write_in_the_request_makes_a_key_it_read_as_missing_readable_again(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[case] write: RequestWrite,
    #[case] written: Value,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let store = store(Some(connect(&deployment)));
    let request = RequestRedis::new(Handle::current(), LONG_DEADLINE);
    let object = team_id("t");
    let read = store
        .get_many(RedisSend::PreCall(&request), std::slice::from_ref(&object))
        .unwrap();
    assert_eq!(read.await.unwrap(), vec![None]);

    match write {
        RequestWrite::Set => store
            .set(Some(&request), &object, written.clone(), None)
            .await
            .unwrap(),
        RequestWrite::SetStr => store
            .set_str(Some(&request), &object, written.clone(), None)
            .await
            .unwrap(),
        RequestWrite::SetMany => store
            .set_many(
                Some(&request),
                vec![(object.clone(), written.clone())],
                None,
            )
            .await
            .unwrap(),
        RequestWrite::WriteBack => store
            .write_back(
                RedisSend::PreCall(&request),
                vec![(object.clone(), written.clone(), None)],
            )
            .unwrap()
            .await
            .unwrap(),
    }
    request.flush_pre().await;
    store.delete_local(&object.key()).unwrap();

    assert_eq!(
        store.get(Some(&request), &object).await.unwrap(),
        Some(written)
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
    let objects = [key_object(), team_id("t1")];
    let keys: Vec<String> = objects.iter().map(AuthObject::key).collect();
    store
        .set_many(
            None,
            objects
                .iter()
                .map(|object| (object.clone(), team("t")))
                .collect(),
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

    for (object, key) in objects.iter().zip(&keys) {
        assert_eq!(store.get_local(object).unwrap(), None);
        assert_eq!(raw.send(commands::get::<String>(key)).await.unwrap(), None);
    }
}

#[tokio::test]
async fn an_eviction_by_wire_key_drops_a_hex_user_id_and_a_virtual_key_from_their_partitions() {
    let store = store(None);
    store
        .set(None, &user(KEY_OBJECT), team("u"), None)
        .await
        .unwrap();
    store
        .set(None, &key_object(), team("k"), None)
        .await
        .unwrap();
    assert_eq!(store.len_local().unwrap(), (1, 1));

    store.delete(&owned(&[KEY_OBJECT])).await.unwrap();

    assert_eq!(store.len_local().unwrap(), (0, 0));
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
            .increment_if_present(&AuthObject::GlobalProxySpend, Number::Float(2.5))
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        raw.send(commands::get::<String>("default_user_id:spend"))
            .await
            .unwrap(),
        None
    );
    store
        .set(None, &AuthObject::GlobalProxySpend, Value::Float(1.5), None)
        .await
        .unwrap();

    assert_eq!(
        store
            .increment_if_present(&AuthObject::GlobalProxySpend, Number::Float(2.5))
            .await
            .unwrap(),
        Some(4.0)
    );
}

#[tokio::test]
async fn without_redis_every_operation_is_memory_only() {
    let store = store(None);
    let request = RequestRedis::new(Handle::current(), LONG_DEADLINE);

    store
        .set(None, &team_id("t1"), team("t"), None)
        .await
        .unwrap();

    store
        .declare_set(
            RedisSend::PreCall(&request),
            &team_id("t2"),
            team("t"),
            None,
        )
        .unwrap()
        .await
        .unwrap();
    assert_eq!(store.get_local(&team_id("t2")).unwrap(), Some(team("t")));
    store
        .declare_delete(RedisSend::PreCall(&request), "team_id:t2")
        .unwrap()
        .await
        .unwrap();
    assert_eq!(store.get_local(&team_id("t2")).unwrap(), None);
    assert!(
        store
            .get_many(RedisSend::PreCall(&request), &[team_id("t1")])
            .is_none()
    );
    assert!(store.get_many(RedisSend::Now, &[team_id("t1")]).is_none());
    assert_eq!(
        store
            .increment_if_present(&AuthObject::GlobalProxySpend, Number::Float(1.0))
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        store.get(None, &team_id("t1")).await.unwrap(),
        Some(team("t"))
    );
    assert_eq!(store.get_local(&team_id("t2")).unwrap(), None);
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
            None,
            vec![(key_object(), team("t0")), (team_id("t1"), team("t1"))],
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
        .increment_if_present(&AuthObject::GlobalProxySpend, Number::Int(5))
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
