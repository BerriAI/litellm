mod support;

use std::{sync::Arc, time::Duration};

use litellm_python_compat::{Value, json};
use litellm_redis_client::{Operation, Ttl, commands};
use litellm_redis_store::{
    RequestRedis, Send as RedisSend,
    router_session_pins::{Eligible, PinKey, RouterSessionPins},
};
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::{connect, remote, within};
use tokio::runtime::Handle;

const KEY: &str = "deployment_affinity:v1:session:group:unscoped:session-1";

fn key() -> PinKey {
    PinKey::DeploymentSession {
        model_group: "group".to_owned(),
        caller: None,
        session_id: "session-1".to_owned(),
    }
}
const TTL: Duration = Duration::from_secs(600);

fn pin(model_id: &str) -> Value {
    json::loads(&format!(r#"{{"model_id": "{model_id}"}}"#)).unwrap()
}

fn store(deployment: &RedisDeployment) -> Arc<RouterSessionPins> {
    Arc::new(RouterSessionPins::new(
        Some(remote(connect(deployment))),
        None,
    ))
}

async fn stored(raw: &litellm_redis_client::RedisClient) -> Option<String> {
    raw.send(commands::get(KEY)).await.unwrap()
}

async fn cli(deployment: &RedisDeployment, args: &[&str]) -> Vec<String> {
    match deployment {
        RedisDeployment::Standalone(server) => vec![server.cli(args).await.unwrap()],
        RedisDeployment::Cluster(cluster) => cluster.cli_outputs(args).await.unwrap(),
    }
}

enum Script {
    Read,
    Claim,
}

// sha1 of _READ_PIN_SCRIPT and _CLAIM_PIN_SCRIPT on main
#[rstest]
#[case::read_pin(Script::Read, "f5f07e272c68ee98239d41ee590bd6d17c2ed2c2")]
#[case::claim_pin(Script::Claim, "e3942e42b49307b19d1f79eaeff8f207bcc96934")]
#[tokio::test]
async fn each_script_runs_under_mains_sha(
    #[case] script: Script,
    #[case] sha: &str,
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let pins = store(&deployment);

    match script {
        Script::Read => {
            pins.read(RedisSend::Now, &key(), TTL)
                .unwrap()
                .await
                .unwrap();
        }
        Script::Claim => {
            pins.claim(&key(), &pin("dep-a"), TTL, &Eligible::default())
                .await
                .unwrap();
        }
    }

    let cached = cli(&deployment, &["SCRIPT", "EXISTS", sha]).await;
    assert!(cached.iter().any(|reply| reply.trim() == "1"), "{cached:?}");
}

// claim_affinity_pin writes json.dumps(pin) with SET ... EX, and the first writer keeps the key
#[rstest]
#[tokio::test]
async fn the_first_claim_wins_and_is_stored_as_mains_json(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let first = store(&deployment);
    let second = store(&deployment);

    let won = first
        .claim(&key(), &pin("dep-a"), TTL, &Eligible::default())
        .await
        .unwrap();
    let lost = second
        .claim(&key(), &pin("dep-b"), TTL, &Eligible::default())
        .await
        .unwrap();

    assert_eq!(won, Some(pin("dep-a")));
    assert_eq!(lost, Some(pin("dep-a")));
    assert_eq!(second.get_local(&key()).unwrap(), Some(pin("dep-a")));
    assert_eq!(
        stored(&raw).await,
        Some(r#"{"model_id": "dep-a"}"#.to_owned())
    );
    let Ttl::Expires(remaining) = raw.send(commands::ttl(KEY)).await.unwrap() else {
        panic!("the pin has no TTL");
    };
    assert!(within(remaining, 600), "{remaining:?}");
}

#[rstest]
#[case::eligible_holder_kept(r#"{"model_id": "dep-a"}"#, false, "dep-a")]
#[case::ineligible_holder_replaced(r#"{"model_id": "dep-gone"}"#, false, "dep-b")]
#[case::legacy_id_kept(r#""dep-a""#, true, "dep-a")]
#[case::legacy_id_replaced_without_the_flag("dep-a", false, "dep-b")]
#[tokio::test]
async fn an_eligible_claim_replaces_only_a_holder_no_longer_eligible(
    #[case] held: &str,
    #[case] preserve_legacy_model_ids: bool,
    #[case] winner: &str,
) {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let raw = connect(&deployment);
    let pins = store(&deployment);
    raw.send(commands::set(KEY, held, TTL)).await.unwrap();
    let eligible = Eligible {
        values: Some(vec![pin("dep-a"), pin("dep-b")]),
        preserve_legacy_model_ids,
    };

    let claimed = pins
        .claim(&key(), &pin("dep-b"), TTL, &eligible)
        .await
        .unwrap()
        .unwrap();

    let expected = if winner == "dep-b" {
        pin("dep-b")
    } else {
        json::loads(held).unwrap()
    };
    assert_eq!(claimed, expected);
}

// read_affinity_pin's script rides the request's pre-call pipeline and refreshes the pin's TTL
#[rstest]
#[tokio::test]
async fn a_pin_read_rides_the_request_batch_and_refreshes_ttl_and_memory(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let (client, trips) = support::connect_observed(&deployment);
    let pins = Arc::new(RouterSessionPins::new(Some(remote(client)), None));
    raw.send(commands::set(
        KEY,
        r#"{"model_id": "dep-a"}"#,
        Duration::from_secs(5),
    ))
    .await
    .unwrap();
    let request = RequestRedis::new(Handle::current(), Duration::from_secs(60));

    let declared = pins
        .read(RedisSend::PreCall(&request), &key(), TTL)
        .unwrap();
    assert!(trips.seen().is_empty());
    request.flush_pre().await;
    let read = declared.await.unwrap();

    let one_script = match topology {
        Topology::Standalone => Operation::Pipeline { ops: 1 },
        Topology::Cluster => Operation::Script,
    };
    assert_eq!(trips.seen(), [one_script]);
    assert_eq!(read, Some(pin("dep-a")));
    assert_eq!(pins.get_local(&key()).unwrap(), Some(pin("dep-a")));
    let Ttl::Expires(remaining) = raw.send(commands::ttl(KEY)).await.unwrap() else {
        panic!("the pin has no TTL");
    };
    assert!(within(remaining, 600), "{remaining:?}");
}

#[rstest]
#[tokio::test]
async fn a_pin_read_missing_in_redis_drops_the_memory_pin() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let pins = store(&deployment);
    pins.set_local(&key(), pin("dep-a"), TTL).unwrap();

    let read = pins
        .read(RedisSend::Now, &key(), TTL)
        .unwrap()
        .await
        .unwrap();

    assert_eq!(read, None);
    assert_eq!(pins.get_local(&key()).unwrap(), None);
}

#[rstest]
#[tokio::test]
async fn redis_failures_reach_the_caller_which_keeps_its_memory_pin() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let pins = store(&deployment);
    pins.set_local(&key(), pin("dep-a"), TTL).unwrap();
    deployment.stop().await.unwrap();

    let read = pins.read(RedisSend::Now, &key(), TTL).unwrap().await;
    let claim = pins
        .claim(&key(), &pin("dep-b"), TTL, &Eligible::default())
        .await;
    let shared = pins.get_shared(&key()).await;

    assert!(read.is_err());
    assert!(claim.is_err());
    assert!(shared.is_err());
    assert_eq!(pins.get_local(&key()).unwrap(), Some(pin("dep-a")));
}

#[rstest]
#[tokio::test]
async fn without_redis_reads_answer_from_memory_and_claims_have_no_winner() {
    let pins = Arc::new(RouterSessionPins::new(None, None));
    pins.set_local(&key(), pin("dep-a"), TTL).unwrap();

    assert_eq!(
        pins.read(RedisSend::Now, &key(), TTL)
            .unwrap()
            .await
            .unwrap(),
        Some(pin("dep-a"))
    );
    assert_eq!(
        pins.claim(&key(), &pin("dep-b"), TTL, &Eligible::default())
            .await
            .unwrap(),
        None
    );
    assert_eq!(pins.get_shared(&key()).await.unwrap(), None);
}

// DualCache.async_set_cache writes json.dumps(value), which every pod's async_get_cache reads back
#[rstest]
#[tokio::test]
async fn a_set_pin_is_mains_json_and_another_router_reads_it_from_redis(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let writer = store(&deployment);
    let reader = store(&deployment);
    let session_pin =
        json::loads(r#"{"model": "anthropic/claude-sonnet-5-5", "tier": "COMPLEX"}"#).unwrap();

    writer.set(&key(), session_pin.clone(), TTL).await.unwrap();

    assert_eq!(
        stored(&raw).await,
        Some(r#"{"model": "anthropic/claude-sonnet-5-5", "tier": "COMPLEX"}"#.to_owned())
    );
    assert_eq!(reader.get(&key()).await.unwrap(), Some(session_pin.clone()));
    assert_eq!(reader.get_local(&key()).unwrap(), Some(session_pin.clone()));
    assert_eq!(reader.get_shared(&key()).await.unwrap(), Some(session_pin));
}

// set_local_affinity_pin deletes before setting, since InMemoryCache.set_cache keeps a live key's
// expiry: a re-pinned key lives for the new TTL
#[rstest]
#[tokio::test]
async fn a_local_pin_set_again_takes_the_new_ttl() {
    let seconds = Arc::new(std::sync::atomic::AtomicU64::new(1_000));
    let now = {
        let seconds = seconds.clone();
        Arc::new(move || Duration::from_secs(seconds.load(std::sync::atomic::Ordering::SeqCst)))
    };
    let pins = RouterSessionPins::with_clock(None, None, now);
    pins.set_local(&key(), pin("dep-a"), Duration::from_secs(10))
        .unwrap();
    seconds.fetch_add(5, std::sync::atomic::Ordering::SeqCst);

    pins.set_local(&key(), pin("dep-b"), TTL).unwrap();
    seconds.fetch_add(30, std::sync::atomic::Ordering::SeqCst);

    assert_eq!(pins.get_local(&key()).unwrap(), Some(pin("dep-b")));
}

// InMemoryCache.get_cache hands a stored string back as json.loads reads it
#[rstest]
#[case::json_text(r#"{ "model" : "first" }"#, r#"{"model": "first"}"#)]
#[case::bare_text("first", r#""first""#)]
#[tokio::test]
async fn a_memory_string_pin_reads_back_as_json_when_it_parses(
    #[case] stored: &str,
    #[case] expected: &str,
) {
    let pins = RouterSessionPins::new(None, None);
    pins.set_local(&key(), Value::Str(stored.to_owned()), TTL)
        .unwrap();

    assert_eq!(
        pins.get_local(&key()).unwrap(),
        Some(json::loads(expected).unwrap())
    );
    assert_eq!(
        pins.get(&key()).await.unwrap(),
        Some(json::loads(expected).unwrap())
    );
}

// DualCache.async_get_cache keeps a pin it read from Redis for the router's default_in_memory_ttl
#[rstest]
#[tokio::test]
async fn a_pin_read_from_redis_stays_in_memory_for_the_default_in_memory_ttl() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let raw = connect(&deployment);
    raw.send(commands::set(KEY, r#"{"model_id": "dep-a"}"#, TTL))
        .await
        .unwrap();
    let seconds = Arc::new(std::sync::atomic::AtomicU64::new(1_000));
    let now = {
        let seconds = seconds.clone();
        Arc::new(move || Duration::from_secs(seconds.load(std::sync::atomic::Ordering::SeqCst)))
    };
    let pins = RouterSessionPins::with_clock(
        Some(remote(connect(&deployment))),
        Some(Duration::from_secs(5)),
        now,
    );

    assert_eq!(pins.get(&key()).await.unwrap(), Some(pin("dep-a")));
    seconds.fetch_add(4, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(pins.get_local(&key()).unwrap(), Some(pin("dep-a")));
    seconds.fetch_add(2, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(pins.get_local(&key()).unwrap(), None);
}

#[rstest]
#[tokio::test]
async fn a_local_claim_keeps_the_first_writer_in_memory_until_it_is_no_longer_eligible() {
    let store = RouterSessionPins::new(None, None);
    let listing_b = Eligible {
        values: Some(vec![pin("b")]),
        preserve_legacy_model_ids: false,
    };

    let first = store
        .claim_local(&key(), &pin("a"), TTL, &Eligible::default())
        .unwrap();
    let second = store
        .claim_local(&key(), &pin("b"), TTL, &Eligible::default())
        .unwrap();
    let replaced = store
        .claim_local(&key(), &pin("b"), TTL, &listing_b)
        .unwrap();

    assert_eq!(
        (first, second, replaced, store.get_local(&key()).unwrap()),
        (pin("a"), pin("a"), pin("b"), Some(pin("b")))
    );
}
