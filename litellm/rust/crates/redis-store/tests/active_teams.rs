mod support;

use std::{sync::Arc, time::Duration};

use litellm_redis_client::{RedisClient, Ttl, commands};
use litellm_redis_store::{
    Clock,
    active_teams::{ActiveMember, ActiveTeams},
};
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::{connect, remote, within};

/// The same literals sit in `tests/unit/redis/stores/test_active_teams.py`, where main's Python
/// is checked to spell them.
const KEY: &str = "dynamic_rate_limiter:active_teams:chat";
const TEAM: &str = "team\u{1f}t1";
const TEAM_MEMBER: &str = "team\u{1f}t1\u{1f}interactive";
const NOW: u64 = 1_000;
const WINDOW: Duration = Duration::from_secs(60);

fn at(second: u64) -> Clock {
    Arc::new(move || Duration::from_secs(second))
}

fn active(member: &str, priority_class: &str) -> ActiveMember {
    ActiveMember {
        member: member.to_owned(),
        priority_class: priority_class.to_owned(),
    }
}

async fn scored(raw: &RedisClient) -> Vec<String> {
    raw.send(commands::eval_source(
        "return redis.call('ZRANGE', KEYS[1], 0, -1, 'WITHSCORES')",
        &[KEY],
        Vec::<String>::new(),
    ))
    .await
    .unwrap()
}

async fn seen_at(raw: &RedisClient, member: &str, second: u64) {
    let _: i64 = raw
        .send(commands::eval_source(
            "return redis.call('ZADD', KEYS[1], ARGV[1], ARGV[2])",
            &[KEY],
            vec![second.to_string(), member.to_owned()],
        ))
        .await
        .unwrap();
}

#[rstest]
#[tokio::test]
async fn a_member_one_pod_records_is_listed_by_another_under_mains_key_and_member(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let writer = ActiveTeams::with_clock(Some(remote(connect(&deployment))), at(NOW));
    let reader = ActiveTeams::with_clock(Some(remote(connect(&deployment))), at(NOW + 1));

    let alone = writer
        .touch("chat", TEAM, "interactive", WINDOW)
        .await
        .unwrap();

    let Ttl::Expires(remaining) = raw.send(commands::ttl(KEY)).await.unwrap() else {
        panic!("{KEY} has no TTL");
    };
    assert_eq!(alone, [active(TEAM, "interactive")]);
    assert_eq!(scored(&raw).await, [TEAM_MEMBER, "1000"]);
    assert!(within(remaining, 60), "{remaining:?}");
    assert_eq!(
        reader
            .touch("chat", "key\u{1f}k1", "batch", WINDOW)
            .await
            .unwrap(),
        [active(TEAM, "interactive"), active("key\u{1f}k1", "batch")]
    );
}

#[rstest]
#[tokio::test]
async fn members_main_wrote_are_listed_oldest_first_and_dropped_a_window_after_they_were_seen(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    seen_at(&raw, "team\u{1f}newer\u{1f}batch", 990).await;
    seen_at(&raw, "team\u{1f}older\u{1f}batch", 941).await;
    seen_at(&raw, "team\u{1f}gone\u{1f}batch", 940).await;
    let teams = ActiveTeams::with_clock(Some(remote(connect(&deployment))), at(NOW));

    let listed = teams
        .touch("chat", TEAM, "interactive", WINDOW)
        .await
        .unwrap();

    assert_eq!(
        listed,
        [
            active("team\u{1f}older", "batch"),
            active("team\u{1f}newer", "batch"),
            active(TEAM, "interactive"),
        ]
    );
}

#[rstest]
#[tokio::test]
async fn a_redis_that_is_down_fails_the_touch_and_never_answers_from_memory() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let teams = ActiveTeams::with_clock(Some(remote(connect(&deployment))), at(NOW));
    deployment.stop().await.unwrap();

    let touched = teams.touch("chat", TEAM, "interactive", WINDOW).await;

    assert!(touched.is_err(), "{touched:?}");
}
