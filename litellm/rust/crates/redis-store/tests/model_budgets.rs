mod support;

use std::time::Duration;

use jiff::Timestamp;
use litellm_redis_client::commands;
use litellm_redis_store::{
    Remote,
    model_budgets::{Counter, ModelBudgets, Period, Scope},
};
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::connect;

#[rstest]
#[case::key(Scope::Key, "virtual_key_spend", "virtual_key_budget_start_time")]
#[case::user(Scope::User, "user_model_spend", "user_model_budget_start_time")]
#[case::team(Scope::Team, "team_model_spend", "team_model_budget_start_time")]
#[case::end_user(Scope::EndUser, "end_user_model_spend", "end_user_budget_start_time")]
#[case::team_member(
    Scope::TeamMember,
    "team_member_model_spend",
    "team_member_model_budget_start_time"
)]
#[tokio::test]
async fn charges_share_the_python_counter_and_start_new_windows(
    #[case] scope: Scope,
    #[case] spend_prefix: &str,
    #[case] start_prefix: &str,
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let redis = connect(&deployment);
    let ttl = Duration::from_secs(3600);
    let spend_key = format!("{spend_prefix}:caller:claude*:1h");
    let start_key = format!("{start_prefix}:caller:claude*:1h");
    let remote = Remote {
        client: redis.clone(),
        default_ttl: ttl,
    };
    let store = ModelBudgets::new(Some(remote.clone()));
    let counter = Counter {
        scope,
        id: "caller",
        model: "claude*",
        duration: "1h",
    };
    let period = Period::parse(counter.duration).unwrap();
    let now = Timestamp::from_second(1_800_000_000).unwrap();

    assert_eq!(store.spend(&counter).await.unwrap(), 0.0);
    redis
        .send(commands::set(&spend_key, "0.25", ttl))
        .await
        .unwrap();
    redis
        .send(commands::set(&start_key, "1800000000.0", ttl))
        .await
        .unwrap();
    assert_eq!(store.spend(&counter).await.unwrap(), 0.25);
    store.charge(&counter, period, now, 0.125).await.unwrap();
    assert_eq!(
        redis.send(commands::get::<f64>(&spend_key)).await.unwrap(),
        Some(0.375)
    );

    let restarted = ModelBudgets::new(Some(remote));
    let next_window = Timestamp::from_second(now.as_second() + 3601).unwrap();
    restarted
        .charge(&counter, period, next_window, 0.125)
        .await
        .unwrap();
    assert_eq!(
        redis.send(commands::get::<f64>(&spend_key)).await.unwrap(),
        Some(0.125)
    );
    assert_eq!(
        redis.send(commands::get::<f64>(&start_key)).await.unwrap(),
        Some(next_window.as_second() as f64),
    );
}
