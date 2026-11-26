//! The auth cache's typed team entry on real Redis. The entry literals are the ones
//! `tests/unit/redis/stores/test_auth_cache_team.py` checks against Python's own model and team
//! writer and reader, and `auth-store/tests/team.rs` finds after a database read.
//! Change all three together.

mod support;

use std::time::Duration;

use litellm_redis_client::commands;
use litellm_redis_store::{
    FromValue, IntoValue,
    auth_cache::{AuthCache, AuthObject, CachedTeam, Settings},
};
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::{connect, remote};

const TTL: Duration = Duration::from_secs(60);

const CREATED_ENTRY: &str = concat!(
    r#"{"team_alias": null, "team_id": "created", "organization_id": null, "admins": [],"#,
    r#" "members": [], "members_with_roles": [], "team_member_permissions": [],"#,
    r#" "metadata": {}, "tpm_limit": null, "rpm_limit": null, "tpd_limit": null,"#,
    r#" "max_budget": null, "soft_budget": null, "budget_duration": null,"#,
    r#" "budget_limits": null, "models": [], "blocked": false, "router_settings": {},"#,
    r#" "access_group_ids": [], "default_team_member_models": [], "spend": 0.0,"#,
    r#" "max_parallel_requests": null, "budget_reset_at": null, "model_id": null,"#,
    r#" "model_spend": {}, "model_max_budget": {}, "policies": [],"#,
    r#" "allow_team_guardrail_config": false, "litellm_model_table": null,"#,
    r#" "object_permission": null, "object_permission_id": null,"#,
    r#" "updated_at": "2026-11-19T10:00:00Z", "created_at": "2026-11-19T10:00:00.123000Z","#,
    r#" "last_refreshed_at": null}"#,
);
const FULL_ENTRY: &str = concat!(
    r#"{"team_alias": "Full", "team_id": "full", "organization_id": "org","#,
    r#" "admins": ["admin"], "members": ["member"],"#,
    r#" "members_with_roles": [{"user_id": "admin", "user_email": null, "role": "admin"},"#,
    r#" {"user_id": null, "user_email": "m@example.com", "role": "user"}],"#,
    r#" "team_member_permissions": ["/key/generate"], "metadata": {"guardrails": ["pii"]},"#,
    r#" "tpm_limit": 1000, "rpm_limit": 10, "tpd_limit": 100000, "max_budget": 10.5,"#,
    r#" "soft_budget": 5.25, "budget_duration": "30d","#,
    r#" "budget_limits": [{"budget_duration": "1d", "max_budget": 1.0, "reset_at": null,"#,
    r#" "window_start": "2026-11-01T00:00:00Z"}, {"budget_duration": "30d","#,
    r#" "max_budget": 2.5, "reset_at": null}], "models": ["claude", "all-proxy-models"],"#,
    r#" "blocked": true, "router_settings": {"num_retries": 2},"#,
    r#" "access_group_ids": ["group"], "default_team_member_models": ["claude"],"#,
    r#" "spend": 1.5, "max_parallel_requests": 3, "budget_reset_at": "2026-12-01T00:00:00Z","#,
    r#" "model_id": 7, "model_spend": {"claude": 1.5},"#,
    r#" "model_max_budget": {"claude": {"budget_limit": 2.0}}, "policies": ["policy"],"#,
    r#" "allow_team_guardrail_config": true, "litellm_model_table": {"id": 7,"#,
    r#" "model_aliases": {"gpt": "claude"}, "created_by": "test", "updated_by": "test","#,
    r#" "team": null}, "object_permission": null, "object_permission_id": "permission","#,
    r#" "updated_at": "2026-11-19T10:00:00Z", "created_at": "2026-11-19T10:00:00.123000Z","#,
    r#" "last_refreshed_at": null}"#,
);

#[rstest]
#[case::as_created("created", CREATED_ENTRY)]
#[case::every_column_set("full", FULL_ENTRY)]
#[tokio::test]
async fn a_team_entry_python_wrote_is_read_typed_and_written_back_in_the_same_bytes(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[case] team_id: &str,
    #[case] entry: &str,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let cache = AuthCache::new(Some(remote(connect(&deployment))), Settings::default());
    let read = AuthObject::Team(team_id.to_owned());
    let rewritten = AuthObject::Team("rewritten".to_owned());
    raw.send(commands::set(&read.key(), entry, TTL))
        .await
        .unwrap();

    let held = cache.get(None, &read).await.unwrap().unwrap();
    let team = CachedTeam::from_value(&held).unwrap();
    cache
        .set(None, &rewritten, team.clone().into_value(), Some(TTL))
        .await
        .unwrap();

    assert_eq!(
        (
            team.team_id.as_str(),
            team.unread,
            raw.send(commands::get::<String>(&rewritten.key()))
                .await
                .unwrap()
                .as_deref(),
        ),
        (team_id, Vec::new(), Some(entry))
    );
}
