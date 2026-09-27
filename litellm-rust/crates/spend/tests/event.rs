use litellm_spend::{
    Attribution, Cost, CounterKey, DailyEntity, EntityKey, Outcome, SpendEvent, Usage, charges,
};
use rstest::{fixture, rstest};
use time::macros::{date, datetime};

#[fixture]
fn event() -> SpendEvent {
    SpendEvent {
        cost: 0.25,
        attribution: Attribution {
            key: Some("hashed".to_owned()),
            user: Some("user-a".to_owned()),
            team: Some("team-a".to_owned()),
            organization: Some("org-a".to_owned()),
            tags: vec!["prod".to_owned(), "batch".to_owned()],
            ..Attribution::default()
        },
        started_at: datetime!(2026-09-26 23:30 -02:00),
        model: Some("model-a".to_owned()),
        model_group: Some("group-a".to_owned()),
        custom_llm_provider: Some("provider-a".to_owned()),
        endpoint: Some("/v1/messages".to_owned()),
        mcp_namespaced_tool_name: None,
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            ..Usage::default()
        },
        outcome: Outcome::Success,
        response_time_ms: Some(1_200),
    }
}

fn team_member() -> EntityKey {
    EntityKey::TeamMember {
        team_id: "team-a".to_owned(),
        user_id: "user-a".to_owned(),
    }
}

fn organization_member() -> EntityKey {
    EntityKey::OrganizationMember {
        organization_id: "org-a".to_owned(),
        user_id: "user-a".to_owned(),
    }
}

#[rstest]
fn every_attributed_entity_is_charged_the_full_cost_once(event: SpendEvent) {
    let charged: Vec<_> = charges(&event)
        .totals
        .iter()
        .map(|(entity, cost)| (entity.clone(), *cost))
        .collect();

    let mut expected = vec![
        (EntityKey::Key("hashed".to_owned()), Cost(0.25)),
        (EntityKey::User("user-a".to_owned()), Cost(0.25)),
        (EntityKey::Team("team-a".to_owned()), Cost(0.25)),
        (team_member(), Cost(0.25)),
        (EntityKey::Organization("org-a".to_owned()), Cost(0.25)),
        (organization_member(), Cost(0.25)),
        (EntityKey::Tag("prod".to_owned()), Cost(0.25)),
        (EntityKey::Tag("batch".to_owned()), Cost(0.25)),
    ];
    expected.sort_by(|(left, _), (right, _)| left.cmp(right));
    assert_eq!(charged, expected);
}

#[rstest]
#[case::team_without_user(Attribution { team: Some("team-a".to_owned()), ..Attribution::default() })]
#[case::user_without_team(Attribution { user: Some("user-a".to_owned()), ..Attribution::default() })]
fn a_membership_is_charged_only_when_both_sides_are_known(
    event: SpendEvent,
    #[case] attribution: Attribution,
) {
    let charged = charges(&SpendEvent {
        attribution,
        ..event
    });

    assert_eq!(charged.totals.len(), 1);
    assert_eq!(charged.totals.get(&team_member()), None);
}

#[rstest]
#[case::budgeted(team_member(), true)]
#[case::unbudgeted(organization_member(), false)]
fn counters_exist_only_for_entities_that_carry_a_budget(
    event: SpendEvent,
    #[case] entity: EntityKey,
    #[case] has_counter: bool,
) {
    let counters = charges(&event).counters;

    assert_eq!(
        counters.contains(&(CounterKey::lifetime(entity), Cost(0.25))),
        has_counter
    );
}

#[rstest]
fn daily_rollups_use_the_utc_day_and_one_row_per_entity_and_tag(event: SpendEvent) {
    let daily = charges(&event).daily;

    let mut entities: Vec<_> = daily.iter().map(|(key, _)| key.entity.clone()).collect();
    entities.sort();
    let mut expected = vec![
        DailyEntity::User("user-a".to_owned()),
        DailyEntity::Team("team-a".to_owned()),
        DailyEntity::Organization("org-a".to_owned()),
        DailyEntity::Tag("prod".to_owned()),
        DailyEntity::Tag("batch".to_owned()),
    ];
    expected.sort();
    assert_eq!(entities, expected);
    assert!(
        daily
            .iter()
            .all(|(key, _)| key.date == date!(2026 - 09 - 27))
    );
}

#[rstest]
#[case::success(Outcome::Success, Some(900), (1, 0, 900, 1))]
#[case::failure(Outcome::Failure, Some(900), (0, 1, 900, 1))]
#[case::untimed(Outcome::Success, None, (1, 0, 0, 0))]
fn a_daily_tally_counts_the_outcome_and_the_response_time(
    event: SpendEvent,
    #[case] outcome: Outcome,
    #[case] response_time_ms: Option<u64>,
    #[case] expected: (u64, u64, u64, u64),
) {
    let daily = charges(&SpendEvent {
        outcome,
        response_time_ms,
        ..event
    })
    .daily;

    let (_, tally) = daily.iter().next().unwrap();
    assert_eq!(
        (
            tally.successful_requests,
            tally.failed_requests,
            tally.total_response_time_ms,
            tally.timed_requests
        ),
        expected
    );
    assert_eq!(
        (
            tally.api_requests,
            tally.prompt_tokens,
            tally.completion_tokens
        ),
        (1, 10, 20)
    );
}
