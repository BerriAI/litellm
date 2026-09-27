use litellm_spend::{Cost, DailyEntity, DailyKey, DailyRollups, DailyTally, EntityKey, Totals};
use rstest::rstest;
use time::macros::date;

fn key(id: &str) -> EntityKey {
    EntityKey::Key(id.to_owned())
}

fn delta(entity: EntityKey, cost: f64) -> (EntityKey, Cost) {
    (entity, Cost(cost))
}

#[rstest]
#[case::same_entity_sums(vec![delta(key("a"), 0.25), delta(key("a"), 0.5)], key("a"), Some(0.75))]
#[case::other_entities_do_not_leak(vec![delta(key("a"), 0.25), delta(key("b"), 0.5)], key("a"), Some(0.25))]
#[case::key_and_team_with_one_id_stay_apart(
    vec![delta(key("x"), 0.25), delta(EntityKey::Team("x".to_owned()), 0.5)],
    key("x"),
    Some(0.25)
)]
#[case::absent_entity(vec![delta(key("a"), 0.25)], key("z"), None)]
fn from_entries_sums_per_entity(
    #[case] deltas: Vec<(EntityKey, Cost)>,
    #[case] entity: EntityKey,
    #[case] expected: Option<f64>,
) {
    assert_eq!(
        Totals::from_entries(deltas).get(&entity).map(|cost| cost.0),
        expected
    );
}

#[test]
fn merge_sums_overlapping_entities_and_keeps_the_rest() {
    let left = Totals::from_entries([delta(key("a"), 0.25), delta(key("b"), 1.0)]);
    let right = Totals::from_entries([delta(key("a"), 0.5), delta(key("c"), 2.0)]);

    let merged = left.merge(right);

    assert_eq!(
        (
            merged.get(&key("a")),
            merged.get(&key("b")),
            merged.get(&key("c")),
            merged.len()
        ),
        (Some(&Cost(0.75)), Some(&Cost(1.0)), Some(&Cost(2.0)), 3)
    );
}

#[rstest]
#[case::fewer_than_max(2, 5, 2, 0)]
#[case::exactly_max(3, 3, 3, 0)]
#[case::more_than_max(5, 2, 2, 3)]
#[case::zero_max(2, 0, 0, 2)]
fn split_at_caps_the_head_and_loses_nothing(
    #[case] entities: usize,
    #[case] max: usize,
    #[case] head_len: usize,
    #[case] tail_len: usize,
) {
    let batch = Totals::from_entries((0..entities).map(|i| delta(key(&i.to_string()), 0.25)));

    let (head, tail) = batch.clone().split_at(max);

    assert_eq!((head.len(), tail.len()), (head_len, tail_len));
    assert_eq!(head.merge(tail), batch);
}

#[test]
fn daily_rollups_sum_every_counter_and_keep_the_first_model_group() {
    let key = DailyKey {
        date: date!(2026 - 09 - 26),
        entity: DailyEntity::Team("team-a".to_owned()),
        api_key: "hashed".to_owned(),
        model: Some("model-a".to_owned()),
        custom_llm_provider: None,
        mcp_namespaced_tool_name: None,
        endpoint: None,
    };
    let first = DailyTally {
        model_group: Some("group-a".to_owned()),
        spend: 0.25,
        prompt_tokens: 10,
        api_requests: 1,
        successful_requests: 1,
        ..DailyTally::default()
    };
    let second = DailyTally {
        model_group: Some("group-b".to_owned()),
        spend: 0.5,
        prompt_tokens: 5,
        completion_tokens: 7,
        api_requests: 1,
        failed_requests: 1,
        ..DailyTally::default()
    };

    let rollups = DailyRollups::from_entries([(key.clone(), first), (key.clone(), second)]);

    assert_eq!(
        rollups.get(&key),
        Some(&DailyTally {
            model_group: Some("group-a".to_owned()),
            spend: 0.75,
            prompt_tokens: 15,
            completion_tokens: 7,
            api_requests: 2,
            successful_requests: 1,
            failed_requests: 1,
            ..DailyTally::default()
        })
    );
}
