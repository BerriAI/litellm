//! Tag entries and the tag names round-trip through Redis in the bytes the Python proxy writes.

mod support;

use litellm_redis_store::auth_cache::{AuthObject, TagEntry, TagNames};
use litellm_redis_testing::Topology;
use rstest::rstest;

use support::auth_cache::round_trip;

const PLAIN_ENTRY: &str = concat!(
    r#"{"tag_name": "plain", "description": null, "models": [], "model_info": null,"#,
    r#" "spend": 0.0, "budget_id": null, "litellm_budget_table": null,"#,
    r#" "created_at": "2026-11-19T10:00:00Z", "created_by": null, "updated_at": "2026-11-19T10:00:00Z"}"#,
);
const LIMITED_ENTRY: &str = concat!(
    r#"{"tag_name": "limited", "description": "Team tag", "models": ["claude"],"#,
    r#" "model_info": {"deployment-1": "claude"}, "spend": 1.5, "budget_id": "bare","#,
    r#" "litellm_budget_table": {"budget_id": "bare", "soft_budget": null, "max_budget": 10.5,"#,
    r#" "max_parallel_requests": null, "tpm_limit": 1000, "rpm_limit": 3, "tpd_limit": null,"#,
    r#" "model_max_budget": null, "model_tpm_limit": null, "model_rpm_limit": null, "budget_duration": null,"#,
    r#" "allowed_models": [], "temp_budget_increase": null, "temp_budget_expiry": null},"#,
    r#" "created_at": "2026-11-19T10:00:00.123000Z", "created_by": "creator","#,
    r#" "updated_at": "2026-11-19T10:00:00Z"}"#,
);

#[rstest]
#[case::no_budget("plain", PLAIN_ENTRY)]
#[case::linked_budget("limited", LIMITED_ENTRY)]
#[tokio::test]
async fn a_tag_entry_python_wrote_is_read_typed_and_written_back_in_the_same_bytes(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[case] tag_name: &str,
    #[case] entry: &str,
) {
    let (tag, written) = round_trip::<TagEntry>(
        topology,
        AuthObject::Tag(tag_name.to_owned()),
        AuthObject::Tag("rewritten".to_owned()),
        entry,
    )
    .await;

    assert_eq!(
        (tag.tag_name.as_str(), tag.unread, written.as_deref()),
        (tag_name, Vec::new(), Some(entry))
    );
}

#[rstest]
#[tokio::test]
async fn null_models_and_spend_are_normalized_when_a_tag_entry_is_read(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let legacy = PLAIN_ENTRY
        .replace(r#""models": []"#, r#""models": null"#)
        .replace(r#""spend": 0.0"#, r#""spend": null"#);
    let (_, written) = round_trip::<TagEntry>(
        topology,
        AuthObject::Tag("plain".to_owned()),
        AuthObject::Tag("rewritten".to_owned()),
        &legacy,
    )
    .await;

    assert_eq!(written.as_deref(), Some(PLAIN_ENTRY));
}

#[rstest]
#[case::every_name(
    r#"["limited", "plain"]"#,
    TagNames::Listed(vec!["limited".to_owned(), "plain".to_owned()])
)]
#[case::more_names_than_python_holds(r#""__tag_registry_overflow__""#, TagNames::Unusable)]
#[tokio::test]
async fn the_tag_names_python_wrote_are_read_typed_and_written_back_in_the_same_bytes(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[case] entry: &str,
    #[case] names: TagNames,
) {
    let (read, written) = round_trip::<TagNames>(
        topology,
        AuthObject::TagRegistry,
        AuthObject::Tag("rewritten".to_owned()),
        entry,
    )
    .await;

    assert_eq!((read, written.as_deref()), (names, Some(entry)));
}
