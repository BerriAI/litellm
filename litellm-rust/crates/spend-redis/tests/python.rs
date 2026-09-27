use litellm_spend::{Cost, CounterKey, EntityKey, Totals};
use litellm_spend_redis::{
    BatchCodec, CounterNaming, PythonCounterNaming, PythonFormatError, PythonTotalsCodec,
};
use rstest::rstest;

fn team_member(team_id: &str, user_id: &str) -> EntityKey {
    EntityKey::TeamMember {
        team_id: team_id.to_owned(),
        user_id: user_id.to_owned(),
    }
}

fn organization_member(organization_id: &str, user_id: &str) -> EntityKey {
    EntityKey::OrganizationMember {
        organization_id: organization_id.to_owned(),
        user_id: user_id.to_owned(),
    }
}

#[rstest]
#[case::key(EntityKey::Key("hashed".to_owned()), None, Some("spend:key:hashed"))]
#[case::key_window(EntityKey::Key("hashed".to_owned()), Some("1d"), Some("spend:key:hashed:window:1d"))]
#[case::team_window(EntityKey::Team("t".to_owned()), Some("30d"), Some("spend:team:t:window:30d"))]
#[case::team_member_puts_the_user_first(team_member("t", "u"), None, Some("spend:team_member:u:t"))]
#[case::organization(EntityKey::Organization("o".to_owned()), None, Some("spend:org:o"))]
#[case::end_user(EntityKey::EndUser("e".to_owned()), None, Some("spend:end_user:e"))]
#[case::agent_has_no_counter(EntityKey::Agent("a".to_owned()), None, None)]
#[case::organization_member_has_no_counter(organization_member("o", "u"), None, None)]
fn python_counter_names(
    #[case] entity: EntityKey,
    #[case] window: Option<&str>,
    #[case] expected: Option<&str>,
) {
    let key = CounterKey {
        entity,
        window: window.map(str::to_owned),
    };

    assert_eq!(PythonCounterNaming.counter(&key).as_deref(), expected);
}

#[test]
fn a_blob_written_by_python_decodes_into_typed_entities() {
    let blob = r#"{
        "user_list_transactions": {"u": 0.25},
        "end_user_list_transactions": null,
        "key_list_transactions": {"hashed": 0.5},
        "team_list_transactions": {},
        "team_member_list_transactions": {"team_id::t::user_id::u": 0.75},
        "org_member_list_transactions": {"organization_id::o%3A%3A1::user_id::u%2F2": 1.0}
    }"#;

    let decoded = PythonTotalsCodec.decode(blob).unwrap();

    assert_eq!(
        decoded,
        Totals::from_entries([
            (EntityKey::User("u".to_owned()), Cost(0.25)),
            (EntityKey::Key("hashed".to_owned()), Cost(0.5)),
            (team_member("t", "u"), Cost(0.75)),
            (organization_member("o::1", "u/2"), Cost(1.0)),
        ])
    );
}

#[test]
fn every_entity_kind_survives_an_encode_and_decode() {
    let batch = Totals::from_entries(
        [
            EntityKey::User("same".to_owned()),
            EntityKey::EndUser("same".to_owned()),
            EntityKey::Key("same".to_owned()),
            EntityKey::Team("same".to_owned()),
            team_member("t", "u"),
            EntityKey::Organization("same".to_owned()),
            organization_member("o::1", "u 2"),
            EntityKey::Project("same".to_owned()),
            EntityKey::Tag("same".to_owned()),
            EntityKey::ModelAccessGroup("same".to_owned()),
            EntityKey::Agent("same".to_owned()),
        ]
        .into_iter()
        .zip(1..)
        .map(|(entity, i)| (entity, Cost(0.25 * f64::from(i)))),
    );

    let blob = PythonTotalsCodec.encode(&batch).unwrap();

    assert_eq!(PythonTotalsCodec.decode(&blob).unwrap(), batch);
}

#[rstest]
#[case::in_the_team(team_member("a::b", "u"))]
#[case::in_the_user(team_member("t", "a::b"))]
fn a_team_member_id_holding_the_separator_is_refused_not_misfiled(#[case] entity: EntityKey) {
    let batch = Totals::from_entries([(entity, Cost(0.25))]);

    assert!(matches!(
        PythonTotalsCodec.encode(&batch),
        Err(PythonFormatError::SeparatorInId(_))
    ));
}

#[rstest]
#[case::extra_segment(r#"{"team_member_list_transactions": {"team_id::a::b::user_id::u": 0.25}}"#)]
#[case::wrong_label(
    r#"{"team_member_list_transactions": {"organization_id::t::user_id::u": 0.25}}"#
)]
#[case::missing_user(r#"{"org_member_list_transactions": {"organization_id::o": 0.25}}"#)]
fn a_member_key_that_does_not_name_one_group_and_one_user_is_an_error(#[case] blob: &str) {
    assert!(matches!(
        PythonTotalsCodec.decode(blob),
        Err(PythonFormatError::MemberKey(_))
    ));
}
