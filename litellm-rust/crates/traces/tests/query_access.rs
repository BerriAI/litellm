use litellm_traces::{InvalidScope, QueryScope};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::admin(json!({"kind": "admin"}), true)]
#[case::team(json!({"kind": "team", "team_id": "team"}), true)]
#[case::empty_team(json!({"kind": "team", "team_id": ""}), false)]
#[case::user_logs(json!({"kind": "logs", "user_id": "user", "team_ids": []}), true)]
#[case::permitted_teams(json!({"kind": "logs", "user_id": "", "team_ids": ["team"]}), true)]
#[case::anonymous_logs(json!({"kind": "logs", "user_id": "", "team_ids": []}), false)]
#[case::empty_permitted_team(json!({"kind": "logs", "user_id": "user", "team_ids": [""]}), false)]
fn scope_validation_preserves_authorization_and_wire_shape(
    #[case] wire: Value,
    #[case] valid: bool,
) {
    let scope: QueryScope = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(&scope).unwrap(), wire);
    match (scope.validate(), valid) {
        (Ok(()), true) | (Err(InvalidScope), false) => (),
        (result, _) => panic!("unexpected validation: {result:?}"),
    }
}

#[rstest]
#[case::unknown_kind(json!({"kind": "all"}))]
#[case::unknown_field(json!({"kind": "team", "team_id": "team", "extra": true}))]
#[case::key_scope(json!({"kind": "key", "team_id": "team", "api_key_hash": "key"}))]
#[case::key_grant(json!({"kind": "logs", "user_id": "user", "team_ids": [], "api_key_hash": "key"}))]
fn scope_rejects_invalid_wire_shape(#[case] wire: Value) {
    assert!(serde_json::from_value::<QueryScope>(wire).is_err());
}

#[rstest]
fn admin_preserves_existing_extra_field_handling() {
    let scope: QueryScope =
        serde_json::from_value(json!({"kind": "admin", "team_id": "ignored"})).unwrap();
    assert!(matches!(scope, QueryScope::Admin));
    assert!(scope.validate().is_ok());
    assert_eq!(
        serde_json::to_value(scope).unwrap(),
        json!({"kind": "admin"})
    );
}
