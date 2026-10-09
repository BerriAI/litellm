use litellm_traces::{InvalidScope, QueryScope};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[case::all(json!({"kind": "all"}), true)]
#[case::own_user(json!({"kind": "owned", "user_id": "user", "team_ids": []}), true)]
#[case::permitted_teams(json!({"kind": "owned", "user_id": "", "team_ids": ["team"]}), true)]
#[case::own_user_and_permitted_teams(json!({"kind": "owned", "user_id": "user", "team_ids": ["team"]}), true)]
#[case::no_identity(json!({"kind": "owned", "user_id": "", "team_ids": []}), false)]
#[case::empty_permitted_team(json!({"kind": "owned", "user_id": "user", "team_ids": [""]}), false)]
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
#[case::unknown_kind(json!({"kind": "unknown"}))]
#[case::unknown_field(json!({"kind": "owned", "user_id": "user", "team_ids": [], "extra": true}))]
#[case::legacy_admin(json!({"kind": "admin"}))]
#[case::legacy_logs(json!({"kind": "logs", "user_id": "user", "team_ids": []}))]
#[case::legacy_team(json!({"kind": "team", "team_id": "team"}))]
#[case::key_scope(json!({"kind": "key", "team_id": "team", "api_key_hash": "key"}))]
#[case::key_grant(json!({"kind": "owned", "user_id": "user", "team_ids": [], "api_key_hash": "key"}))]
fn scope_rejects_invalid_wire_shape(#[case] wire: Value) {
    assert!(serde_json::from_value::<QueryScope>(wire).is_err());
}

#[rstest]
fn all_preserves_existing_extra_field_handling() {
    let scope: QueryScope =
        serde_json::from_value(json!({"kind": "all", "team_id": "ignored"})).unwrap();
    assert!(matches!(scope, QueryScope::All));
    assert!(scope.validate().is_ok());
    assert_eq!(serde_json::to_value(scope).unwrap(), json!({"kind": "all"}));
}
