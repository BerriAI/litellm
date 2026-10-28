use std::borrow::Cow;
use std::collections::BTreeMap;

use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{
    Keys, RegistrationCandidate, RegistrationCandidates, Registrations, RowRef, registration_base,
    registration_target,
};
use rstest::rstest;
use serde_json::{Value, json};

struct Map(BTreeMap<String, Fields>);

impl Keys for Map {
    fn contains(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }

    fn row(&self, key: &str) -> Option<RowRef<'_>> {
        self.0.get(key).map(Cow::Borrowed)
    }

    fn by_normalized(&self, name: &str) -> Option<String> {
        let lower = name.to_lowercase();
        self.0
            .keys()
            .find(|key| key.to_lowercase() == lower)
            .cloned()
    }
}

fn fields(value: Value) -> Fields {
    let Value::Object(fields) = value else {
        unreachable!()
    };
    fields
}

fn map(rows: Value) -> Map {
    Map(fields(rows)
        .into_iter()
        .map(|(key, row)| (key, fields(row)))
        .collect())
}

fn lookup(name: &str, provider: &str) -> RegistrationCandidate {
    RegistrationCandidate {
        name: name.into(),
        provider: Some(provider.into()),
        requires_provider: false,
        names_the_row: true,
        seeded: false,
    }
}

#[rstest]
#[case::a_stripped_sibling_is_inherited_not_written(
    json!({"o3-pro": {"litellm_provider": "openai"}}),
    "openai/o3-pro",
    vec![lookup("openai/o3-pro", "openai"), lookup("o3-pro", "openai")],
    Some("o3-pro")
)]
#[case::a_case_variant_is_a_new_row(
    json!({"hosted_vllm/Acme-Model": {"litellm_provider": "hosted_vllm"}}),
    "hosted_vllm/acme-model",
    vec![lookup("hosted_vllm/acme-model", "hosted_vllm")],
    None
)]
#[case::another_providers_row_is_skipped(
    json!({"claude-sonnet-4-6": {"litellm_provider": "anthropic"}}),
    "azure/claude-sonnet-4-6",
    vec![lookup("azure/claude-sonnet-4-6", "azure"), lookup("claude-sonnet-4-6", "azure")],
    None
)]
fn the_caller_key_is_written_and_inheritance_is_picked_apart(
    #[case] rows: Value,
    #[case] key: &str,
    #[case] candidates: Vec<RegistrationCandidate>,
    #[case] inherit_from: Option<&str>,
) {
    let target = registration_target(&map(rows), key, &RegistrationCandidates(candidates));
    assert_eq!(target.write, key);
    assert_eq!(target.inherit_from.as_deref(), inherit_from);
}

#[rstest]
fn a_stripped_name_match_is_seeded_and_loses_its_provider() {
    let rows = map(
        json!({"claude-x": {"litellm_provider": "anthropic", "max_tokens": 10, "input_cost_per_token_batches": 1.0}}),
    );
    let candidate = RegistrationCandidate {
        name: "claude-x".into(),
        provider: Some("anthropic".into()),
        requires_provider: true,
        names_the_row: false,
        seeded: true,
    };
    let registration = fields(json!({"input_cost_per_token": 2.0}));
    let target = registration_target(
        &rows,
        "bedrock/claude-x",
        &RegistrationCandidates(vec![candidate]),
    );

    let base = registration_base(
        &target,
        rows.0.get("claude-x"),
        Some(&fields(json!({"supports_vision": true}))),
        &registration,
    );

    assert_eq!(
        Value::Object(base),
        json!({"supports_vision": true, "max_tokens": 10})
    );
}

#[rstest]
fn persisted_registrations_merge_and_replay_in_recording_order() {
    let mut registrations = Registrations::default();
    registrations.record("b", &fields(json!({"input_cost_per_token": 1.0})));
    registrations.record("a", &fields(json!({"input_cost_per_token": 2.0})));
    registrations.record("b", &fields(json!({"output_cost_per_token": 3.0})));

    let replayed: Vec<(&str, Value)> = registrations
        .replay()
        .map(|(key, fields)| (key, Value::Object(fields.clone())))
        .collect();
    assert_eq!(
        replayed,
        [
            (
                "b",
                json!({"input_cost_per_token": 1.0, "output_cost_per_token": 3.0})
            ),
            ("a", json!({"input_cost_per_token": 2.0}))
        ]
    );
}
