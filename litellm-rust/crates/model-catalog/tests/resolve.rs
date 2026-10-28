use std::borrow::Cow;
use std::collections::BTreeMap;

use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{CandidateSlot, Candidates, Keys, Resolution, RowRef, Via, resolve};
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

fn map(rows: Value) -> Map {
    let Value::Object(rows) = rows else {
        unreachable!()
    };
    Map(rows
        .into_iter()
        .map(|(key, row)| {
            let Value::Object(fields) = row else {
                unreachable!()
            };
            (key, fields)
        })
        .collect())
}

fn candidates(model: &str, combined: &str, split: &str) -> Candidates {
    Candidates {
        combined_model_name: Some(combined.to_owned()),
        model: model.to_owned(),
        region_free_combined_model_name: Some(combined.to_owned()),
        split_model: Some(split.to_owned()),
        combined_stripped_model_name: Some(combined.to_owned()),
        stripped_model_name: Some(split.to_owned()),
        provider_prefixed_model_name: Some(combined.to_owned()),
    }
}

fn found(key: &str, slot: CandidateSlot, case_insensitive: bool) -> Resolution {
    Resolution::Found {
        key: key.to_owned(),
        via: Via {
            slot,
            case_insensitive,
        },
    }
}

#[rstest]
#[case::provider_prefixed_row_first(
    json!({"groq/llama": {"litellm_provider": "groq"}, "llama": {"litellm_provider": "groq"}}),
    found("groq/llama", CandidateSlot::CombinedModelName, false)
)]
#[case::bare_row_when_no_prefixed_one(
    json!({"llama": {"litellm_provider": "groq"}}),
    found("llama", CandidateSlot::Model, false)
)]
#[case::a_mismatched_provider_is_skipped(
    json!({"groq/llama": {"litellm_provider": "openai"}, "llama": {"litellm_provider": "groq"}}),
    found("llama", CandidateSlot::Model, false)
)]
#[case::case_insensitive_through_the_index(
    json!({"Groq/Llama": {"litellm_provider": "groq"}}),
    found("Groq/Llama", CandidateSlot::CombinedModelName, true)
)]
fn the_first_matching_candidate_wins(#[case] rows: Value, #[case] expected: Resolution) {
    assert_eq!(
        resolve(
            &map(rows),
            &candidates("llama", "groq/llama", "llama"),
            Some("groq")
        ),
        expected
    );
}

#[rstest]
fn a_miss_lists_every_candidate_tried() {
    let Resolution::Missing { tried } = resolve(
        &map(json!({})),
        &candidates("x", "acme/x", "x"),
        Some("acme"),
    ) else {
        panic!("expected a miss");
    };
    assert_eq!(
        tried,
        ["acme/x", "x", "acme/x", "x", "acme/x", "x", "acme/x"]
    );
}
