use litellm_python_compat::{Value, repr::to_str};

use super::types::Eligible;

/// A claim settled against this router's memory alone: who holds the key, and what memory keeps
/// for it. A holder nobody may replace is answered without being written again.
#[derive(Debug, PartialEq)]
pub(super) struct LocalClaim {
    pub(super) winner: Value,
    pub(super) stored: Option<Value>,
}

/// The claim script's rule for a router without Redis, or one whose Redis claim failed: the first
/// writer keeps the key unless `eligible` no longer lists it.
pub(super) fn local(existing: Option<Value>, pin: &Value, eligible: &Eligible) -> LocalClaim {
    match (existing, &eligible.values) {
        (Some(existing), None) => LocalClaim {
            stored: legacy_matches(&existing, pin).then(|| pin.clone()),
            winner: existing,
        },
        (Some(existing), Some(values))
            if still_eligible(&existing, values, eligible.preserve_legacy_model_ids) =>
        {
            LocalClaim {
                stored: Some(existing.clone()),
                winner: existing,
            }
        }
        _ => LocalClaim {
            stored: Some(pin.clone()),
            winner: pin.clone(),
        },
    }
}

fn still_eligible(existing: &Value, eligible: &[Value], preserve_legacy_model_ids: bool) -> bool {
    eligible.iter().any(|value| same_pin(existing, value))
        || (preserve_legacy_model_ids
            && matches!(existing, Value::Str(_))
            && eligible.iter().any(|value| {
                field(value, &Value::Str("model_id".to_owned())).is_some()
                    && legacy_matches(existing, value)
            }))
}

/// A stored mapping carries every field of `pin` as its text, or the bare string an older writer
/// left is the one value `pin` holds.
fn legacy_matches(stored: &Value, pin: &Value) -> bool {
    let Value::Dict(fields) = pin else {
        return false;
    };
    match stored {
        Value::Dict(_) => fields.iter().all(|(key, value)| {
            field(stored, key)
                .is_some_and(|held| *held != Value::None && Value::Str(to_str(held)) == *value)
        }),
        Value::Str(_) => matches!(fields.as_slice(), [(_, value)] if value == stored),
        _ => false,
    }
}

/// Python's `==` between two pins: mappings compare by their fields, in whatever order.
fn same_pin(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Dict(fields), Value::Dict(other)) => {
            fields.len() == other.len()
                && fields
                    .iter()
                    .all(|(key, value)| field(right, key) == Some(value))
        }
        _ => left == right,
    }
}

fn field<'a>(mapping: &'a Value, key: &Value) -> Option<&'a Value> {
    let Value::Dict(fields) = mapping else {
        return None;
    };
    fields
        .iter()
        .find(|(held, _)| held == key)
        .map(|(_, value)| value)
}

#[cfg(test)]
mod tests {
    use litellm_python_compat::json;
    use rstest::rstest;

    use super::*;

    fn value(text: &str) -> Value {
        json::loads(text).unwrap()
    }

    fn eligible(values: &[&str], preserve_legacy_model_ids: bool) -> Eligible {
        Eligible {
            values: Some(values.iter().copied().map(value).collect()),
            preserve_legacy_model_ids,
        }
    }

    #[rstest]
    #[case::nobody_holds_the_key(
        None,
        r#"{"model_id": "b"}"#,
        r#"{"model_id": "b"}"#,
        Some(r#"{"model_id": "b"}"#)
    )]
    #[case::the_first_writer_keeps_it(
        Some(r#"{"model_id": "a"}"#),
        r#"{"model_id": "b"}"#,
        r#"{"model_id": "a"}"#,
        None
    )]
    #[case::the_holder_claiming_again_is_refreshed(
        Some(r#"{"model_id": "a"}"#),
        r#"{"model_id": "a"}"#,
        r#"{"model_id": "a"}"#,
        Some(r#"{"model_id": "a"}"#)
    )]
    #[case::a_holder_with_a_number_matches_its_text(
        Some(r#"{"model_id": 7}"#),
        r#"{"model_id": "7"}"#,
        r#"{"model_id": 7}"#,
        Some(r#"{"model_id": "7"}"#)
    )]
    #[case::a_holder_with_a_null_field_is_not_refreshed(
        Some(r#"{"model_id": null}"#),
        r#"{"model_id": "None"}"#,
        r#"{"model_id": null}"#,
        None
    )]
    #[case::a_bare_id_an_older_writer_left_is_refreshed_as_the_pin(
        Some(r#""a""#),
        r#"{"model_id": "a"}"#,
        r#""a""#,
        Some(r#"{"model_id": "a"}"#)
    )]
    #[case::a_bare_id_does_not_match_a_pin_of_two_fields(
        Some(r#""a""#),
        r#"{"model_id": "a", "tier": "a"}"#,
        r#""a""#,
        None
    )]
    fn a_claim_naming_no_eligible_values_keeps_the_first_writer(
        #[case] existing: Option<&str>,
        #[case] pin: &str,
        #[case] winner: &str,
        #[case] stored: Option<&str>,
    ) {
        let settled = local(existing.map(value), &value(pin), &Eligible::default());

        assert_eq!(
            settled,
            LocalClaim {
                winner: value(winner),
                stored: stored.map(value),
            }
        );
    }

    #[rstest]
    #[case::nobody_holds_the_key(None, &[r#"{"model_id": "b"}"#], false, r#"{"model_id": "b"}"#)]
    #[case::an_eligible_holder_keeps_it(Some(r#"{"model_id": "a"}"#), &[r#"{"model_id": "a"}"#, r#"{"model_id": "b"}"#], false, r#"{"model_id": "a"}"#)]
    #[case::fields_match_in_any_order(Some(r#"{"model": "m", "tier": "t"}"#), &[r#"{"tier": "t", "model": "m"}"#], false, r#"{"model": "m", "tier": "t"}"#)]
    #[case::a_holder_no_longer_eligible_is_replaced(Some(r#"{"model_id": "z"}"#), &[r#"{"model_id": "b"}"#], false, r#"{"model_id": "b"}"#)]
    #[case::a_holder_with_an_extra_field_is_replaced(Some(r#"{"model_id": "b", "tier": "t"}"#), &[r#"{"model_id": "b"}"#], false, r#"{"model_id": "b"}"#)]
    #[case::a_bare_id_is_kept_when_legacy_ids_are_preserved(Some(r#""a""#), &[r#"{"model_id": "a"}"#], true, r#""a""#)]
    #[case::a_bare_id_is_replaced_when_they_are_not(Some(r#""a""#), &[r#"{"model_id": "a"}"#], false, r#"{"model_id": "b"}"#)]
    #[case::a_bare_id_only_matches_a_model_id(Some(r#""a""#), &[r#"{"tier": "a"}"#], true, r#"{"model_id": "b"}"#)]
    fn a_claim_naming_eligible_values_replaces_only_a_holder_they_do_not_list(
        #[case] existing: Option<&str>,
        #[case] listed: &[&str],
        #[case] preserve_legacy_model_ids: bool,
        #[case] winner: &str,
    ) {
        let settled = local(
            existing.map(value),
            &value(r#"{"model_id": "b"}"#),
            &eligible(listed, preserve_legacy_model_ids),
        );

        assert_eq!(
            settled,
            LocalClaim {
                winner: value(winner),
                stored: Some(value(winner)),
            }
        );
    }
}
