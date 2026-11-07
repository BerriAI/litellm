use std::collections::BTreeSet;

use indexmap::IndexMap;
use serde_json::{Number, Value};

use crate::wire::Fields;

/// Where one `register` call writes and which row it inherits from
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegistrationTarget {
    /// Always the caller's key, never a case variant of it
    pub write: String,
    /// The first candidate that is an exact key whose row passes `provider_matches`
    pub inherit_from: Option<String>,
}

/// Python's inheritance candidates in main's order, `get_model_info`'s then `_resolve_builtin_model_cost_entry`'s
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegistrationCandidates(pub Vec<String>);

/// `register_model(persist_across_reloads=True)` calls, replayed above every installed document
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Registrations {
    /// Each registered key's merged fields, ordered by its latest registration
    entries: IndexMap<String, Fields>,
    /// The registered keys whose write landed on each row key, in landing order
    landed_on: IndexMap<String, Vec<String>>,
}

impl Registrations {
    /// Merges `fields` into the key's earlier registration and makes it the latest one
    pub fn register(&mut self, key: &str, fields: &Fields) {
        let earlier = self.entries.shift_remove(key).unwrap_or_default();
        self.entries
            .insert(key.to_owned(), update_fields(earlier, fields));
    }

    /// Records that the registration under `registered` was written to the row under `written`
    pub fn landed(&mut self, written: &str, registered: &str) {
        let names = self.landed_on.entry(written.to_owned()).or_default();
        if !names.iter().any(|name| name == registered) {
            names.push(registered.to_owned());
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every registration, oldest first
    pub fn entries(&self) -> impl Iterator<Item = (&str, &Fields)> {
        self.entries
            .iter()
            .map(|(key, fields)| (key.as_str(), fields))
    }

    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Row keys a registration was written to, then registered keys never written anywhere
    pub(crate) fn targets(&self) -> impl Iterator<Item = &str> {
        let landed = self.landed_on.keys().map(String::as_str);
        let unlanded = self
            .entries
            .keys()
            .filter(|key| !self.landed_on.values().flatten().any(|name| name == *key))
            .map(String::as_str);
        landed.chain(unlanded)
    }

    /// The registrations that apply to the row under any of `names`, oldest first, as `_RegistrationsOverCatalog` reads them
    pub(crate) fn over<'a>(&'a self, names: &[&str]) -> Vec<&'a Fields> {
        let mut applied: Vec<(usize, &Fields)> = names
            .iter()
            .flat_map(|name| {
                self.landed_on
                    .get(*name)
                    .into_iter()
                    .flatten()
                    .map(String::as_str)
                    .chain(std::iter::once(*name))
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter_map(|key| self.entries.get_full(key))
            .map(|(order, _, fields)| (order, fields))
            .collect();
        applied.sort_unstable_by_key(|(order, _)| *order);
        applied.into_iter().map(|(_, fields)| fields).collect()
    }
}

/// `litellm.utils.update_dictionary`: a `None` value is skipped, a string that reads as a number becomes that
/// number, and a nested object is merged one level deep into the object it replaces
pub fn update_fields(existing: Fields, update: &Fields) -> Fields {
    update.iter().filter(|(_, value)| !value.is_null()).fold(
        existing,
        |mut merged, (name, value)| {
            let next = match value {
                Value::String(text) => stringified_number(text),
                Value::Object(nested) => match merged.get(name) {
                    Some(Value::Object(earlier)) => Value::Object(
                        earlier
                            .iter()
                            .chain(nested)
                            .map(|(key, value)| (key.clone(), value.clone()))
                            .collect(),
                    ),
                    _ => value.clone(),
                },
                value => value.clone(),
            };
            merged.insert(name.clone(), next);
            merged
        },
    )
}

/// `litellm.utils._convert_stringified_numbers`
fn stringified_number(text: &str) -> Value {
    let floating = text.to_lowercase().contains('e') || text.contains('.');
    let number = if floating {
        litellm_python_compat::number::float(text).and_then(Number::from_f64)
    } else {
        litellm_python_compat::number::int(text).and_then(|value| {
            i64::try_from(&value)
                .map(Number::from)
                .or_else(|_| u64::try_from(&value).map(Number::from))
                .ok()
        })
    };
    number.map_or_else(|| Value::String(text.to_owned()), Value::Number)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn fields(value: Value) -> Fields {
        let Value::Object(fields) = value else {
            panic!("an object")
        };
        fields
    }

    #[rstest]
    #[case::skips_none(json!({"a": 1}), json!({"a": null, "b": 2}), json!({"a": 1, "b": 2}))]
    #[case::scientific_string(json!({}), json!({"a": "3e-07"}), json!({"a": 3e-07}))]
    #[case::decimal_string(json!({}), json!({"a": "0.5"}), json!({"a": 0.5}))]
    #[case::integer_string(json!({}), json!({"a": "8192"}), json!({"a": 8192}))]
    #[case::underscored_integer(json!({}), json!({"a": "1_000"}), json!({"a": 1000}))]
    #[case::word_with_e_stays(json!({}), json!({"a": "none"}), json!({"a": "none"}))]
    #[case::word_stays(json!({}), json!({"a": "openai"}), json!({"a": "openai"}))]
    #[case::nested_merges_one_level(
        json!({"n": {"x": 1, "y": {"deep": 1}}}),
        json!({"n": {"y": {"other": 2}, "z": 3}}),
        json!({"n": {"x": 1, "y": {"other": 2}, "z": 3}})
    )]
    #[case::nested_replaces_scalar(json!({"n": 1}), json!({"n": {"x": 1}}), json!({"n": {"x": 1}}))]
    #[case::existing_key_keeps_position(
        json!({"a": 1, "b": 2}),
        json!({"c": 3, "a": 4}),
        json!({"a": 4, "b": 2, "c": 3})
    )]
    fn update_fields_follows_update_dictionary(
        #[case] existing: Value,
        #[case] update: Value,
        #[case] expected: Value,
    ) {
        let merged = update_fields(fields(existing), &fields(update));
        assert_eq!(Value::Object(merged.clone()), expected);
        assert_eq!(
            merged.keys().collect::<Vec<_>>(),
            fields(expected).keys().collect::<Vec<_>>()
        );
    }

    #[rstest]
    fn reregistering_a_key_merges_and_makes_it_the_latest() {
        let mut registrations = Registrations::default();
        registrations.register("a", &fields(json!({"x": 1, "y": 1})));
        registrations.register("b", &fields(json!({"x": 2})));
        registrations.register("a", &fields(json!({"y": 3})));
        let entries: Vec<_> = registrations
            .entries()
            .map(|(key, fields)| (key, Value::Object(fields.clone())))
            .collect();
        assert_eq!(
            entries,
            vec![("b", json!({"x": 2})), ("a", json!({"x": 1, "y": 3}))]
        );
    }

    #[rstest]
    fn over_applies_names_landed_on_the_row_in_registration_order() {
        let mut registrations = Registrations::default();
        registrations.register("openai/o3-pro", &fields(json!({"x": 1})));
        registrations.register("o3-pro", &fields(json!({"x": 2})));
        registrations.register("other", &fields(json!({"x": 3})));
        registrations.landed("o3-pro", "openai/o3-pro");
        registrations.landed("o3-pro", "o3-pro");
        registrations.register("openai/o3-pro", &fields(json!({"y": 4})));
        let applied: Vec<_> = registrations
            .over(&["o3-pro"])
            .into_iter()
            .map(|fields| Value::Object(fields.clone()))
            .collect();
        assert_eq!(applied, vec![json!({"x": 2}), json!({"x": 1, "y": 4})]);
    }

    #[rstest]
    fn targets_list_landed_rows_then_unlanded_registrations() {
        let mut registrations = Registrations::default();
        registrations.register("openai/o3-pro", &fields(json!({"x": 1})));
        registrations.register("never-written", &fields(json!({"x": 2})));
        registrations.landed("o3-pro", "openai/o3-pro");
        assert_eq!(
            registrations.targets().collect::<Vec<_>>(),
            vec!["o3-pro", "never-written"]
        );
    }
}
