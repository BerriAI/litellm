use std::collections::BTreeSet;

use indexmap::{IndexMap, IndexSet};
use serde_json::Value;

use crate::catalog::Catalog;
use crate::wire::Fields;

pub type SparseRow = IndexMap<String, Option<Value>>;

#[derive(Clone, Debug, PartialEq)]
pub enum ApplicationWrite {
    Registration {
        key: String,
        inherits_from: Option<String>,
        fields: SparseRow,
    },
    Direct {
        key: String,
        fields: SparseRow,
    },
    Tombstone {
        key: String,
    },
}

#[derive(Clone, Debug, Default)]
pub struct Layers {
    writes: Vec<ApplicationWrite>,
}

#[derive(Clone, Debug)]
pub struct ComposedRow<'a> {
    pub key: &'a str,
    pub fields: Fields,
    pub registered: BTreeSet<String>,
}

impl Layers {
    pub fn record(&mut self, write: ApplicationWrite) {
        self.writes.push(write);
    }

    pub fn forget_writes(&mut self) {
        self.writes.clear();
    }

    pub fn compose<'a>(&'a self, base: &'a Catalog) -> impl Iterator<Item = ComposedRow<'a>> + 'a {
        let keys: IndexSet<&'a str> = base
            .served()
            .map(|(key, _)| key)
            .chain(self.writes.iter().map(ApplicationWrite::key))
            .collect();
        keys.into_iter()
            .filter_map(move |key| self.composed_row(base, key))
    }

    pub fn compose_keys<'a>(
        &'a self,
        base: &'a Catalog,
        keys: &'a BTreeSet<String>,
    ) -> impl Iterator<Item = Option<ComposedRow<'a>>> + 'a {
        keys.iter().map(move |key| self.composed_row(base, key))
    }

    fn composed_row<'a>(&'a self, base: &'a Catalog, key: &str) -> Option<ComposedRow<'a>> {
        let exact = exact_fields(base, key);
        let initial = exact.map(|(key, fields)| (key, fields.clone(), BTreeSet::new()));
        let composed = self
            .writes
            .iter()
            .filter(|write| write.key() == key)
            .fold(initial, |row, write| apply(base, row, write));
        composed.map(|(key, fields, registered)| ComposedRow {
            key,
            fields,
            registered,
        })
    }
}

type Row<'a> = (&'a str, Fields, BTreeSet<String>);

fn apply<'a>(
    base: &'a Catalog,
    row: Option<Row<'a>>,
    write: &'a ApplicationWrite,
) -> Option<Row<'a>> {
    match write {
        ApplicationWrite::Tombstone { .. } => None,
        ApplicationWrite::Registration {
            key,
            inherits_from,
            fields,
        } => {
            let (key, mut current, mut registered) = row.unwrap_or_else(|| {
                let inherited = inherits_from
                    .as_deref()
                    .and_then(|name| exact_fields(base, name))
                    .map(|(_, fields)| fields.clone())
                    .unwrap_or_default();
                (key.as_str(), inherited, BTreeSet::new())
            });
            apply_sparse(&mut current, fields);
            registered.extend(fields.keys().cloned());
            Some((key, current, registered))
        }
        ApplicationWrite::Direct { key, fields } => {
            let (key, mut current, mut registered) =
                row.unwrap_or_else(|| (key.as_str(), Fields::new(), BTreeSet::new()));
            apply_sparse(&mut current, fields);
            registered.retain(|field| !fields.contains_key(field));
            Some((key, current, registered))
        }
    }
}

fn exact_fields<'a>(base: &'a Catalog, key: &str) -> Option<(&'a str, &'a Fields)> {
    base.lookup(key)
        .filter(|matched| matched.matched_key == key)
        .map(|matched| (matched.matched_key, matched.entry.fields()))
}

impl ApplicationWrite {
    pub fn key(&self) -> &str {
        match self {
            Self::Registration { key, .. } | Self::Direct { key, .. } | Self::Tombstone { key } => {
                key
            }
        }
    }
}

fn apply_sparse(row: &mut Fields, fields: &SparseRow) {
    for (field, value) in fields {
        match value {
            Some(value) => {
                row.insert(field.clone(), value.clone());
            }
            None => {
                row.shift_remove(field);
            }
        }
    }
}

pub fn diff_row(
    key: &str,
    installed: Option<&Fields>,
    live: Option<&Fields>,
) -> Option<ApplicationWrite> {
    match (installed, live) {
        (None, None) => None,
        (Some(_), None) => Some(ApplicationWrite::Tombstone {
            key: key.to_owned(),
        }),
        (None, Some(live)) => Some(ApplicationWrite::Direct {
            key: key.to_owned(),
            fields: live
                .iter()
                .map(|(field, value)| (field.clone(), Some(value.clone())))
                .collect(),
        }),
        (Some(installed), Some(live)) => {
            let changed = live
                .iter()
                .filter(|(field, value)| installed.get(*field) != Some(*value))
                .map(|(field, value)| (field.clone(), Some(value.clone())));
            let removed = installed
                .keys()
                .filter(|field| !live.contains_key(*field))
                .map(|field| (field.clone(), None));
            let fields: SparseRow = changed.chain(removed).collect();
            (!fields.is_empty()).then(|| ApplicationWrite::Direct {
                key: key.to_owned(),
                fields,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serde_json::{Value, json};

    fn catalog(cache_read: f64) -> Catalog {
        Catalog::parse(
            json!({
                "gpt-4o": {
                    "litellm_provider": "openai",
                    "aliases": ["gpt-4o-alias"],
                    "input_cost_per_token": 2.5e-06,
                    "cache_read_input_token_cost": cache_read
                },
                "responses/gpt-4o": {"litellm_provider": "openai", "input_cost_per_token": 2.5e-06}
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap()
        .value
    }

    fn fields(value: Value) -> Fields {
        serde_json::from_value(value).unwrap()
    }

    fn sparse(value: Value) -> SparseRow {
        fields(value)
            .into_iter()
            .map(|(key, value)| (key, (!value.is_null()).then_some(value)))
            .collect()
    }

    fn registration(key: &str, inherits_from: Option<&str>, value: Value) -> ApplicationWrite {
        ApplicationWrite::Registration {
            key: key.into(),
            inherits_from: inherits_from.map(str::to_owned),
            fields: sparse(value),
        }
    }

    fn direct(key: &str, value: Value) -> ApplicationWrite {
        ApplicationWrite::Direct {
            key: key.into(),
            fields: sparse(value),
        }
    }

    fn tombstone(key: &str) -> ApplicationWrite {
        ApplicationWrite::Tombstone { key: key.into() }
    }

    fn layers(writes: Vec<ApplicationWrite>) -> Layers {
        writes
            .into_iter()
            .fold(Layers::default(), |mut layers, write| {
                layers.record(write);
                layers
            })
    }

    fn composed(layers: &Layers, base: &Catalog, key: &str) -> Option<Fields> {
        layers
            .compose(base)
            .find(|row| row.key == key)
            .map(|row| row.fields)
    }

    fn input_and_cache_read(row: Option<Fields>) -> Option<(Option<Value>, Option<Value>)> {
        row.map(|row| {
            (
                row.get("input_cost_per_token").cloned(),
                row.get("cache_read_input_token_cost").cloned(),
            )
        })
    }

    #[rstest]
    #[case::no_writes(vec![], 1.25e-06, "gpt-4o", Some((Some(json!(2.5e-06)), Some(json!(1.25e-06)))))]
    #[case::registration(
        vec![registration("gpt-4o", None, json!({"input_cost_per_token": 0.5}))],
        1.25e-06, "gpt-4o", Some((Some(json!(0.5)), Some(json!(1.25e-06))))
    )]
    #[case::then_direct(
        vec![
            registration("gpt-4o", None, json!({"input_cost_per_token": 0.5})),
            direct("gpt-4o", json!({"input_cost_per_token": 0.7, "cache_read_input_token_cost": null})),
        ],
        1.25e-06, "gpt-4o", Some((Some(json!(0.7)), None))
    )]
    #[case::base_changes_on_reload(
        vec![
            registration("gpt-4o", None, json!({"input_cost_per_token": 0.5})),
            direct("gpt-4o", json!({"input_cost_per_token": 0.7, "cache_read_input_token_cost": null})),
        ],
        1e-06, "gpt-4o", Some((Some(json!(0.7)), None))
    )]
    #[case::tombstone(vec![tombstone("responses/gpt-4o")], 1.25e-06, "responses/gpt-4o", None)]
    #[case::direct_on_top_of_a_new_base_value(
        vec![direct("gpt-4o", json!({"input_cost_per_token": 0.7}))],
        1e-06, "gpt-4o", Some((Some(json!(0.7)), Some(json!(1e-06))))
    )]
    #[case::registration_after_tombstone_inherits(
        vec![tombstone("gpt-4o"), registration("gpt-4o", Some("responses/gpt-4o"), json!({}))],
        1.25e-06, "gpt-4o", Some((Some(json!(2.5e-06)), None))
    )]
    fn writes_compose_over_the_base_in_log_order(
        #[case] writes: Vec<ApplicationWrite>,
        #[case] base_cache_read: f64,
        #[case] key: &str,
        #[case] expected: Option<(Option<Value>, Option<Value>)>,
    ) {
        let base = catalog(base_cache_read);
        assert_eq!(
            input_and_cache_read(composed(&layers(writes), &base, key)),
            expected
        );
    }

    #[rstest]
    fn compose_serves_every_base_key_then_new_keys_in_write_order() {
        let base = catalog(1.25e-06);
        let layers = layers(vec![
            direct("new-b", json!({"x": 1})),
            registration("new-a", Some("gpt-4o"), json!({})),
            direct("gpt-4o", json!({"x": 1})),
            tombstone("gpt-4o-alias"),
        ]);
        assert_eq!(
            layers.compose(&base).map(|row| row.key).collect::<Vec<_>>(),
            ["gpt-4o", "responses/gpt-4o", "new-b", "new-a"]
        );
    }

    #[rstest]
    fn a_registration_on_a_new_key_inherits_its_source_and_records_what_it_set() {
        let base = catalog(1.25e-06);
        let layers = layers(vec![
            registration(
                "my-gpt",
                Some("gpt-4o-alias"),
                json!({"cache_read_input_token_cost": null, "output_cost_per_token": 1.0}),
            ),
            direct("my-gpt", json!({"output_cost_per_token": 2.0})),
        ]);
        let row = layers
            .compose(&base)
            .find(|row| row.key == "my-gpt")
            .unwrap();
        assert_eq!(
            row.fields.get("input_cost_per_token"),
            Some(&json!(2.5e-06))
        );
        assert_eq!(row.fields.get("cache_read_input_token_cost"), None);
        assert_eq!(row.fields.get("output_cost_per_token"), Some(&json!(2.0)));
        assert_eq!(
            row.registered,
            BTreeSet::from(["cache_read_input_token_cost".to_owned()])
        );
        assert_eq!(
            base.lookup("gpt-4o")
                .unwrap()
                .entry
                .field("cache_read_input_token_cost"),
            Some(&json!(1.25e-06))
        );
    }

    #[rstest]
    fn compose_keys_answers_each_requested_key_and_none_when_it_is_not_served() {
        let base = catalog(1.25e-06);
        let layers = layers(vec![tombstone("gpt-4o"), direct("added", json!({"x": 1}))]);
        let keys = BTreeSet::from([
            "absent".to_owned(),
            "added".to_owned(),
            "gpt-4o".to_owned(),
            "responses/gpt-4o".to_owned(),
        ]);
        assert_eq!(
            layers
                .compose_keys(&base, &keys)
                .map(|row| row.map(|row| row.key))
                .collect::<Vec<_>>(),
            [None, Some("added"), None, Some("responses/gpt-4o")]
        );
    }

    #[rstest]
    fn forget_writes_serves_the_base_again() {
        let base = catalog(1.25e-06);
        let mut layers = layers(vec![tombstone("gpt-4o"), direct("added", json!({"x": 1}))]);
        layers.forget_writes();
        assert_eq!(
            layers.compose(&base).map(|row| row.key).collect::<Vec<_>>(),
            base.served().map(|(key, _)| key).collect::<Vec<_>>()
        );
    }

    #[rstest]
    #[case::equal(Some(json!({"a": 1, "b": [1, 2]})), Some(json!({"a": 1, "b": [1, 2]})), None)]
    #[case::changed_and_removed(
        Some(json!({"a": 1, "b": 2, "c": 3})),
        Some(json!({"a": 1, "b": 5, "d": 4})),
        Some(direct("k", json!({"b": 5, "d": 4, "c": null})))
    )]
    #[case::new_key(None, Some(json!({"a": 1, "b": null})), Some(ApplicationWrite::Direct {
        key: "k".into(),
        fields: [("a".to_owned(), Some(json!(1))), ("b".to_owned(), Some(Value::Null))].into_iter().collect(),
    }))]
    #[case::removed_key(Some(json!({"a": 1})), None, Some(tombstone("k")))]
    #[case::nested_list_edited_in_place(
        Some(json!({"modalities": ["text", "image"], "a": 1})),
        Some(json!({"modalities": ["text", "audio"], "a": 1})),
        Some(direct("k", json!({"modalities": ["text", "audio"]})))
    )]
    fn diff_row_records_what_changed_since_install(
        #[case] installed: Option<Value>,
        #[case] live: Option<Value>,
        #[case] expected: Option<ApplicationWrite>,
    ) {
        let installed = installed.map(fields);
        let live = live.map(fields);
        assert_eq!(diff_row("k", installed.as_ref(), live.as_ref()), expected);
    }

    #[rstest]
    fn a_diffed_direct_write_composes_back_to_the_live_row() {
        let base = catalog(1.25e-06);
        let installed = base.lookup("gpt-4o").unwrap().entry.fields().clone();
        let live = fields(json!({
            "litellm_provider": "openai",
            "aliases": ["gpt-4o-alias"],
            "input_cost_per_token": 3.0,
            "max_tokens": 10
        }));
        let write = diff_row("gpt-4o", Some(&installed), Some(&live)).unwrap();
        assert_eq!(composed(&layers(vec![write]), &base, "gpt-4o"), Some(live));
    }
}
