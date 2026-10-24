use std::collections::BTreeSet;

use indexmap::IndexMap;
use serde_json::Value;

use crate::catalog::Catalog;
use crate::wire::Fields;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precedence {
    Fallback,
    Override,
}

pub type SparseRow = IndexMap<String, Option<Value>>;

#[derive(Clone, Debug)]
pub enum ApplicationWrite {
    Registration {
        key: String,
        inherits_from: Option<String>,
        fields: SparseRow,
    },
    Direct {
        key: String,
        row: Fields,
    },
    Tombstone {
        key: String,
    },
}

#[derive(Clone, Debug)]
pub struct Owner {
    pub name: String,
    pub precedence: Precedence,
    pub rows: IndexMap<String, SparseRow>,
}

#[derive(Clone, Debug, Default)]
pub struct Layers {
    writes: Vec<ApplicationWrite>,
    owners: IndexMap<String, Owner>,
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

    pub fn set_owner(&mut self, owner: Owner) {
        self.owners.shift_remove(&owner.name);
        self.owners.insert(owner.name.clone(), owner);
    }

    pub fn drop_owner(&mut self, name: &str) {
        self.owners.shift_remove(name);
    }

    pub fn forget_writes(&mut self) {
        self.writes.clear();
    }

    pub fn compose<'a>(&'a self, base: &'a Catalog) -> impl Iterator<Item = ComposedRow<'a>> + 'a {
        let keys = self.all_keys(base);
        keys.into_iter()
            .filter_map(move |key| self.composed_row(base, &key))
    }

    pub fn compose_keys<'a>(
        &'a self,
        base: &'a Catalog,
        keys: &'a BTreeSet<String>,
    ) -> impl Iterator<Item = Option<ComposedRow<'a>>> + 'a {
        keys.iter()
            .map(move |key| self.composed_row(base, key.as_str()))
    }

    fn all_keys(&self, base: &Catalog) -> BTreeSet<String> {
        base.served()
            .map(|(key, _)| key.to_owned())
            .chain(self.writes.iter().map(|write| write.key().to_owned()))
            .chain(
                self.owners
                    .values()
                    .flat_map(|owner| owner.rows.keys().cloned()),
            )
            .collect()
    }

    fn composed_row<'a>(&'a self, base: &'a Catalog, key: &str) -> Option<ComposedRow<'a>> {
        let (fields, registered) = self.compose_fields(base, key)?;
        Some(ComposedRow {
            key: self.key_ref(base, key)?,
            fields,
            registered,
        })
    }

    fn key_ref<'a>(&'a self, base: &'a Catalog, key: &str) -> Option<&'a str> {
        base.lookup(key)
            .filter(|matched| matched.matched_key == key)
            .map(|matched| matched.matched_key)
            .or_else(|| {
                self.writes
                    .iter()
                    .find_map(|write| (write.key() == key).then_some(write.key_ref()))
            })
            .or_else(|| {
                self.owners.values().find_map(|owner| {
                    owner
                        .rows
                        .keys()
                        .find_map(|candidate| (candidate == key).then_some(candidate.as_str()))
                })
            })
    }

    fn compose_fields(&self, base: &Catalog, key: &str) -> Option<(Fields, BTreeSet<String>)> {
        let base_fields = base
            .lookup(key)
            .filter(|matched| matched.matched_key == key)
            .map(|entry| entry.entry.fields().clone());
        let has_application_write = self.writes.iter().any(|write| write.key() == key);
        let has_override_owner = self
            .owners
            .values()
            .any(|owner| owner.precedence == Precedence::Override && owner.rows.contains_key(key));

        let mut row = if base_fields.is_none() && !has_application_write && !has_override_owner {
            self.apply_owner_rows(key, Precedence::Fallback, None)
        } else {
            base_fields.map(|fields| (fields, BTreeSet::new()))
        };

        for write in self.writes.iter().filter(|write| write.key() == key) {
            match write {
                ApplicationWrite::Registration {
                    inherits_from,
                    fields,
                    ..
                } => {
                    if row.is_none() {
                        let inherited = inherits_from.as_deref().and_then(|name| {
                            base.lookup(name)
                                .filter(|matched| matched.matched_key == name)
                                .map(|entry| entry.entry.fields().clone())
                        });
                        row = Some((inherited.unwrap_or_default(), BTreeSet::new()));
                    }
                    let (current, registered) = row.get_or_insert_default();
                    apply_sparse(current, registered, fields, true);
                }
                ApplicationWrite::Direct { row: direct, .. } => {
                    row = Some((direct.clone(), BTreeSet::new()));
                }
                ApplicationWrite::Tombstone { .. } => {
                    row = None;
                }
            }
        }
        self.apply_owner_rows(key, Precedence::Override, row)
    }

    fn apply_owner_rows(
        &self,
        key: &str,
        precedence: Precedence,
        mut row: Option<(Fields, BTreeSet<String>)>,
    ) -> Option<(Fields, BTreeSet<String>)> {
        for owner in self
            .owners
            .values()
            .filter(|owner| owner.precedence == precedence)
        {
            if let Some(fields) = owner.rows.get(key) {
                let (current, registered) = row.get_or_insert_default();
                apply_sparse(current, registered, fields, false);
            }
        }
        row
    }
}

impl ApplicationWrite {
    fn key(&self) -> &str {
        match self {
            Self::Registration { key, .. } | Self::Direct { key, .. } | Self::Tombstone { key } => {
                key
            }
        }
    }

    fn key_ref(&self) -> &str {
        self.key()
    }
}

fn apply_sparse(
    row: &mut Fields,
    registered: &mut BTreeSet<String>,
    fields: &SparseRow,
    track_registration: bool,
) {
    for (field, value) in fields {
        match value {
            Some(value) => {
                row.insert(field.clone(), value.clone());
            }
            None => {
                row.remove(field);
            }
        }
        if track_registration {
            registered.insert(field.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    fn catalog(document: Value) -> Catalog {
        Catalog::parse(document.to_string().as_bytes())
            .unwrap()
            .value
    }

    fn sparse(fields: Value) -> SparseRow {
        let Value::Object(fields) = fields else {
            panic!("sparse row must be an object")
        };
        fields
            .into_iter()
            .map(|(key, value)| (key, (!value.is_null()).then_some(value)))
            .collect()
    }

    fn owner(name: &str, precedence: Precedence, rows: Value) -> Owner {
        let Value::Object(rows) = rows else {
            panic!("owner rows must be an object")
        };
        Owner {
            name: name.to_owned(),
            precedence,
            rows: rows
                .into_iter()
                .map(|(key, fields)| (key, sparse(fields)))
                .collect(),
        }
    }

    fn row<'a>(layers: &'a Layers, base: &'a Catalog, key: &str) -> ComposedRow<'a> {
        layers
            .compose(base)
            .find(|row| row.key == key)
            .unwrap_or_else(|| panic!("missing composed row {key}"))
    }

    #[rstest]
    fn precedence_and_fallback_isolation_hold() {
        let base = catalog(json!({
            "base": {"litellm_provider": "base", "shared": "base", "keep": true}
        }));
        let mut layers = Layers::default();
        layers.set_owner(owner(
            "fallback",
            Precedence::Fallback,
            json!({
                "base": {"shared": "fallback", "fallback_only": true},
                "fallback-row": {"from_fallback": true}
            }),
        ));
        layers.record(ApplicationWrite::Registration {
            key: "base".into(),
            inherits_from: None,
            fields: sparse(json!({"shared": "application"})),
        });
        layers.record(ApplicationWrite::Direct {
            key: "application-row".into(),
            row: serde_json::from_value(json!({"shared": "application"})).unwrap(),
        });
        layers.set_owner(owner(
            "override",
            Precedence::Override,
            json!({"base": {"shared": "override"}, "application-row": {"added": true}}),
        ));

        let base_row = row(&layers, &base, "base");
        assert_eq!(base_row.fields.get("shared"), Some(&json!("override")));
        assert_eq!(base_row.fields.get("fallback_only"), None);
        assert_eq!(base_row.fields.get("keep"), Some(&json!(true)));
        let fallback_row = row(&layers, &base, "fallback-row");
        assert_eq!(fallback_row.fields.get("from_fallback"), Some(&json!(true)));
        let application_row = row(&layers, &base, "application-row");
        assert_eq!(
            application_row.fields.get("shared"),
            Some(&json!("application"))
        );
        assert_eq!(application_row.fields.get("added"), Some(&json!(true)));
    }

    #[rstest]
    fn aliases_compose_as_independent_exact_keys() {
        let base = catalog(json!({
            "canonical": {"litellm_provider": "base", "aliases": ["alias"], "base": true}
        }));
        let layers = Layers::default();
        assert_eq!(
            layers.compose(&base).map(|row| row.key).collect::<Vec<_>>(),
            ["alias", "canonical"]
        );

        let mut layers = Layers::default();
        layers.record(ApplicationWrite::Direct {
            key: "alias".into(),
            row: serde_json::from_value(json!({"alias_only": true})).unwrap(),
        });
        assert_eq!(
            row(&layers, &base, "alias").fields.get("alias_only"),
            Some(&json!(true))
        );
        assert_eq!(
            row(&layers, &base, "canonical").fields.get("base"),
            Some(&json!(true))
        );
        assert_eq!(
            base.lookup("alias").unwrap().entry.fields().get("base"),
            Some(&json!(true))
        );

        layers.record(ApplicationWrite::Tombstone {
            key: "alias".into(),
        });
        assert!(
            layers
                .compose(&base)
                .find(|composed| composed.key == "alias")
                .is_none()
        );
        assert_eq!(
            base.lookup("canonical").unwrap().entry.fields().get("base"),
            Some(&json!(true))
        );

        let mut case_variant = Layers::default();
        case_variant.record(ApplicationWrite::Direct {
            key: "CANONICAL".into(),
            row: serde_json::from_value(json!({"separate": true})).unwrap(),
        });
        assert_eq!(
            row(&case_variant, &base, "canonical").fields.get("base"),
            Some(&json!(true))
        );
        assert_eq!(
            row(&case_variant, &base, "CANONICAL")
                .fields
                .get("separate"),
            Some(&json!(true))
        );
    }

    #[rstest]
    fn registrations_inherit_fresh_base_then_accumulate_sparse_changes() {
        let base = catalog(json!({
            "source": {"litellm_provider": "base", "aliases": ["source-alias"], "keep": true, "remove": true}
        }));
        let mut layers = Layers::default();
        layers.record(ApplicationWrite::Registration {
            key: "new".into(),
            inherits_from: Some("source-alias".into()),
            fields: sparse(json!({"remove": null, "first": 1})),
        });
        layers.record(ApplicationWrite::Registration {
            key: "new".into(),
            inherits_from: None,
            fields: sparse(json!({"second": 2})),
        });

        let composed = row(&layers, &base, "new");
        assert_eq!(composed.fields.get("keep"), Some(&json!(true)));
        assert_eq!(composed.fields.get("remove"), None);
        assert_eq!(composed.fields.get("first"), Some(&json!(1)));
        assert_eq!(composed.fields.get("second"), Some(&json!(2)));
        assert_eq!(
            composed.registered,
            BTreeSet::from(["first".to_owned(), "remove".to_owned(), "second".to_owned()])
        );
        assert_eq!(
            base.lookup("source").unwrap().entry.fields().get("remove"),
            Some(&json!(true))
        );
    }

    #[rstest]
    fn direct_write_resets_registration_provenance_and_tombstone_can_be_revived() {
        let base = catalog(json!({"base": {"litellm_provider": "base"}}));
        let mut layers = Layers::default();
        layers.record(ApplicationWrite::Registration {
            key: "entry".into(),
            inherits_from: None,
            fields: sparse(json!({"registered": true})),
        });
        layers.record(ApplicationWrite::Direct {
            key: "entry".into(),
            row: serde_json::from_value(json!({"direct": true})).unwrap(),
        });
        let direct = row(&layers, &base, "entry");
        assert_eq!(
            direct.fields,
            serde_json::from_value::<Fields>(json!({"direct": true})).unwrap()
        );
        assert!(direct.registered.is_empty());

        layers.record(ApplicationWrite::Tombstone {
            key: "entry".into(),
        });
        assert!(
            layers
                .compose(&base)
                .find(|row| row.key == "entry")
                .is_none()
        );
        layers.record(ApplicationWrite::Direct {
            key: "entry".into(),
            row: serde_json::from_value(json!({"revived": true})).unwrap(),
        });
        assert_eq!(
            row(&layers, &base, "entry").fields.get("revived"),
            Some(&json!(true))
        );
        layers.forget_writes();
        assert!(
            layers
                .compose(&base)
                .find(|row| row.key == "entry")
                .is_none()
        );
    }

    #[rstest]
    fn replacing_owner_moves_it_to_the_end_of_its_precedence_order() {
        let base = catalog(json!({"base": {"litellm_provider": "base"}}));
        let mut layers = Layers::default();
        layers.set_owner(owner(
            "first",
            Precedence::Override,
            json!({"base": {"winner": "first"}}),
        ));
        layers.set_owner(owner(
            "second",
            Precedence::Override,
            json!({"base": {"winner": "second"}}),
        ));
        layers.set_owner(owner(
            "first",
            Precedence::Override,
            json!({"base": {"winner": "replaced"}}),
        ));

        assert_eq!(
            row(&layers, &base, "base").fields.get("winner"),
            Some(&json!("replaced"))
        );
        layers.drop_owner("first");
        assert_eq!(
            row(&layers, &base, "base").fields.get("winner"),
            Some(&json!("second"))
        );
    }

    #[rstest]
    fn override_owner_replaces_fallback_only_row_and_drop_restores_fallback() {
        let base = catalog(json!({"base": {"litellm_provider": "base"}}));
        let mut layers = Layers::default();
        layers.set_owner(owner(
            "fallback",
            Precedence::Fallback,
            json!({"new": {"shared": "fallback", "fallback_only": true}}),
        ));
        layers.set_owner(owner(
            "override",
            Precedence::Override,
            json!({"new": {"shared": "override"}}),
        ));

        let overridden = row(&layers, &base, "new");
        assert_eq!(overridden.fields.get("shared"), Some(&json!("override")));
        assert_eq!(overridden.fields.get("fallback_only"), None);

        layers.drop_owner("override");
        let fallback = row(&layers, &base, "new");
        assert_eq!(fallback.fields.get("shared"), Some(&json!("fallback")));
        assert_eq!(fallback.fields.get("fallback_only"), Some(&json!(true)));
    }

    #[rstest]
    fn compose_keys_returns_requested_order_and_none_for_missing_or_tombstoned_rows() {
        let base = catalog(json!({
            "z": {"litellm_provider": "base"},
            "a": {"litellm_provider": "base"},
            "gone": {"litellm_provider": "base"}
        }));
        let mut layers = Layers::default();
        layers.record(ApplicationWrite::Tombstone { key: "gone".into() });
        let keys = BTreeSet::from([
            "absent".to_owned(),
            "a".to_owned(),
            "gone".to_owned(),
            "z".to_owned(),
        ]);
        let result: Vec<Option<String>> = layers
            .compose_keys(&base, &keys)
            .map(|row| row.map(|row| row.key.to_owned()))
            .collect();
        assert_eq!(result, [Some("a".into()), None, None, Some("z".into())]);
    }
}
