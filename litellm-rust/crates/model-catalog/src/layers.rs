use std::collections::BTreeSet;

use indexmap::{IndexMap, IndexSet};
use serde_json::Value;

use crate::catalog::Catalog;
use crate::wire::Fields;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precedence {
    Fallback,
    Override,
}

pub type SparseRow = IndexMap<String, Option<Value>>;
pub type RegistrationHandle = u64;

#[derive(Clone, Debug)]
pub struct RegistrationOptions {
    pub alias_keys: Vec<String>,
    pub defaults: SparseRow,
    pub reset_fields: BTreeSet<String>,
    pub cleanup_overrides: Vec<String>,
    pub fallback: bool,
    pub names_the_row: bool,
    pub ignore_none: bool,
}

impl Default for RegistrationOptions {
    fn default() -> Self {
        Self {
            alias_keys: Vec::new(),
            defaults: SparseRow::new(),
            reset_fields: BTreeSet::new(),
            cleanup_overrides: Vec::new(),
            fallback: false,
            names_the_row: false,
            ignore_none: true,
        }
    }
}

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
    handles: Vec<Option<RegistrationHandle>>,
    options: Vec<Option<RegistrationOptions>>,
    owners: IndexMap<String, Owner>,
    next_handle: RegistrationHandle,
}

#[derive(Clone, Debug)]
pub struct ComposedRow<'a> {
    pub key: &'a str,
    pub group: String,
    pub fields: Fields,
    pub registered: BTreeSet<String>,
}

impl Layers {
    pub fn record(&mut self, write: ApplicationWrite) -> Option<RegistrationHandle> {
        let registration = matches!(write, ApplicationWrite::Registration { .. });
        let handle = registration.then(|| {
            self.next_handle += 1;
            self.next_handle
        });
        self.writes.push(write);
        self.handles.push(handle);
        self.options
            .push(registration.then(RegistrationOptions::default));
        handle
    }

    pub fn record_registration(
        &mut self,
        key: String,
        inherits_from: Option<String>,
        fields: SparseRow,
        options: RegistrationOptions,
    ) -> RegistrationHandle {
        self.next_handle += 1;
        let handle = self.next_handle;
        self.writes.push(ApplicationWrite::Registration {
            key,
            inherits_from,
            fields,
        });
        self.handles.push(Some(handle));
        self.options.push(Some(options));
        handle
    }

    pub fn rebind_registration(
        &mut self,
        handle: RegistrationHandle,
        key: String,
        inherits_from: Option<String>,
        fields: SparseRow,
        options: RegistrationOptions,
    ) -> bool {
        let Some(index) = self
            .handles
            .iter()
            .position(|candidate| *candidate == Some(handle))
        else {
            return false;
        };
        let Some(ApplicationWrite::Registration {
            key: current_key,
            inherits_from: current_inherits,
            fields: current_fields,
        }) = self.writes.get_mut(index)
        else {
            return false;
        };
        *current_key = key;
        *current_inherits = inherits_from;
        *current_fields = fields;
        self.options[index] = Some(options);
        true
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
        self.handles.clear();
        self.options.clear();
    }

    pub fn compose<'a>(&'a self, base: &'a Catalog) -> impl Iterator<Item = ComposedRow<'a>> + 'a {
        let keys = self.all_keys(base, self.writes.len());
        keys.into_iter()
            .filter_map(move |key| self.composed_row(base, &key, self.writes.len()))
    }

    pub fn compose_keys<'a>(
        &'a self,
        base: &'a Catalog,
        keys: &'a BTreeSet<String>,
    ) -> impl Iterator<Item = Option<ComposedRow<'a>>> + 'a {
        keys.iter()
            .map(move |key| self.composed_row(base, key.as_str(), self.writes.len()))
    }

    pub fn compose_prefix<'a>(
        &'a self,
        base: &'a Catalog,
        handle: RegistrationHandle,
    ) -> Option<Vec<ComposedRow<'a>>> {
        let index = self
            .handles
            .iter()
            .position(|candidate| *candidate == Some(handle))?;
        Some(
            self.all_keys(base, index)
                .into_iter()
                .filter_map(|key| self.composed_row(base, &key, index))
                .collect(),
        )
    }

    fn all_keys(&self, base: &Catalog, write_limit: usize) -> IndexSet<String> {
        base.served()
            .map(|(key, _)| key.to_owned())
            .chain(
                self.writes
                    .iter()
                    .zip(self.options.iter())
                    .take(write_limit)
                    .flat_map(|(write, options)| {
                        std::iter::once(write.key().to_owned()).chain(
                            options
                                .as_ref()
                                .into_iter()
                                .flat_map(|options| options.alias_keys.iter().cloned()),
                        )
                    }),
            )
            .chain(
                self.owners
                    .values()
                    .flat_map(|owner| owner.rows.keys().cloned()),
            )
            .collect()
    }

    fn composed_row<'a>(
        &'a self,
        base: &'a Catalog,
        key: &str,
        write_limit: usize,
    ) -> Option<ComposedRow<'a>> {
        let (fields, registered) = self.compose_fields(base, key, write_limit)?;
        Some(ComposedRow {
            key: self.key_ref(base, key, write_limit)?,
            group: self.group_key(base, key, write_limit),
            fields,
            registered,
        })
    }

    fn group_key(&self, base: &Catalog, key: &str, write_limit: usize) -> String {
        if let Some(group) = self
            .writes
            .iter()
            .zip(self.options.iter())
            .take(write_limit)
            .rev()
            .find_map(|(write, options)| {
                write_keys(write, options.as_ref())
                    .contains(&key.to_owned())
                    .then(|| write.key().to_owned())
            })
        {
            return group;
        }
        base.lookup(key).map_or_else(
            || key.to_owned(),
            |matched| matched.canonical_key.to_owned(),
        )
    }

    fn key_ref<'a>(&'a self, base: &'a Catalog, key: &str, write_limit: usize) -> Option<&'a str> {
        base.lookup(key)
            .filter(|matched| matched.matched_key == key)
            .map(|matched| matched.matched_key)
            .or_else(|| {
                self.writes
                    .iter()
                    .zip(self.options.iter())
                    .take(write_limit)
                    .find_map(|(write, options)| {
                        if write.key() == key {
                            return Some(write.key_ref());
                        }
                        options
                            .as_ref()
                            .and_then(|options| {
                                options
                                    .alias_keys
                                    .iter()
                                    .find(|alias| alias.as_str() == key)
                            })
                            .map(String::as_str)
                    })
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

    fn compose_fields(
        &self,
        base: &Catalog,
        key: &str,
        write_limit: usize,
    ) -> Option<(Fields, BTreeSet<String>)> {
        let base_fields = base.lookup(key).map(|entry| entry.entry.fields().clone());
        let has_application_write = self
            .writes
            .iter()
            .zip(self.options.iter())
            .take(write_limit)
            .any(|(write, options)| write_keys(write, options.as_ref()).contains(&key.to_owned()));
        let has_override_owner = self
            .owners
            .values()
            .any(|owner| owner.precedence == Precedence::Override && owner.rows.contains_key(key));

        let mut row = if base_fields.is_none() && !has_application_write && !has_override_owner {
            self.apply_owner_rows(key, Precedence::Fallback, None)
        } else {
            base_fields.map(|fields| (fields, BTreeSet::new()))
        };

        for (index, (write, options)) in self
            .writes
            .iter()
            .zip(self.options.iter())
            .take(write_limit)
            .enumerate()
            .filter(|(_, (write, options))| {
                write_keys(write, options.as_ref()).contains(&key.to_owned())
            })
        {
            match write {
                ApplicationWrite::Registration {
                    inherits_from,
                    fields,
                    ..
                } => {
                    if row.is_none() {
                        let inherited = inherits_from.as_deref().and_then(|name| {
                            self.compose_fields(base, name, index)
                                .map(|(fields, _)| fields)
                        });
                        row = Some((inherited.unwrap_or_default(), BTreeSet::new()));
                    }
                    let (current, registered) = row.get_or_insert_default();
                    let options = options.as_ref().cloned().unwrap_or_default();
                    apply_registration(current, registered, fields, &options);
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

fn write_keys(write: &ApplicationWrite, options: Option<&RegistrationOptions>) -> Vec<String> {
    let mut keys = vec![write.key().to_owned()];
    if let Some(options) = options {
        keys.extend(options.alias_keys.iter().cloned());
    }
    keys
}

fn apply_registration(
    row: &mut Fields,
    registered: &mut BTreeSet<String>,
    fields: &SparseRow,
    options: &RegistrationOptions,
) {
    if options.fallback || options.names_the_row {
        row.remove("litellm_provider");
    }
    if options.fallback && !options.names_the_row && !options.cleanup_overrides.is_empty() {
        let cleanup = options
            .cleanup_overrides
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        cleanup_fallback_prices(row, &cleanup, fields);
    }
    for (field, value) in &options.defaults {
        if let Some(value) = value {
            row.entry(field.clone()).or_insert_with(|| value.clone());
        }
    }
    for (field, value) in fields {
        match value {
            Some(value) => {
                let merged = match (row.get(field), value) {
                    (Some(Value::Object(existing)), Value::Object(new)) => {
                        let mut object = existing.clone();
                        object.extend(new.clone());
                        Value::Object(object)
                    }
                    _ => value.clone(),
                };
                row.insert(field.clone(), merged);
                registered.insert(field.clone());
            }
            None if options.ignore_none && options.reset_fields.contains(field) => {
                row.insert(field.clone(), Value::Null);
                registered.insert(field.clone());
            }
            None if options.ignore_none => {}
            None => {
                row.remove(field);
                registered.insert(field.clone());
            }
        }
    }
    for field in &options.reset_fields {
        row.insert(field.clone(), Value::Null);
    }
}

fn cleanup_fallback_prices(row: &mut Fields, overridden: &[&str], fields: &SparseRow) {
    row.remove("litellm_provider");
    let matches = |name: &str| crate::wire::matches_fallback_direction(name, overridden);
    let flat: Fields = fields
        .iter()
        .filter_map(|(name, value)| value.clone().map(|value| (name.clone(), value)))
        .filter(|(name, _)| overridden.iter().any(|direction| name == direction))
        .collect();
    let Some(Value::Array(tiers)) = row.get_mut("tiered_pricing") else {
        if let Some(Value::Object(off_peak)) = row.get_mut("off_peak_pricing") {
            off_peak.retain(|name, _| !matches(name));
        }
        row.retain(|name, _| !matches(name));
        return;
    };
    for tier in tiers.iter_mut().filter_map(Value::as_object_mut) {
        tier.retain(|name, _| !matches(name));
        tier.extend(flat.clone());
    }
    if let Some(Value::Object(off_peak)) = row.get_mut("off_peak_pricing") {
        off_peak.retain(|name, _| !matches(name));
    }
    row.retain(|name, _| !matches(name));
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
    fn composition_preserves_served_order_and_appends_new_keys() {
        let base = Catalog::parse(
            br#"{"z":{"litellm_provider":"test","aliases":["b"]},"a":{"litellm_provider":"test"}}"#,
        )
        .unwrap()
        .value;
        let mut layers = Layers::default();
        assert_eq!(
            layers.compose(&base).map(|row| row.key).collect::<Vec<_>>(),
            base.served().map(|(key, _)| key).collect::<Vec<_>>()
        );
        layers.record(ApplicationWrite::Direct {
            key: "d".into(),
            row: Fields::new(),
        });
        layers.record(ApplicationWrite::Direct {
            key: "z".into(),
            row: Fields::new(),
        });
        layers.set_owner(owner("owner", Precedence::Override, json!({"c": {}})));
        assert_eq!(
            layers.compose(&base).map(|row| row.key).collect::<Vec<_>>(),
            vec!["z", "a", "b", "d", "c"]
        );
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
            ["canonical", "alias"]
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
    fn registration_none_is_ignored_while_owner_none_deletes() {
        let base = catalog(json!({"model": {"value": true}}));
        let mut layers = Layers::default();
        layers.record(ApplicationWrite::Registration {
            key: "model".into(),
            inherits_from: None,
            fields: sparse(json!({"value": null})),
        });
        assert_eq!(
            row(&layers, &base, "model").fields.get("value"),
            Some(&json!(true))
        );
        layers.set_owner(owner(
            "override",
            Precedence::Override,
            json!({"model": {"value": null}}),
        ));
        assert_eq!(row(&layers, &base, "model").fields.get("value"), None);
        layers.drop_owner("override");
        assert_eq!(
            row(&layers, &base, "model").fields.get("value"),
            Some(&json!(true))
        );

        let reset_base = catalog(json!({"model": {"explicit_cache_read_input_token_cost": 0.2}}));
        let mut reset = Layers::default();
        reset.record_registration(
            "model".into(),
            None,
            sparse(json!({})),
            RegistrationOptions {
                reset_fields: BTreeSet::from(["explicit_cache_read_input_token_cost".into()]),
                ignore_none: true,
                ..RegistrationOptions::default()
            },
        );
        assert_eq!(
            row(&reset, &reset_base, "model")
                .fields
                .get("explicit_cache_read_input_token_cost"),
            Some(&Value::Null)
        );
    }

    #[rstest]
    fn registration_mapping_fields_merge_shallowly() {
        let base = catalog(json!({"model": {"metadata": {"base": true, "keep": true}}}));
        let mut layers = Layers::default();
        layers.record_registration(
            "model".into(),
            None,
            sparse(json!({"metadata": {"new": true}})),
            RegistrationOptions {
                ignore_none: true,
                ..RegistrationOptions::default()
            },
        );
        assert_eq!(
            row(&layers, &base, "model").fields.get("metadata"),
            Some(&json!({"base": true, "keep": true, "new": true}))
        );
    }

    #[rstest]
    fn fallback_cleanup_is_conditional() {
        let base = catalog(json!({
            "source": {
                "litellm_provider": "openai",
                "input_cost_per_token": 1,
                "input_cost_per_token_above_1": 2,
                "input_cost_per_token_cache_read": 3,
                "output_cost_per_token": 8,
                "output_cost_per_reasoning_token": 9,
                "tiered_pricing": [{
                    "input_cost_per_token": 4,
                    "input_cost_per_token_above_1": 5,
                    "output_cost_per_token": 6,
                    "output_cost_per_reasoning_token": 7
                }],
                "off_peak_pricing": {"input_cost_per_token": 7}
            }
        }));
        let mut layers = Layers::default();
        layers.record_registration(
            "target".into(),
            Some("source".into()),
            sparse(json!({"input_cost_per_token": 10})),
            RegistrationOptions {
                cleanup_overrides: vec!["input_cost_per_token".into()],
                fallback: true,
                ignore_none: true,
                ..RegistrationOptions::default()
            },
        );
        let fields = &row(&layers, &base, "target").fields;
        assert_eq!(fields.get("litellm_provider"), None);
        assert_eq!(fields.get("input_cost_per_token"), Some(&json!(10)));
        assert_eq!(fields.get("input_cost_per_token_above_1"), None);
        assert_eq!(
            fields.get("input_cost_per_token_cache_read"),
            Some(&json!(3))
        );
        assert_eq!(
            fields.get("output_cost_per_reasoning_token"),
            Some(&json!(9))
        );
        assert_eq!(
            fields.get("tiered_pricing"),
            Some(&json!([{
                "input_cost_per_token": 10,
                "output_cost_per_token": 6,
                "output_cost_per_reasoning_token": 7
            }]))
        );
        assert_eq!(fields.get("off_peak_pricing"), Some(&json!({})));

        let mut named = Layers::default();
        named.record_registration(
            "target".into(),
            Some("source".into()),
            sparse(json!({"input_cost_per_token": 10})),
            RegistrationOptions {
                names_the_row: true,
                cleanup_overrides: vec!["input_cost_per_token".into()],
                ignore_none: true,
                ..RegistrationOptions::default()
            },
        );
        let named_fields = &row(&named, &base, "target").fields;
        assert_eq!(named_fields.get("litellm_provider"), None);
        assert_eq!(
            named_fields.get("input_cost_per_token_above_1"),
            Some(&json!(2))
        );

        let mut output = Layers::default();
        output.record_registration(
            "target".into(),
            Some("source".into()),
            sparse(json!({"output_cost_per_token": 11})),
            RegistrationOptions {
                cleanup_overrides: vec![
                    "output_cost_per_token".into(),
                    "output_cost_per_reasoning_token".into(),
                ],
                fallback: true,
                ignore_none: true,
                ..RegistrationOptions::default()
            },
        );
        let output_fields = &row(&output, &base, "target").fields;
        assert_eq!(output_fields.get("output_cost_per_token"), Some(&json!(11)));
        assert_eq!(output_fields.get("output_cost_per_reasoning_token"), None);
        assert_eq!(
            output_fields.get("tiered_pricing"),
            Some(
                &json!([{"input_cost_per_token": 4, "input_cost_per_token_above_1": 5, "output_cost_per_token": 11}])
            )
        );

        let mut builtin = Layers::default();
        builtin.record_registration(
            "source".into(),
            None,
            sparse(json!({"input_cost_per_token": 12})),
            RegistrationOptions {
                ignore_none: true,
                ..RegistrationOptions::default()
            },
        );
        let builtin_fields = &row(&builtin, &base, "source").fields;
        assert_eq!(
            builtin_fields.get("litellm_provider"),
            Some(&json!("openai"))
        );
        assert_eq!(
            builtin_fields.get("input_cost_per_token_above_1"),
            Some(&json!(2))
        );
    }

    #[rstest]
    fn fallback_without_flat_override_strips_source_provider() {
        let base = catalog(json!({
            "source": {
                "litellm_provider": "openai",
                "input_cost_per_token": 0.2,
                "supports_function_calling": true
            }
        }));
        let mut layers = Layers::default();
        layers.record_registration(
            "target".into(),
            Some("source".into()),
            sparse(json!({"litellm_provider": "custom", "max_input_tokens": 4096})),
            RegistrationOptions {
                fallback: true,
                ignore_none: true,
                ..RegistrationOptions::default()
            },
        );

        let source = row(&layers, &base, "source");
        let target = row(&layers, &base, "target");
        assert_eq!(
            source.fields.get("litellm_provider"),
            Some(&json!("openai"))
        );
        assert_eq!(
            target.fields.get("litellm_provider"),
            Some(&json!("custom"))
        );
        assert_eq!(target.fields.get("input_cost_per_token"), Some(&json!(0.2)));
        assert_eq!(
            target.fields.get("supports_function_calling"),
            Some(&json!(true))
        );
    }

    #[rstest]
    fn generalized_defaults_are_below_inherited_fields_and_caller_fields() {
        let base = catalog(json!({
            "source": {
                "supports_function_calling": true,
                "supports_vision": true,
                "mode": "chat"
            }
        }));
        let mut layers = Layers::default();
        layers.record_registration(
            "target".into(),
            Some("source".into()),
            sparse(json!({"supports_vision": false})),
            RegistrationOptions {
                defaults: sparse(json!({
                    "supports_function_calling": false,
                    "supports_vision": true,
                    "max_input_tokens": 8192
                })),
                fallback: true,
                ignore_none: true,
                ..RegistrationOptions::default()
            },
        );

        let fields = &row(&layers, &base, "target").fields;
        assert_eq!(fields.get("supports_function_calling"), Some(&json!(true)));
        assert_eq!(fields.get("supports_vision"), Some(&json!(false)));
        assert_eq!(fields.get("max_input_tokens"), Some(&json!(8192)));
        assert_eq!(fields.get("mode"), Some(&json!("chat")));
    }

    #[rstest]
    fn rebind_preserves_order_and_removes_stale_destination() {
        let base = catalog(json!({"other": {"base": true}}));
        let mut layers = Layers::default();
        layers.record(ApplicationWrite::Direct {
            key: "before".into(),
            row: Fields::new(),
        });
        let handle = layers.record_registration(
            "old".into(),
            None,
            sparse(json!({"registered": true})),
            RegistrationOptions::default(),
        );
        layers.record(ApplicationWrite::Tombstone {
            key: "other".into(),
        });
        assert!(layers.rebind_registration(
            handle,
            "new".into(),
            None,
            sparse(json!({"registered": true})),
            RegistrationOptions::default(),
        ));
        assert_eq!(
            layers.compose(&base).map(|row| row.key).collect::<Vec<_>>(),
            vec!["before", "new"]
        );
        assert!(layers.compose(&base).all(|row| row.key != "old"));
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
        assert_eq!(composed.fields.get("remove"), Some(&json!(true)));
        assert_eq!(composed.fields.get("first"), Some(&json!(1)));
        assert_eq!(composed.fields.get("second"), Some(&json!(2)));
        assert_eq!(
            composed.registered,
            BTreeSet::from(["first".to_owned(), "second".to_owned()])
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
