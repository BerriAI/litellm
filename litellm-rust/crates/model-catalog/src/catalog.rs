use std::collections::HashMap;

use indexmap::IndexMap;
use serde_json::Value;

use crate::error::{IntegrityViolation, InvalidDocument, Problem};
use crate::fallback::{FallbackGeneralizations, FallbackRule};
use crate::model::Model;
use crate::wire::{EntryWire, Fields};

/// How far a registry may shrink against the one it replaces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntegrityLimits {
    min_model_count: usize,
    min_reference_ratio: f64,
}

#[derive(Clone, Debug)]
pub struct ModelEntry {
    fields: Fields,
    model: Option<Model>,
}

#[derive(Clone, Copy, Debug)]
pub struct ModelMatch<'a> {
    pub matched_key: &'a str,
    pub canonical_key: &'a str,
    pub entry: &'a ModelEntry,
}

/// A registry document read entry by entry. A bad entry keeps its raw fields and its problems are
/// listed, so the caller decides what to serve.
#[derive(Debug)]
pub struct Catalog {
    entries: IndexMap<String, ModelEntry>,
    aliases: IndexMap<String, String>,
    lowercase_keys: HashMap<String, String>,
    sample_spec: Option<Value>,
    fallback_generalizations: Option<FallbackGeneralizations>,
    problems: Vec<Problem>,
}

impl IntegrityLimits {
    pub fn new(min_model_count: usize, min_reference_ratio: f64) -> Option<Self> {
        (0.0..=1.0).contains(&min_reference_ratio).then_some(Self {
            min_model_count,
            min_reference_ratio,
        })
    }
}

impl ModelEntry {
    pub fn field(&self, name: &str) -> Option<&Value> {
        self.fields.get(name)
    }
    pub fn fields(&self) -> &Fields {
        &self.fields
    }
    pub fn model(&self) -> Option<&Model> {
        self.model.as_ref()
    }
}

impl Catalog {
    pub fn parse(body: &[u8]) -> Result<Self, InvalidDocument> {
        let mut document: IndexMap<String, Value> = serde_json::from_slice(body)?;
        let sample_spec = document.shift_remove("sample_spec");
        let fallback_generalizations = document
            .shift_remove("fallback_generalizations")
            .map(serde_json::from_value)
            .transpose()?;
        if document.is_empty() {
            return Err(InvalidDocument::Empty);
        }

        let mut problems = Vec::new();
        let mut entries = IndexMap::with_capacity(document.len());
        let mut alias_lists = Vec::new();
        for (name, value) in document {
            let Value::Object(fields) = value else {
                problems.push(Problem::NotAnObject { model: name });
                continue;
            };
            let wire = EntryWire::read(&fields);
            if let Some(aliases) = wire.attributes.aliases.clone() {
                alias_lists.push((name.clone(), aliases));
            }
            let model = Model::try_from(wire)
                .map_err(|found| {
                    problems.extend(found.into_iter().map(|problem| Problem::Entry {
                        model: name.clone(),
                        problem,
                    }))
                })
                .ok();
            problems.extend(
                model
                    .as_ref()
                    .and_then(|model| unknown_timezone(&name, model)),
            );
            entries.insert(name, ModelEntry { fields, model });
        }

        let mut aliases: IndexMap<String, String> = IndexMap::new();
        for (model, names) in alias_lists {
            for alias in names {
                if entries.contains_key(&alias) {
                    problems.push(Problem::AliasShadowsModel {
                        model: model.clone(),
                        alias,
                    });
                } else if let Some(taken_by) = aliases.get(&alias) {
                    problems.push(Problem::AliasTaken {
                        model: model.clone(),
                        taken_by: taken_by.clone(),
                        alias,
                    });
                } else {
                    aliases.insert(alias, model.clone());
                }
            }
        }

        let lowercase_keys = entries
            .keys()
            .chain(aliases.keys())
            .map(|key| (key.to_lowercase(), key.clone()))
            .collect();
        Ok(Self {
            entries,
            aliases,
            lowercase_keys,
            sample_spec,
            fallback_generalizations,
            problems,
        })
    }

    /// Whether this catalog is big enough to replace one of `reference` models.
    pub fn check_size(
        &self,
        reference: usize,
        limits: IntegrityLimits,
    ) -> Result<(), IntegrityViolation> {
        let actual = self.model_count();
        if actual < limits.min_model_count {
            return Err(IntegrityViolation::BelowMinimum {
                actual,
                minimum: limits.min_model_count,
            });
        }
        if (actual as f64) < (reference as f64) * limits.min_reference_ratio {
            return Err(IntegrityViolation::Shrunk {
                actual,
                reference,
                ratio: limits.min_reference_ratio,
            });
        }
        Ok(())
    }

    pub fn problems(&self) -> &[Problem] {
        &self.problems
    }

    /// Every `(model, key)` no type of this build reads.
    pub fn unrecognized(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.iter().flat_map(|(name, entry)| {
            entry
                .model
                .iter()
                .flat_map(|model| &model.unrecognized)
                .map(move |key| (name.as_str(), key.as_str()))
        })
    }

    pub fn lookup(&self, key: &str) -> Option<ModelMatch<'_>> {
        let matched_key = if self.entries.contains_key(key) || self.aliases.contains_key(key) {
            key
        } else {
            self.lowercase_keys.get(&key.to_lowercase())?.as_str()
        };
        let canonical_key = self
            .aliases
            .get(matched_key)
            .map(String::as_str)
            .unwrap_or(matched_key);
        let (canonical_key, entry) = self.entries.get_key_value(canonical_key)?;
        let matched_key = self
            .entries
            .get_key_value(matched_key)
            .map(|(key, _)| key.as_str())
            .or_else(|| {
                self.aliases
                    .get_key_value(matched_key)
                    .map(|(key, _)| key.as_str())
            })?;
        Some(ModelMatch {
            matched_key,
            canonical_key,
            entry,
        })
    }

    pub fn model_count(&self) -> usize {
        self.entries.len()
    }
    pub fn model_names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }
    pub fn aliases(&self) -> &IndexMap<String, String> {
        &self.aliases
    }
    pub fn sample_spec(&self) -> Option<&Value> {
        self.sample_spec.as_ref()
    }
    pub fn fallback_rules(&self) -> Option<&[FallbackRule]> {
        Some(self.fallback_generalizations.as_ref()?.rules.as_slice())
    }
}

fn unknown_timezone(name: &str, model: &Model) -> Option<Problem> {
    let zone = model
        .prices
        .off_peak()?
        .schedule()
        .weekday_timezone
        .as_deref()?
        .trim();
    (!zone.is_empty() && jiff::tz::TimeZone::get(zone).is_err()).then(|| Problem::UnknownTimezone {
        model: name.to_owned(),
        zone: zone.to_owned(),
    })
}
