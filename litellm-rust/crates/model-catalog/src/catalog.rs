use indexmap::IndexMap;
use serde_json::{Map, Value};
use std::collections::HashMap;

use crate::error::{Error, Problem, Rejected};
use crate::fallback::{FallbackGeneralizations, FallbackRule};
use crate::model_info::{Fields, Model};

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Provenance {
    pub source: Option<String>,
    pub revision: Option<String>,
    pub etag: Option<String>,
}

/// How far a fetched registry may shrink against the one it replaces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntegrityLimits {
    pub min_model_count: usize,
    pub min_reference_ratio: f64,
}

#[derive(Clone, Debug)]
pub struct ModelEntry {
    fields: Fields,
    model: Model,
}

impl ModelEntry {
    pub fn field(&self, name: &str) -> Option<&Value> {
        self.fields.get(name)
    }
    pub fn fields(&self) -> &Fields {
        &self.fields
    }
    pub fn model(&self) -> &Model {
        &self.model
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ModelMatch<'a> {
    pub matched_key: &'a str,
    pub canonical_key: &'a str,
    pub entry: &'a ModelEntry,
}

/// The registry document: named sections, and every other key a model.
#[macro_rules_attribute::apply(crate::wire_type)]
pub(crate) struct Registry {
    sample_spec: Option<Map<String, Value>>,
    fallback_generalizations: Option<FallbackGeneralizations>,
    #[serde(flatten)]
    #[cfg_attr(
        feature = "schema",
        schemars(with = "std::collections::BTreeMap<String, Value>")
    )]
    models: IndexMap<String, Value>,
}

#[derive(Debug)]
pub struct Catalog {
    entries: IndexMap<String, ModelEntry>,
    aliases: IndexMap<String, String>,
    lowercase_keys: HashMap<String, String>,
    sample_spec: Option<Map<String, Value>>,
    fallback_generalizations: Option<FallbackGeneralizations>,
    provenance: Provenance,
}

impl Catalog {
    /// Reads a registry document, rejecting it with every problem found. Keys no type knows are
    /// kept and listed by `unrecognized`.
    pub fn parse(body: &[u8], provenance: Provenance) -> Result<Self, Error> {
        let Registry {
            sample_spec,
            fallback_generalizations,
            models,
        } = serde_json::from_slice(body)?;
        if models.is_empty() {
            return Err(Error::Empty);
        }

        let mut problems = Vec::new();
        let mut entries = IndexMap::with_capacity(models.len());
        let mut alias_lists = Vec::new();
        for (name, value) in models {
            let Value::Object(mut fields) = value else {
                problems.push(Problem::NotAnObject { model: name });
                continue;
            };
            match fields.remove("aliases").map(alias_names) {
                None => {}
                Some(Some(names)) => alias_lists.push((name.clone(), names)),
                Some(None) => problems.push(Problem::InvalidAliases {
                    model: name.clone(),
                }),
            }
            match Model::from_fields(&name, &fields) {
                Ok(model) => {
                    if model.attributes.litellm_provider.is_none() {
                        problems.push(Problem::MissingProvider {
                            model: name.clone(),
                        });
                    }
                    entries.insert(name, ModelEntry { fields, model });
                }
                Err(found) => problems.extend(found),
            }
        }

        let mut aliases: IndexMap<String, String> = IndexMap::new();
        for (model, names) in alias_lists {
            for alias in names {
                if entries.contains_key(&alias) {
                    problems.push(Problem::AliasShadowsModel {
                        model: model.clone(),
                        alias,
                    });
                } else if let Some(owner) = aliases.get(&alias) {
                    problems.push(Problem::AliasTaken {
                        model: model.clone(),
                        owner: owner.clone(),
                        alias,
                    });
                } else {
                    aliases.insert(alias, model.clone());
                }
            }
        }
        if !problems.is_empty() {
            return Err(Rejected(problems).into());
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
            provenance,
        })
    }

    /// `body` read as the replacement for `current`, rejected when it does not parse or has
    /// shrunk past `limits`.
    pub fn fetched(
        body: &[u8],
        provenance: Provenance,
        current: &Catalog,
        limits: IntegrityLimits,
    ) -> Result<Self, Error> {
        let fetched = Self::parse(body, provenance)?;
        fetched.check_size(current.model_count(), limits)?;
        Ok(fetched)
    }

    /// Whether this catalog is big enough to replace one of `reference` models.
    pub fn check_size(&self, reference: usize, limits: IntegrityLimits) -> Result<(), Error> {
        if !limits.min_reference_ratio.is_finite()
            || !(0.0..=1.0).contains(&limits.min_reference_ratio)
        {
            return Err(Error::InvalidRatio);
        }
        let actual = self.model_count();
        if actual < limits.min_model_count {
            return Err(Error::BelowMinimum {
                actual,
                minimum: limits.min_model_count,
            });
        }
        if (actual as f64) < (reference as f64) * limits.min_reference_ratio {
            return Err(Error::Shrunk {
                actual,
                reference,
                ratio: limits.min_reference_ratio,
            });
        }
        Ok(())
    }

    /// Every `(model, key)` no type of this build reads. Python still serves these keys.
    pub fn unrecognized(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.iter().flat_map(|(name, entry)| {
            entry
                .model
                .unrecognized
                .iter()
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
    pub fn alias_count(&self) -> usize {
        self.aliases.len()
    }
    pub fn aliases(&self) -> &IndexMap<String, String> {
        &self.aliases
    }
    pub fn sample_spec(&self) -> Option<&Map<String, Value>> {
        self.sample_spec.as_ref()
    }
    pub fn fallback_generalizations(&self) -> Option<&FallbackGeneralizations> {
        self.fallback_generalizations.as_ref()
    }
    pub fn fallback_rules(&self) -> Option<&[FallbackRule]> {
        Some(self.fallback_generalizations.as_ref()?.rules.as_slice())
    }
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
}

fn alias_names(aliases: Value) -> Option<Vec<String>> {
    let Value::Array(names) = aliases else {
        return None;
    };
    names
        .into_iter()
        .map(|name| match name {
            Value::String(name) => Some(name),
            _ => None,
        })
        .collect()
}
