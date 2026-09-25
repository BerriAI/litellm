use crate::error::Error;
use crate::model_info::{FallbackGeneralizations, FallbackRule, ModelInfo};
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::HashMap;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Provenance {
    pub source: Option<String>,
    pub revision: Option<String>,
    pub etag: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntegrityLimits {
    pub backup_model_count: usize,
    pub min_model_count: usize,
    pub min_backup_ratio: f64,
}

impl IntegrityLimits {
    pub fn python_defaults(backup_model_count: usize) -> Self {
        Self {
            backup_model_count,
            min_model_count: 50,
            min_backup_ratio: 0.5,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AliasIssue {
    InvalidList { model: String },
    InvalidName { model: String },
    CanonicalCollision { model: String, alias: String },
    AliasCollision { model: String, alias: String },
}

#[derive(Clone, Debug)]
pub struct ModelEntry {
    fields: Map<String, Value>,
    info: ModelInfo,
}

impl ModelEntry {
    pub fn field(&self, name: &str) -> Option<&Value> {
        self.fields.get(name)
    }
    pub fn fields(&self) -> &Map<String, Value> {
        &self.fields
    }
    /// The entry deserialized into the typed mirror of the catalog schema.
    pub fn info(&self) -> &ModelInfo {
        &self.info
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ModelMatch<'a> {
    pub matched_key: &'a str,
    pub canonical_key: &'a str,
    pub entry: &'a ModelEntry,
}

#[derive(Debug)]
pub struct Catalog {
    entries: IndexMap<String, ModelEntry>,
    aliases: IndexMap<String, String>,
    lowercase_keys: HashMap<String, String>,
    sample_spec: Option<Value>,
    fallback_generalizations: Option<FallbackGeneralizations>,
    provenance: Provenance,
    alias_issues: Vec<AliasIssue>,
}

impl Catalog {
    pub fn parse(body: &[u8], provenance: Provenance) -> Result<Self, Error> {
        let root: IndexMap<String, Value> = serde_json::from_slice(body)?;
        if root.is_empty() {
            return Err(Error::Empty);
        }

        let mut entries = IndexMap::with_capacity(root.len());
        let mut alias_lists = Vec::new();
        let mut alias_issues = Vec::new();
        let mut sample_spec = None;
        let mut fallback_generalizations = None;
        for (name, value) in root {
            match name.as_str() {
                "sample_spec" => {
                    sample_spec = Some(value);
                    continue;
                }
                "fallback_generalizations" => {
                    fallback_generalizations =
                        Some(serde_json::from_value::<FallbackGeneralizations>(value)?);
                    continue;
                }
                _ => {}
            }
            let Value::Object(ref object) = value else {
                return Err(Error::EntryNotObject { model: name });
            };
            let info = ModelInfo::deserialize(object)?;
            let Value::Object(mut fields) = value else {
                unreachable!("value checked is_object above")
            };
            if let Some(aliases) = fields.remove("aliases")
                && !aliases.is_null()
            {
                match aliases {
                    Value::Array(names) => alias_lists.push((name.clone(), names)),
                    _ => alias_issues.push(AliasIssue::InvalidList {
                        model: name.clone(),
                    }),
                }
            }
            entries.insert(name, ModelEntry { fields, info });
        }

        let mut aliases = IndexMap::new();
        for (model, names) in alias_lists {
            for name in names {
                let Value::String(alias) = name else {
                    alias_issues.push(AliasIssue::InvalidName {
                        model: model.clone(),
                    });
                    continue;
                };
                if entries.contains_key(&alias) {
                    alias_issues.push(AliasIssue::CanonicalCollision {
                        model: model.clone(),
                        alias,
                    });
                } else if aliases.contains_key(&alias) {
                    alias_issues.push(AliasIssue::AliasCollision {
                        model: model.clone(),
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
            provenance,
            alias_issues,
        })
    }

    pub fn validate(&self, limits: IntegrityLimits) -> Result<(), Error> {
        if !limits.min_backup_ratio.is_finite() || !(0.0..=1.0).contains(&limits.min_backup_ratio) {
            return Err(Error::InvalidRatio);
        }
        let actual = self.entries.len();
        if actual < limits.min_model_count {
            return Err(Error::BelowMinimum {
                actual,
                minimum: limits.min_model_count,
            });
        }
        if limits.backup_model_count > 0
            && (actual as f64) < (limits.backup_model_count as f64) * limits.min_backup_ratio
        {
            return Err(Error::Shrunk {
                actual,
                backup: limits.backup_model_count,
                ratio: limits.min_backup_ratio,
            });
        }
        Ok(())
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
    pub fn alias_issues(&self) -> &[AliasIssue] {
        &self.alias_issues
    }
    pub fn sample_spec(&self) -> Option<&Value> {
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
