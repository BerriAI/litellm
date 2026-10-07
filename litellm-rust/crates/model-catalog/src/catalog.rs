use crate::error::Error;
use crate::fallback::{FallbackGeneralizations, FallbackRule};
use crate::index::{ModelIndex, parse_index};
use crate::model_info::ModelInfo;
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Provenance {
    pub source: Option<String>,
    pub revision: Option<String>,
    pub etag: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntegrityLimits {
    pub reference_model_count: usize,
    pub min_model_count: usize,
    pub min_reference_ratio: f64,
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
    index: ModelIndex<ModelEntry>,
    sample_spec: Option<Value>,
    fallback_generalizations: Option<FallbackGeneralizations>,
    provenance: Provenance,
}

impl Catalog {
    pub fn parse(body: &[u8], provenance: Provenance) -> Result<Self, Error> {
        let parsed = parse_index(body, |fields| {
            let info = ModelInfo::deserialize(&fields)?;
            Ok(ModelEntry { fields, info })
        })?;
        Ok(Self {
            index: parsed.index,
            sample_spec: parsed.sample_spec,
            fallback_generalizations: parsed
                .fallback_generalizations
                .map(serde_json::from_value)
                .transpose()?,
            provenance,
        })
    }

    pub fn validate(&self, limits: IntegrityLimits) -> Result<(), Error> {
        if !limits.min_reference_ratio.is_finite()
            || !(0.0..=1.0).contains(&limits.min_reference_ratio)
        {
            return Err(Error::InvalidRatio);
        }
        let actual = self.index.entries.len();
        if actual < limits.min_model_count {
            return Err(Error::BelowMinimum {
                actual,
                minimum: limits.min_model_count,
            });
        }
        if limits.reference_model_count > 0
            && (actual as f64) < (limits.reference_model_count as f64) * limits.min_reference_ratio
        {
            return Err(Error::Shrunk {
                actual,
                reference: limits.reference_model_count,
                ratio: limits.min_reference_ratio,
            });
        }
        Ok(())
    }

    pub fn lookup(&self, key: &str) -> Option<ModelMatch<'_>> {
        let matched = self.index.lookup(key)?;
        Some(ModelMatch {
            matched_key: matched.matched_key,
            canonical_key: matched.canonical_key,
            entry: matched.entry,
        })
    }

    pub fn model_count(&self) -> usize {
        self.index.entries.len()
    }
    pub fn model_names(&self) -> impl Iterator<Item = &str> {
        self.index.entries.keys().map(String::as_str)
    }
    pub fn alias_count(&self) -> usize {
        self.index.aliases.len()
    }
    pub fn aliases(&self) -> &IndexMap<String, String> {
        &self.index.aliases
    }
    pub fn alias_issues(&self) -> &[AliasIssue] {
        &self.index.alias_issues
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
