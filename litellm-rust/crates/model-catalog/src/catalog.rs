use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::Value;

use crate::error::{Error, Loaded, Located};
use crate::fallback::{FallbackGeneralizations, FallbackRule};
use crate::lower::rejects_row;
use crate::model::Model;
use crate::wire::{EntryWire, Fields};

const SAMPLE_SPEC: &str = "sample_spec";
const FALLBACK_GENERALIZATIONS: &str = "fallback_generalizations";

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

#[derive(Clone, Debug)]
struct Section {
    index: usize,
    key: String,
    value: Value,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(untagged)]
pub enum Written<'a> {
    Entry(&'a Fields),
    Section(&'a Value),
}

#[derive(Clone, Debug)]
pub struct Catalog {
    entries: IndexMap<String, ModelEntry>,
    sections: Vec<Section>,
    aliases: IndexMap<String, String>,
    lowercase_keys: HashMap<String, String>,
    fallback_generalizations: Option<FallbackGeneralizations>,
}

impl IntegrityLimits {
    pub const DEFAULT: Self = Self {
        min_model_count: 50,
        min_reference_ratio: 0.5,
    };

    pub fn new(min_model_count: usize, min_reference_ratio: f64) -> Option<Self> {
        (0.0..=1.0).contains(&min_reference_ratio).then_some(Self {
            min_model_count,
            min_reference_ratio,
        })
    }

    pub fn check(self, actual: usize, reference: usize) -> Result<(), Located> {
        if actual < self.min_model_count {
            return Err(Located::new(
                "",
                Error::BelowMinimum {
                    actual,
                    minimum: self.min_model_count,
                },
            ));
        }
        if (actual as f64) < (reference as f64) * self.min_reference_ratio {
            return Err(Located::new(
                "",
                Error::Shrunk {
                    actual,
                    reference,
                    ratio: self.min_reference_ratio,
                },
            ));
        }
        Ok(())
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
    pub fn parse(body: &[u8]) -> Result<Loaded<Self>, Located> {
        let document = serde_json::from_slice::<Fields>(body)
            .map_err(|error| Located::new("", Error::Json(error)))?;

        let mut problems = Vec::new();
        let mut sections = Vec::new();
        let mut entries = IndexMap::with_capacity(document.len());
        let mut alias_lists = Vec::new();
        for (index, (name, value)) in document.into_iter().enumerate() {
            let fields = match value {
                Value::Object(fields) if !is_section(&name) => fields,
                value => {
                    if !is_section(&name) {
                        problems.push(Located::from_key(&name, Error::NotAnObject));
                    }
                    sections.push(Section {
                        index,
                        key: name,
                        value,
                    });
                    continue;
                }
            };
            let (model, aliases, mut row_problems) = read_row(&fields);
            row_problems
                .iter_mut()
                .for_each(|problem| problem.prepend(&name));
            problems.extend(row_problems);
            if let Some(aliases) = aliases {
                alias_lists.push((name.clone(), aliases));
            }
            entries.insert(name, ModelEntry { fields, model });
        }
        if entries.is_empty() {
            return Err(Located::new("", Error::Empty));
        }

        let mut aliases: IndexMap<String, String> = IndexMap::new();
        for (model, names) in alias_lists {
            for alias in names {
                if entries.contains_key(&alias) {
                    problems.push(Located::from_key(
                        &model,
                        Error::AliasShadowsModel { alias },
                    ));
                } else if let Some(taken_by) = aliases.get(&alias) {
                    problems.push(Located::from_key(
                        &model,
                        Error::AliasTaken {
                            alias,
                            taken_by: taken_by.clone(),
                        },
                    ));
                } else {
                    aliases.insert(alias, model.clone());
                }
            }
        }

        let fallback_generalizations = sections
            .iter()
            .find(|section| section.key == FALLBACK_GENERALIZATIONS)
            .and_then(|section| {
                FallbackGeneralizations::deserialize(&section.value)
                    .map_err(|error| {
                        problems.push(Located::from_key(&section.key, Error::Invalid(error)))
                    })
                    .ok()
            });
        let lowercase_keys = entries
            .keys()
            .chain(aliases.keys())
            .map(|key| (key.to_lowercase(), key.clone()))
            .collect();
        Ok(Loaded {
            value: Self {
                entries,
                sections,
                aliases,
                lowercase_keys,
                fallback_generalizations,
            },
            errors: problems,
        })
    }

    pub fn row_errors(model: &str, fields: &Fields) -> Vec<Located> {
        let (_, _, mut errors) = read_row(fields);
        errors.iter_mut().for_each(|error| error.prepend(model));
        errors
    }

    pub fn written(&self) -> impl Iterator<Item = (&str, Written<'_>)> {
        let mut sections = self.sections.iter().peekable();
        let mut entries = self.entries.iter();
        (0..self.entries.len() + self.sections.len()).map_while(move |index| {
            match sections.next_if(|section| section.index == index) {
                Some(section) => Some((section.key.as_str(), Written::Section(&section.value))),
                None => entries
                    .next()
                    .map(|(name, entry)| (name.as_str(), Written::Entry(&entry.fields))),
            }
        })
    }

    pub fn served(&self) -> impl Iterator<Item = (&str, &Fields)> {
        self.entries
            .iter()
            .map(|(name, entry)| (name.as_str(), entry.fields()))
            .chain(
                self.aliases
                    .iter()
                    .map(|(alias, owner)| (alias.as_str(), self.entries[owner].fields())),
            )
    }

    pub fn check_size(&self, reference: usize, limits: IntegrityLimits) -> Result<(), Located> {
        limits.check(self.model_count(), reference)
    }

    pub fn model_count_without_rejected(&self, errors: &[Located]) -> usize {
        let rejected = self.rejected_rows(errors);
        self.entries
            .keys()
            .filter(|name| !rejected.contains(name.as_str()))
            .count()
    }

    fn rejected_rows(&self, errors: &[Located]) -> HashSet<String> {
        errors
            .iter()
            .filter(|error| rejects_row(&error.error))
            .filter_map(|error| crate::error::pointer_root(&error.path))
            .filter(|root| self.entries.contains_key(root) || self.has_bad_row(root))
            .collect()
    }

    fn has_bad_row(&self, key: &str) -> bool {
        !is_section(key) && self.sections.iter().any(|section| section.key == key)
    }

    pub fn without_rejected(self, errors: &[Located]) -> (Self, Vec<&Located>) {
        let rejected = self.rejected_rows(errors);
        let removed_errors = errors
            .iter()
            .filter(|error| {
                crate::error::pointer_root(&error.path).is_some_and(|root| rejected.contains(&root))
            })
            .collect();
        let removed_positions: HashSet<usize> = self
            .written()
            .enumerate()
            .filter(|(_, (key, _))| rejected.contains(*key))
            .map(|(index, _)| index)
            .collect();
        let mut catalog = self;
        catalog.entries.retain(|name, _| !rejected.contains(name));
        catalog
            .sections
            .retain(|section| is_section(&section.key) || !rejected.contains(&section.key));
        catalog.aliases.retain(|_, owner| !rejected.contains(owner));
        for section in &mut catalog.sections {
            let removed_before = removed_positions
                .iter()
                .filter(|index| **index < section.index)
                .count();
            section.index -= removed_before;
        }
        catalog.lowercase_keys = catalog
            .entries
            .keys()
            .chain(catalog.aliases.keys())
            .map(|key| (key.to_lowercase(), key.clone()))
            .collect();
        (catalog, removed_errors)
    }

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
        self.sections
            .iter()
            .find(|section| section.key == SAMPLE_SPEC)
            .map(|section| &section.value)
    }

    pub fn fallback_rules(&self) -> Option<&[FallbackRule]> {
        Some(self.fallback_generalizations.as_ref()?.rules.as_slice())
    }
}

fn read_row(fields: &Fields) -> (Option<Model>, Option<Vec<String>>, Vec<Located>) {
    let wire = EntryWire::read(fields);
    let aliases = wire.attributes.aliases.clone();
    let timezone_error = unknown_timezone(&wire);
    match Model::try_from(wire) {
        Ok(model) => (Some(model), aliases, timezone_error.into_iter().collect()),
        Err(mut errors) => {
            errors.extend(timezone_error);
            (None, aliases, errors)
        }
    }
}

fn is_section(key: &str) -> bool {
    key == SAMPLE_SPEC || key == FALLBACK_GENERALIZATIONS
}

fn unknown_timezone(wire: &EntryWire) -> Option<Located> {
    let zone = wire.off_peak.as_ref()?.weekday_timezone.as_deref()?.trim();
    (!zone.is_empty() && jiff::tz::TimeZone::get(zone).is_err()).then(|| {
        Located::from_path(
            "off_peak_pricing.weekday_timezone",
            Error::UnknownTimezone {
                zone: zone.to_owned(),
            },
        )
    })
}
