use std::collections::HashMap;
use std::sync::Arc;

use indexmap::IndexMap;
use serde_json::Value;

use crate::catalog::Catalog;
use crate::layers::Layers;
use crate::wire::Fields;

/// What the bound dict holds for one field, as litellm compares it
#[derive(Clone, Debug, PartialEq)]
pub enum Held {
    Absent,
    Json(Value),
    /// A value that does not convert to JSON, which only the user can have written
    Foreign,
}

/// What the bound dict holds under one key
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    Missing,
    /// Anything but a dict, which only the user can have written
    NotARow,
    Row,
}

/// The dict install writes into, `litellm.model_cost` in the Python host
pub trait BoundMap {
    type Error;

    fn slot(&self, key: &str) -> Result<Slot, Self::Error>;
    /// Called only for a key whose slot is `Row`
    fn held(&self, key: &str, field: &str) -> Result<Held, Self::Error>;
    /// The row's string-named fields, called only for a key whose slot is `Row`
    fn fields(&self, key: &str) -> Result<Vec<(String, Held)>, Self::Error>;
    fn row_is_empty(&self, key: &str) -> Result<bool, Self::Error>;
    fn set_field(&mut self, key: &str, field: &str, value: &Value) -> Result<(), Self::Error>;
    fn del_field(&mut self, key: &str, field: &str) -> Result<(), Self::Error>;
    /// A new row under `key`, the same object for every key given the same `group` in one install
    fn insert_row(&mut self, key: &str, group: &str, fields: &Fields) -> Result<(), Self::Error>;
    fn del_key(&mut self, key: &str) -> Result<(), Self::Error>;
    /// Deletes every key, string or not, that `keep` refuses, and returns how many
    fn retain_keys(&mut self, keep: &dyn Fn(&str) -> bool) -> Result<usize, Self::Error>;
}

/// The field values litellm last wrote into one dict object
#[derive(Clone, Debug)]
pub struct Record {
    dict_id: usize,
    rows: HashMap<String, Fields>,
}

#[derive(Debug)]
pub struct Outcome {
    pub record: Record,
    /// Keys set because the row differed or was new
    pub written: usize,
    /// Keys deleted because the map no longer serves them
    pub removed: usize,
}

/// The layers and what the last install wrote, kept between installs
#[derive(Debug)]
pub struct Installs {
    layers: Layers,
    record: Option<Record>,
}

/// One install between `Installs::begin` and `Installs::finish`
#[derive(Debug)]
pub struct Pending {
    layers: Layers,
    previous: Option<Record>,
    dict_id: usize,
}

impl Installs {
    pub fn new(bundled: Arc<Catalog>) -> Self {
        Self {
            layers: Layers::new(bundled),
            record: None,
        }
    }

    /// Replaces the fetched layer. The last record applies only while `dict_id` names the dict it was written into
    pub fn begin(&mut self, fetched: Arc<Catalog>, dict_id: usize) -> Pending {
        self.layers.replace_fetched(fetched);
        Pending {
            layers: self.layers.clone(),
            previous: self
                .record
                .clone()
                .filter(|record| record.dict_id == dict_id),
            dict_id,
        }
    }

    pub fn finish(&mut self, record: Record) {
        self.record = Some(record);
    }
}

impl Pending {
    pub fn compose(&self) -> IndexMap<String, Fields> {
        self.layers.compose()
    }

    /// A field is written only while litellm still owns it, meaning the dict holds the value last written there, or
    /// nothing where nothing was written. Without a record for this dict the dict is written in full, which is main's
    /// result after rebind then adopt
    pub fn write<M: BoundMap>(
        self,
        map: &mut M,
        composed: &IndexMap<String, Fields>,
    ) -> Result<Outcome, M::Error> {
        match self.previous {
            Some(previous) => write_owned(map, composed, &self.layers, previous),
            None => write_in_full(map, composed, &self.layers, self.dict_id),
        }
    }
}

fn write_owned<M: BoundMap>(
    map: &mut M,
    composed: &IndexMap<String, Fields>,
    layers: &Layers,
    previous: Record,
) -> Result<Outcome, M::Error> {
    let mut rows = previous.rows;
    let mut written = 0;
    let mut removed = 0;
    let keys: Vec<String> = composed
        .keys()
        .chain(rows.keys().filter(|key| !composed.contains_key(*key)))
        .cloned()
        .collect();
    for key in keys {
        let target = composed.get(&key);
        let last = rows.remove(&key);
        match map.slot(&key)? {
            Slot::Missing => match (target, last) {
                (Some(target), None) => {
                    map.insert_row(&key, group(layers, &key), target)?;
                    rows.insert(key, target.clone());
                    written += 1;
                }
                (Some(_), Some(last)) => {
                    rows.insert(key, last);
                }
                (None, _) => {}
            },
            Slot::NotARow => {
                if let Some(last) = last {
                    rows.insert(key, last);
                }
            }
            Slot::Row => {
                let last = last.unwrap_or_default();
                let (owned, changed) = write_owned_fields(map, &key, target, &last)?;
                if target.is_none() && changed && map.row_is_empty(&key)? {
                    map.del_key(&key)?;
                    removed += 1;
                    continue;
                }
                written += usize::from(changed);
                if !owned.is_empty() {
                    rows.insert(key, owned);
                }
            }
        }
    }
    Ok(Outcome {
        record: Record {
            dict_id: previous.dict_id,
            rows,
        },
        written,
        removed,
    })
}

fn write_in_full<M: BoundMap>(
    map: &mut M,
    composed: &IndexMap<String, Fields>,
    layers: &Layers,
    dict_id: usize,
) -> Result<Outcome, M::Error> {
    let removed = map.retain_keys(&|key| composed.contains_key(key))?;
    let mut written = 0;
    for (key, target) in composed {
        if map.slot(key)? != Slot::Row {
            map.insert_row(key, group(layers, key), target)?;
            written += 1;
            continue;
        }
        let mut current = Fields::new();
        for (field, held) in map.fields(key)? {
            match held {
                Held::Json(value) => {
                    current.insert(field, value);
                }
                Held::Absent | Held::Foreign => map.del_field(key, &field)?,
            }
        }
        let (_, changed) = write_owned_fields(map, key, Some(target), &current)?;
        written += usize::from(changed);
    }
    Ok(Outcome {
        record: Record {
            dict_id,
            rows: composed
                .iter()
                .map(|(key, fields)| (key.clone(), fields.clone()))
                .collect(),
        },
        written,
        removed,
    })
}

/// Sets or removes each field litellm owns, and returns the owned fields as now written and whether any changed
fn write_owned_fields<M: BoundMap>(
    map: &mut M,
    key: &str,
    target: Option<&Fields>,
    last: &Fields,
) -> Result<(Fields, bool), M::Error> {
    let empty = Fields::new();
    let target = target.unwrap_or(&empty);
    let fields: Vec<&String> = target
        .keys()
        .chain(last.keys().filter(|field| !target.contains_key(*field)))
        .collect();
    let mut owned = Fields::new();
    let mut changed = false;
    for field in fields {
        let held = map.held(key, field)?;
        let litellm_owns = match (&held, last.get(field)) {
            (Held::Absent, None) => true,
            (Held::Json(value), Some(last)) => value == last,
            _ => false,
        };
        if !litellm_owns {
            if let Some(last) = last.get(field) {
                owned.insert(field.clone(), last.clone());
            }
            continue;
        }
        match target.get(field) {
            Some(value) => {
                if held != Held::Json(value.clone()) {
                    map.set_field(key, field, value)?;
                    changed = true;
                }
                owned.insert(field.clone(), value.clone());
            }
            None if held != Held::Absent => {
                map.del_field(key, field)?;
                changed = true;
            }
            None => {}
        }
    }
    Ok((owned, changed))
}

fn group<'a>(layers: &'a Layers, key: &'a str) -> &'a str {
    layers.alias_owner(key).unwrap_or(key)
}
