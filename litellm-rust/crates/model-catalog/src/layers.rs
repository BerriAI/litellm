use std::sync::Arc;

use indexmap::IndexMap;
use serde_json::Value;

use crate::catalog::Catalog;
use crate::register::Registrations;
use crate::wire::Fields;

const ALIASES: &str = "aliases";

/// One field of a fetched row over the layer below, RFC 7386. An absent field inherits, JSON `null` is `Remove`
#[derive(Clone, Debug, PartialEq)]
pub enum Patch {
    Set(Value),
    Remove,
}

/// The four layers served rows are composed from, lowest first
#[derive(Clone, Debug)]
pub struct Layers {
    /// 0, the wheel's embedded map
    bundled: Arc<Catalog>,
    /// 1, the last installed fetch, rejected rows left out
    fetched: Option<Arc<Catalog>>,
    /// 2, `register(persist=False)`, cleared by every install and refilled by the router replay
    transient: IndexMap<String, Fields>,
    /// 3, `register(persist=True)`, applied over every install
    #[expect(dead_code, reason = "07 applies registrations over the fetched layer")]
    persisted: Registrations,
}

impl Patch {
    pub fn from_value(value: &Value) -> Self {
        match value {
            Value::Null => Self::Remove,
            value => Self::Set(value.clone()),
        }
    }
}

impl Layers {
    pub fn new(bundled: Arc<Catalog>) -> Self {
        Self {
            bundled,
            fetched: None,
            transient: IndexMap::new(),
            persisted: Registrations::default(),
        }
    }

    /// Install replaces layer 1 and clears layer 2, which the router replay refills
    pub fn replace_fetched(&mut self, fetched: Arc<Catalog>) {
        self.fetched = Some(fetched);
        self.transient.clear();
    }

    /// What install writes. A fetched row patches the bundled row field by field, a fetched `null` row removes it
    pub fn compose(&self) -> IndexMap<String, Fields> {
        let mut rows: IndexMap<String, Fields> = self
            .bundled
            .served()
            .map(|(key, fields)| (key.to_owned(), without_aliases(fields)))
            .collect();
        let Some(fetched) = &self.fetched else {
            return rows;
        };
        for key in fetched.removals() {
            rows.shift_remove(key);
        }
        for (key, fields) in fetched.served() {
            let row = rows.entry(key.to_owned()).or_default();
            for (field, value) in fields.iter().filter(|(field, _)| *field != ALIASES) {
                match Patch::from_value(value) {
                    Patch::Set(value) => {
                        row.insert(field.clone(), value);
                    }
                    Patch::Remove => {
                        row.shift_remove(field);
                    }
                }
            }
        }
        rows
    }

    /// The canonical key an alias is served under, from the fetched layer first
    pub fn alias_owner(&self, key: &str) -> Option<&str> {
        self.fetched
            .iter()
            .chain([&self.bundled])
            .find_map(|catalog| catalog.aliases().get(key))
            .map(String::as_str)
    }
}

fn without_aliases(fields: &Fields) -> Fields {
    fields
        .iter()
        .filter(|(field, value)| *field != ALIASES && !value.is_null())
        .map(|(field, value)| (field.clone(), value.clone()))
        .collect()
}
