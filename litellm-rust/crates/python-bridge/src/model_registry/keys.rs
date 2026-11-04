use std::borrow::Cow;
use std::sync::LazyLock;

use litellm_model_catalog::{Rows, Store};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use super::live::current;

/// The bound `litellm.model_cost`, read again on every query because a reload rebinds it
struct LiveModelCost;

impl Rows for LiveModelCost {
    type Row = ();
    /// The dict object and its length, which miss a delete plus an add between two reads
    type Stamp = Option<(usize, usize)>;

    fn fold(name: &str) -> String {
        name.to_lowercase()
    }

    fn stamp(&self) -> Self::Stamp {
        Python::attach(|py| {
            current(py)
                .ok()
                .map(|dict| (dict.as_ptr() as usize, dict.len()))
        })
    }

    fn contains(&self, key: &str) -> bool {
        Python::attach(|py| current(py).is_ok_and(|dict| dict.contains(key).unwrap_or(false)))
    }

    fn row(&self, key: &str) -> Option<Cow<'_, ()>> {
        self.contains(key).then_some(Cow::Owned(()))
    }

    /// A key that is not a `str` cannot be matched by name and is left out
    fn scan(&self, visit: &mut dyn FnMut(&str, Option<&str>)) {
        Python::attach(|py| {
            let Ok(dict) = current(py) else {
                return;
            };
            for (key, row) in dict
                .copy()
                .map(|copy| copy.into_iter().collect::<Vec<_>>())
                .unwrap_or_default()
            {
                let Some(key) = key
                    .cast::<PyString>()
                    .ok()
                    .and_then(|key| key.to_cow().ok())
                else {
                    continue;
                };
                let provider = provider(&row);
                visit(&key, provider.as_deref());
            }
        });
    }
}

fn provider(row: &Bound<'_, PyAny>) -> Option<String> {
    let row = row.cast::<PyDict>().ok()?;
    let provider = row.get_item(intern!(row.py(), "litellm_provider")).ok()??;
    provider.extract().ok()
}

static STORE: LazyLock<Store<LiveModelCost>> = LazyLock::new(|| Store::new(LiveModelCost));

/// The `litellm.model_cost` key for `name`: `name` itself when present, else the key that lowercases to the same
/// name, the later key winning a tie as main's `_model_cost_lowercase_map` did
#[pyfunction]
pub(crate) fn find_model_cost_key(name: &str) -> Option<String> {
    STORE.find_key(name)
}

/// For a write the `(id, len)` stamp cannot see, such as a key replaced by another of the same count
#[pyfunction]
pub(crate) fn invalidate_model_cost_keys() {
    STORE.invalidate();
}
