use std::borrow::Cow;
use std::collections::BTreeSet;

use litellm_host_python::json_from_py;
use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{Rows, TableRow};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyMapping, PyString};

/// The dict Python last handed in as `litellm.model_cost`, read under the GIL on every query
pub(crate) struct LiveDict(Py<PyDict>);

/// One row exactly as the dict holds it
pub(crate) struct LiveRow(Py<PyAny>);

impl Clone for LiveRow {
    fn clone(&self) -> Self {
        Python::attach(|py| Self(self.0.clone_ref(py)))
    }
}

/// The row object's address is its identity. A snapshot's cache holds the row, so the address is not reused
impl TableRow for LiveRow {
    type Identity = usize;

    fn identity(&self) -> usize {
        self.0.as_ptr() as usize
    }

    /// A row that is not a mapping reads as empty
    fn fields(&self) -> Fields {
        Python::attach(|py| {
            self.0
                .bind(py)
                .cast::<PyMapping>()
                .ok()
                .and_then(|row| row_fields(row).ok())
                .unwrap_or_default()
        })
    }

    /// `AliasedCostEntry.registered_fields`
    fn registered_fields(&self) -> BTreeSet<String> {
        Python::attach(|py| {
            self.0
                .bind(py)
                .getattr(intern!(py, "registered_fields"))
                .and_then(|fields| fields.extract())
                .unwrap_or_default()
        })
    }
}

/// A row's string keys and the values JSON can hold. Other keys are left out, as values main could not price
pub(crate) fn row_fields(row: &Bound<'_, PyMapping>) -> PyResult<Fields> {
    Ok(row
        .items()?
        .iter()
        .filter_map(|item| {
            let (key, value) = item
                .extract::<(Bound<'_, PyString>, Bound<'_, PyAny>)>()
                .ok()?;
            Some((key.to_str().ok()?.to_owned(), json_from_py(&value).ok()?))
        })
        .collect())
}

impl LiveDict {
    pub(crate) fn new(dict: &Bound<'_, PyDict>) -> Self {
        Self(dict.clone().unbind())
    }

    pub(crate) fn is(&self, dict: &Bound<'_, PyDict>) -> bool {
        self.0.bind(dict.py()).is(dict)
    }
}

impl Rows for LiveDict {
    type Row = LiveRow;
    /// The dict's address, length and last key
    type Stamp = (usize, usize, Option<String>);

    fn fold(name: &str) -> String {
        name.to_lowercase()
    }

    fn stamp(&self) -> Self::Stamp {
        Python::attach(|py| {
            let rows = self.0.bind(py);
            let last = rows
                .call_method0(intern!(py, "__reversed__"))
                .and_then(|keys| keys.try_iter())
                .ok()
                .and_then(|mut keys| keys.next())
                .and_then(Result::ok)
                .and_then(|key| key.extract::<String>().ok());
            (self.0.as_ptr() as usize, rows.len(), last)
        })
    }

    fn contains(&self, key: &str) -> bool {
        Python::attach(|py| self.0.bind(py).contains(key).unwrap_or(false))
    }

    fn row(&self, key: &str) -> Option<Cow<'_, LiveRow>> {
        Python::attach(|py| {
            let row = self.0.bind(py).get_item(key).ok().flatten()?;
            Some(Cow::Owned(LiveRow(row.unbind())))
        })
    }

    /// Groups each key by its row's `litellm_provider` when that is a string
    fn scan(&self, visit: &mut dyn FnMut(&str, Option<&str>)) {
        Python::attach(|py| {
            for (key, row) in self.0.bind(py).iter() {
                let Ok(key) = key.extract::<String>() else {
                    continue;
                };
                let provider = row
                    .cast::<PyDict>()
                    .ok()
                    .and_then(|row| row.get_item("litellm_provider").ok().flatten())
                    .and_then(|provider| provider.extract::<String>().ok());
                visit(&key, provider.as_deref());
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use litellm_model_catalog::Store;
    use pyo3::ffi::c_str;
    use rstest::rstest;

    use super::*;

    fn dict<'py>(py: Python<'py>, source: &std::ffi::CStr) -> Bound<'py, PyDict> {
        py.eval(source, None, None)
            .unwrap()
            .cast_into::<PyDict>()
            .unwrap()
    }

    #[rstest]
    fn a_key_added_to_the_dict_is_found_on_the_next_query() {
        Python::initialize();
        Python::attach(|py| {
            let rows = dict(py, c_str!("{'gpt-4o': {'litellm_provider': 'openai'}}"));
            let store = Store::new(LiveDict::new(&rows));
            assert_eq!(store.find_key("GPT-4O"), Some("gpt-4o".to_owned()));
            rows.set_item("Claude-X", PyDict::new(py)).unwrap();
            assert_eq!(store.find_key("claude-x"), Some("Claude-X".to_owned()));
        });
    }

    #[rstest]
    fn a_removed_key_is_never_returned() {
        Python::initialize();
        Python::attach(|py| {
            let rows = dict(
                py,
                c_str!("{'gpt-4o': {'litellm_provider': 'openai'}, 'o3': {}}"),
            );
            let store = Store::new(LiveDict::new(&rows));
            assert_eq!(store.keys_in("openai"), ["gpt-4o"]);
            rows.del_item("gpt-4o").unwrap();
            rows.set_item("o4", PyDict::new(py)).unwrap();
            assert_eq!(store.find_key("GPT-4O"), None);
            assert!(store.keys_in("openai").is_empty());
        });
    }

    #[rstest]
    fn a_key_swapped_in_for_a_removed_one_is_found() {
        Python::initialize();
        Python::attach(|py| {
            let rows = dict(py, c_str!("{'gpt-4o': {}, 'Lit-Gone': {}}"));
            let store = Store::new(LiveDict::new(&rows));
            assert_eq!(store.find_key("lit-gone"), Some("Lit-Gone".to_owned()));
            rows.del_item("Lit-Gone").unwrap();
            rows.set_item("Lit-New", PyDict::new(py)).unwrap();
            assert_eq!(store.find_key("lit-new"), Some("Lit-New".to_owned()));
            assert_eq!(store.find_key("lit-gone"), None);
        });
    }

    #[rstest]
    fn two_keys_folding_alike_resolve_to_the_later_one() {
        Python::initialize();
        Python::attach(|py| {
            let rows = dict(py, c_str!("{'Model': {}, 'MODEL': {}}"));
            assert_eq!(
                Store::new(LiveDict::new(&rows)).find_key("model"),
                Some("MODEL".to_owned())
            );
        });
    }

    #[rstest]
    fn rows_are_the_objects_the_dict_holds() {
        Python::initialize();
        Python::attach(|py| {
            let rows = dict(py, c_str!("{'gpt-4o': {'litellm_provider': 'openai'}}"));
            let store = Store::new(LiveDict::new(&rows));
            let row = store.get("gpt-4o").unwrap().into_owned();
            assert!(row.0.bind(py).is(rows.get_item("gpt-4o").unwrap().unwrap()));
        });
    }

    #[rstest]
    fn providers_group_only_string_providers() {
        Python::initialize();
        Python::attach(|py| {
            let rows = dict(
                py,
                c_str!(
                    "{'a': {'litellm_provider': 'p'}, 'b': {'litellm_provider': None}, 'c': 1, 1: {'litellm_provider': 'p'}}"
                ),
            );
            assert_eq!(Store::new(LiveDict::new(&rows)).keys_in("p"), ["a"]);
        });
    }

    #[rstest]
    #[case::registered_row(
        c_str!("type('Row', (dict,), {'registered_fields': frozenset({'input_cost_per_token'})})(input_cost_per_token=1e-6, mode='chat')"),
        &["input_cost_per_token"]
    )]
    #[case::installed_row(c_str!("{'input_cost_per_token': 1e-6, 'mode': 'chat'}"), &[])]
    fn registered_fields_are_the_ones_register_model_wrote(
        #[case] source: &std::ffi::CStr,
        #[case] expected: &[&str],
    ) {
        Python::initialize();
        Python::attach(|py| {
            let row = LiveRow(py.eval(source, None, None).unwrap().unbind());

            assert_eq!(
                row.registered_fields(),
                expected.iter().map(|field| (*field).to_owned()).collect()
            );
            assert_eq!(row.fields()["mode"], "chat");
        });
    }

    #[rstest]
    fn a_row_object_is_one_identity_across_reads() {
        Python::initialize();
        Python::attach(|py| {
            let rows = dict(py, c_str!("{'a': {'mode': 'chat'}, 'b': {'mode': 'chat'}}"));
            let store = Store::new(LiveDict::new(&rows));
            let read = |key| store.get(key).unwrap().into_owned().identity();

            assert_eq!(read("a"), read("a"));
            assert_ne!(read("a"), read("b"));
        });
    }
}
