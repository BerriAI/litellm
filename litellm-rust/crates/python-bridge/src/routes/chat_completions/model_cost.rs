//! Python's `litellm.model_cost` as the cost map chat translation reads, so a capability
//! lookup sees the map Python itself would, models registered at runtime included.

use std::{borrow::Cow, collections::HashSet, sync::OnceLock};

use litellm_host_python::from_py;
use litellm_llms::base_llm::model_info::ModelInfo;
use pyo3::{prelude::*, types::PyDict};
use serde_json::{Map, Value};

pub(super) struct PythonModelCost {
    map: Py<PyDict>,
    declared: OnceLock<HashSet<String>>,
}

impl PythonModelCost {
    /// The dict `litellm.model_cost` holds now. A map Python replaces later is not followed,
    /// as a call already under way would not follow it either.
    pub(super) fn current(py: Python<'_>) -> PyResult<Self> {
        let map = py
            .import("litellm")?
            .getattr("model_cost")?
            .cast_into::<PyDict>()?
            .unbind();
        Ok(Self {
            map,
            declared: OnceLock::new(),
        })
    }
}

impl ModelInfo for PythonModelCost {
    fn entry(&self, model: &str) -> Option<Cow<'_, Map<String, Value>>> {
        Python::attach(|py| {
            let entry = self.map.bind(py).get_item(model).ok()??;
            from_py::<Map<String, Value>>(&entry).ok().map(Cow::Owned)
        })
    }

    fn declares(&self, key: &str) -> bool {
        self.declared
            .get_or_init(|| {
                Python::attach(|py| {
                    self.map
                        .bind(py)
                        .values()
                        .iter()
                        .filter_map(|entry| entry.cast_into::<PyDict>().ok())
                        .flat_map(|entry| {
                            entry
                                .keys()
                                .iter()
                                .filter_map(|key| key.extract::<String>().ok())
                                .collect::<Vec<_>>()
                        })
                        .collect()
                })
            })
            .contains(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_entry_python_holds_and_the_keys_any_entry_sets() {
        Python::initialize();
        Python::attach(|py| {
            let map = PyDict::new(py);
            let entry = PyDict::new(py);
            entry.set_item("supports_reasoning", true).unwrap();
            entry.set_item("max_output_tokens", 64000).unwrap();
            map.set_item("claude-x", entry).unwrap();
            let cost = PythonModelCost {
                map: map.unbind(),
                declared: OnceLock::new(),
            };
            let found = cost.entry("claude-x").unwrap();
            assert_eq!(found.get("supports_reasoning"), Some(&Value::Bool(true)));
            assert_eq!(found.get("max_output_tokens"), Some(&Value::from(64000)));
            assert!(cost.entry("unknown").is_none());
            assert!(cost.declares("max_output_tokens"));
            assert!(!cost.declares("default_reasoning_effort"));
        });
    }
}
