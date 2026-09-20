//! The Python contract of this crate: every function of
//! `litellm/rust_bridge/callbacks_v1.py` that Rust calls, by name.

use pyo3::prelude::*;
use strum::{IntoStaticStr, VariantArray};

const MODULE: &str = "litellm.rust_bridge.callbacks_v1";

#[derive(Clone, Copy, Debug, IntoStaticStr, PartialEq, Eq, VariantArray)]
pub(crate) enum V1Python {
    #[strum(serialize = "snapshot")]
    Snapshot,
    #[strum(serialize = "project_response")]
    ProjectResponse,
    #[strum(serialize = "new_call_id")]
    NewCallId,
    #[strum(serialize = "report")]
    Report,
}

impl V1Python {
    fn name(self) -> &'static str {
        self.into()
    }

    pub(crate) fn call<'py, A>(self, py: Python<'py>, args: A) -> PyResult<Bound<'py, PyAny>>
    where
        A: pyo3::call::PyCallArgs<'py>,
    {
        py.import(MODULE)?.getattr(self.name())?.call1(args)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use strum::VariantArray;

    use super::V1Python;
    use crate::test_support::PYTHON_CONTRACT;

    #[test]
    fn every_function_is_in_the_python_contract() {
        let contract: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(PYTHON_CONTRACT).unwrap();
        let declared: BTreeSet<&str> = contract.keys().map(String::as_str).collect();
        let called: Vec<&str> = V1Python::VARIANTS
            .iter()
            .copied()
            .map(V1Python::name)
            .collect();
        assert_eq!(called.len(), declared.len(), "a function is listed twice");
        assert_eq!(called.into_iter().collect::<BTreeSet<_>>(), declared);
    }
}
