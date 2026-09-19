use pyo3::prelude::*;
use strum::{IntoStaticStr, VariantArray};

const MODULE: &str = "litellm.rust_bridge.callbacks_next";

#[derive(Clone, Copy, Debug, IntoStaticStr, PartialEq, Eq, VariantArray)]
pub(crate) enum NextPython {
    #[strum(serialize = "snapshot")]
    Snapshot,
    #[strum(serialize = "project_response")]
    ProjectResponse,
    #[strum(serialize = "new_call_id")]
    NewCallId,
    #[strum(serialize = "report")]
    Report,
}

impl NextPython {
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

    use super::NextPython;

    #[test]
    fn every_function_is_in_the_python_contract() {
        let contract: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(include_str!("../python_contract.json")).unwrap();
        let declared: BTreeSet<&str> = contract.keys().map(String::as_str).collect();
        let called: BTreeSet<&str> = NextPython::VARIANTS
            .iter()
            .copied()
            .map(NextPython::name)
            .collect();
        assert_eq!(called, declared);
    }
}
