use pyo3::{exceptions::PyModuleNotFoundError, prelude::*};

use crate::coercion::{FieldSpec, ProjectionError};

const MODULE: &str = "litellm.rust_bridge.settings";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PythonSettings {
    Http,
    UrlPolicy,
    ProviderDefaults,
    SecretManager,
    SecretManagerBinding,
}

pub(crate) struct Snapshot<'py> {
    group: PythonSettings,
    value: Bound<'py, PyAny>,
}

impl Snapshot<'_> {
    pub(crate) fn read<T>(&self, spec: &FieldSpec<T>) -> Result<T, ProjectionError> {
        spec.read(&self.value, self.group.name())
    }
}

impl PythonSettings {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Http => "http_settings",
            Self::UrlPolicy => "url_policy",
            Self::ProviderDefaults => "provider_defaults",
            Self::SecretManager => "secret_manager",
            Self::SecretManagerBinding => "secret_manager_binding",
        }
    }

    pub(crate) fn read(self, py: Python<'_>) -> PyResult<Snapshot<'_>> {
        let value = py.import(MODULE)?.getattr(self.name())?.call0()?;
        Ok(Snapshot { group: self, value })
    }

    /// Reads the accessor, or `None` when the litellm package is not installed
    /// (a bare extension module), meaning there are no configured values.
    pub(crate) fn read_or_unset(self, py: Python<'_>) -> PyResult<Option<Snapshot<'_>>> {
        match self.read(py) {
            Ok(snapshot) => Ok(Some(snapshot)),
            Err(error) if error.is_instance_of::<PyModuleNotFoundError>(py) => Ok(None),
            Err(error) => Err(error),
        }
    }

    #[cfg(test)]
    pub(crate) fn snapshot(self, value: Bound<'_, PyAny>) -> Snapshot<'_> {
        Snapshot { group: self, value }
    }
}

#[cfg(test)]
mod tests {
    use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};

    use super::PythonSettings;
    use crate::coercion::FieldSpec;

    #[test]
    fn declarations_select_the_decoder_and_read_only_the_requested_field() {
        const TRUTHY: FieldSpec<bool> = FieldSpec::new("flag", |field| field.truthy());
        const EXACT: FieldSpec<bool> = FieldSpec::new("flag", |field| Ok(field.exact_true()));
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"
reads = []
class Settings:
    value = 1
    @property
    def flag(self):
        reads.append('flag')
        return self.value
    @property
    def unrelated(self):
        raise AssertionError('unrequested field')
settings = Settings()
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let value = locals.get_item("settings").unwrap().unwrap();
            let snapshot = PythonSettings::Http.snapshot(value.clone());
            assert!(snapshot.read(&TRUTHY).unwrap());
            assert!(!snapshot.read(&EXACT).unwrap());
            value.setattr("value", true).unwrap();
            assert!(snapshot.read(&EXACT).unwrap());
            assert_eq!(
                locals
                    .get_item("reads")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                ["flag", "flag", "flag"]
            );
        });
    }

    #[test]
    fn declared_reads_preserve_descriptor_and_decoder_failures_and_name_missing_fields() {
        const FLAG: FieldSpec<bool> = FieldSpec::new("flag", |field| field.truthy());
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"
from types import SimpleNamespace
failure = AttributeError('read failed')
class Descriptor:
    @property
    def flag(self): raise failure
class Truth:
    def __bool__(self): raise failure
values = (Descriptor(), SimpleNamespace(flag=Truth()))
",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let failure = locals.get_item("failure").unwrap().unwrap();
            for value in locals
                .get_item("values")
                .unwrap()
                .unwrap()
                .try_iter()
                .unwrap()
            {
                let snapshot = PythonSettings::Http.snapshot(value.unwrap());
                let error = PyErr::from(snapshot.read(&FLAG).unwrap_err());
                assert!(error.value(py).is(&failure));
                assert!(error.traceback(py).is_some());
            }
            let missing = PythonSettings::Http.snapshot(py.eval(c"object()", None, None).unwrap());
            let error = PyErr::from(missing.read(&FLAG).unwrap_err());
            assert!(error.is_instance_of::<PyRuntimeError>(py));
            assert!(
                error
                    .to_string()
                    .contains("http_settings.flag: missing snapshot field")
            );
        });
    }
}
