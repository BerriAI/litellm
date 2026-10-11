use litellm_llms::base_llm::connection::RuntimeDefaults;
use pyo3::prelude::*;

use crate::{
    coercion::{FieldSpec, ProjectionError},
    python_settings::{PythonSettings, Snapshot},
};

/// The `litellm.<name>` module defaults, read once per call into the typed runtime-defaults
/// layer that provider resolvers rank between explicit arguments and the environment.
pub(crate) fn read(py: Python<'_>) -> PyResult<RuntimeDefaults> {
    match PythonSettings::ProviderDefaults.read_or_unset(py)? {
        Some(snapshot) => project(&snapshot).map_err(PyErr::from),
        None => Ok(RuntimeDefaults::default()),
    }
}

fn project(snapshot: &Snapshot<'_>) -> Result<RuntimeDefaults, ProjectionError> {
    let read = |name: &'static str| {
        snapshot.read(&FieldSpec::new(name, |field| field.falsy_optional_string()))
    };
    Ok(RuntimeDefaults {
        api_base: read("api_base")?,
        vertex_project: read("vertex_project")?,
        vertex_location: read("vertex_location")?,
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn projected(source: &std::ffi::CStr) -> Result<RuntimeDefaults, String> {
        Python::initialize();
        Python::attach(|py| {
            let value = py.eval(source, None, None).unwrap();
            project(&PythonSettings::ProviderDefaults.snapshot(value))
                .map_err(|error| PyErr::from(error).value(py).to_string())
        })
    }

    #[rstest]
    #[case::every_default(
        c"__import__('types').SimpleNamespace(api_base='https://runtime.test', vertex_project='p', vertex_location='us-east5')",
        RuntimeDefaults {
            api_base: Some("https://runtime.test".into()),
            vertex_project: Some("p".into()),
            vertex_location: Some("us-east5".into()),
        }
    )]
    #[case::falsy_values_are_unset(
        c"__import__('types').SimpleNamespace(api_base='', vertex_project=None, vertex_location=[])",
        RuntimeDefaults::default()
    )]
    fn module_defaults_project_into_their_own_layer(
        #[case] source: &std::ffi::CStr,
        #[case] expected: RuntimeDefaults,
    ) {
        assert_eq!(projected(source).unwrap(), expected);
    }

    #[test]
    fn a_misconfigured_module_default_names_the_setting() {
        let error = projected(
            c"__import__('types').SimpleNamespace(api_base=1, vertex_project=None, vertex_location=None)",
        )
        .unwrap_err();
        assert!(error.contains("provider_defaults.api_base"), "{error}");
    }

    #[test]
    fn each_call_reads_the_current_module_defaults() {
        Python::initialize();
        Python::attach(|py| {
            let namespace = py
                .eval(
                    c"__import__('types').SimpleNamespace(api_base='https://first.test', vertex_project=None, vertex_location=None)",
                    None,
                    None,
                )
                .unwrap();
            let first =
                project(&PythonSettings::ProviderDefaults.snapshot(namespace.clone())).unwrap();
            namespace
                .setattr("api_base", "https://second.test")
                .unwrap();
            let second = project(&PythonSettings::ProviderDefaults.snapshot(namespace)).unwrap();
            assert_eq!(
                (first.api_base.as_deref(), second.api_base.as_deref()),
                (Some("https://first.test"), Some("https://second.test"))
            );
        });
    }
}
