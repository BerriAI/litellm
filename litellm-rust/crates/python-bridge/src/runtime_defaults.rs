use litellm_auth::{ParamSpec, VertexParams};
use litellm_llms::base_llm::connection::RuntimeDefaults;
use pyo3::prelude::*;

use crate::{
    coercion::{FieldSpec, ProjectionError},
    python_settings::{PythonSettings, Snapshot},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RuntimeDefault {
    ApiBase,
    VertexProject,
    VertexLocation,
}

impl RuntimeDefault {
    const fn name(self) -> &'static str {
        match self {
            Self::ApiBase => "api_base",
            Self::VertexProject => "vertex_project",
            Self::VertexLocation => "vertex_location",
        }
    }

    /// The Vertex defaults whose spec the call leaves `unset`.
    pub(crate) fn vertex_fallbacks(
        unset: impl Fn(&ParamSpec) -> bool,
    ) -> impl Iterator<Item = Self> {
        [
            (Self::VertexProject, VertexParams::PROJECT),
            (Self::VertexLocation, VertexParams::LOCATION),
        ]
        .into_iter()
        .filter(move |(_, spec)| unset(spec))
        .map(|(default, _)| default)
    }
}

/// The `litellm.<name>` module defaults the call falls back to, read once per call into the
/// typed runtime-defaults layer that provider resolvers rank between explicit arguments and
/// the environment. A default the call does not fall back to is never read, so a malformed
/// one cannot fail an unrelated call.
pub(crate) fn read(py: Python<'_>, wanted: &[RuntimeDefault]) -> PyResult<RuntimeDefaults> {
    if wanted.is_empty() {
        return Ok(RuntimeDefaults::default());
    }
    match PythonSettings::ProviderDefaults.read_or_unset(py)? {
        Some(snapshot) => project(&snapshot, wanted).map_err(PyErr::from),
        None => Ok(RuntimeDefaults::default()),
    }
}

fn project(
    snapshot: &Snapshot<'_>,
    wanted: &[RuntimeDefault],
) -> Result<RuntimeDefaults, ProjectionError> {
    let read = |default: RuntimeDefault| match wanted.contains(&default) {
        true => snapshot.read(&FieldSpec::new(default.name(), |field| {
            field.falsy_optional_string()
        })),
        false => Ok(None),
    };
    Ok(RuntimeDefaults {
        api_base: read(RuntimeDefault::ApiBase)?,
        vertex_project: read(RuntimeDefault::VertexProject)?,
        vertex_location: read(RuntimeDefault::VertexLocation)?,
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const EVERY: [RuntimeDefault; 3] = [
        RuntimeDefault::ApiBase,
        RuntimeDefault::VertexProject,
        RuntimeDefault::VertexLocation,
    ];

    fn projected(
        source: &std::ffi::CStr,
        wanted: &[RuntimeDefault],
    ) -> Result<RuntimeDefaults, String> {
        Python::initialize();
        Python::attach(|py| {
            let value = py.eval(source, None, None).unwrap();
            project(&PythonSettings::ProviderDefaults.snapshot(value), wanted)
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
        assert_eq!(projected(source, &EVERY).unwrap(), expected);
    }

    #[rstest]
    #[case::nothing_wanted(&[], RuntimeDefaults::default())]
    #[case::only_the_api_base(
        &[RuntimeDefault::ApiBase],
        RuntimeDefaults { api_base: Some("https://runtime.test".into()), ..Default::default() }
    )]
    fn a_malformed_default_the_call_does_not_use_is_never_read(
        #[case] wanted: &[RuntimeDefault],
        #[case] expected: RuntimeDefaults,
    ) {
        let source = c"__import__('types').SimpleNamespace(api_base='https://runtime.test', vertex_project=5, vertex_location=5)";
        assert_eq!(projected(source, wanted).unwrap(), expected);
    }

    #[rstest]
    #[case::both_unset(&[], vec![RuntimeDefault::VertexProject, RuntimeDefault::VertexLocation])]
    #[case::project_on_the_call(&["vertex_ai_project"], vec![RuntimeDefault::VertexLocation])]
    #[case::both_on_the_call(&["vertex_project", "vertex_location"], vec![])]
    fn vertex_fallbacks_skip_the_settings_the_call_supplies(
        #[case] supplied: &[&str],
        #[case] expected: Vec<RuntimeDefault>,
    ) {
        let fallbacks = RuntimeDefault::vertex_fallbacks(|spec| {
            !spec.wire.iter().any(|name| supplied.contains(name))
        })
        .collect::<Vec<_>>();
        assert_eq!(fallbacks, expected);
    }

    #[test]
    fn a_misconfigured_module_default_names_the_setting() {
        let error = projected(
            c"__import__('types').SimpleNamespace(api_base=1, vertex_project=None, vertex_location=None)",
            &EVERY,
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
            let first = project(
                &PythonSettings::ProviderDefaults.snapshot(namespace.clone()),
                &EVERY,
            )
            .unwrap();
            namespace
                .setattr("api_base", "https://second.test")
                .unwrap();
            let second = project(
                &PythonSettings::ProviderDefaults.snapshot(namespace),
                &EVERY,
            )
            .unwrap();
            assert_eq!(
                (first.api_base.as_deref(), second.api_base.as_deref()),
                (Some("https://first.test"), Some("https://second.test"))
            );
        });
    }
}
