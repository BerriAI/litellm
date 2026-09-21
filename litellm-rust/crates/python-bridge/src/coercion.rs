use std::collections::BTreeSet;

use litellm_core_utils::serde_compat::parse_str_bool;
use litellm_http::SslVerify;
use pyo3::{
    exceptions::{PyAttributeError, PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBool, PyString},
};

#[derive(Debug)]
pub(crate) enum ProjectionError {
    Python(PyErr),
    InvalidConfiguration(String),
    UnsupportedLiveObject(String),
    InternalSchemaFailure(String),
}

impl From<PyErr> for ProjectionError {
    fn from(error: PyErr) -> Self {
        Self::Python(error)
    }
}

impl From<ProjectionError> for PyErr {
    fn from(error: ProjectionError) -> Self {
        match error {
            ProjectionError::Python(error) => error,
            ProjectionError::InvalidConfiguration(message)
            | ProjectionError::UnsupportedLiveObject(message) => PyValueError::new_err(message),
            ProjectionError::InternalSchemaFailure(message) => PyRuntimeError::new_err(message),
        }
    }
}

pub(crate) struct Truthy(pub bool);
pub(crate) struct ExactTrue(pub bool);
pub(crate) struct StrBool(pub Option<bool>);
pub(crate) struct OptionalStrictString(pub Option<String>);
pub(crate) struct FalsyOptionalString(pub Option<String>);
pub(crate) struct TuningString(pub Option<String>);
pub(crate) struct StringCollection(pub Vec<String>);
pub(crate) struct SslVerifyInput(pub Option<SslVerify>);

pub(crate) struct Field<'py> {
    path: &'static str,
    value: Bound<'py, PyAny>,
}

impl<'py> Field<'py> {
    pub(crate) fn new(path: &'static str, value: Bound<'py, PyAny>) -> Self {
        Self { path, value }
    }

    pub(crate) fn read(
        snapshot: &Bound<'py, PyAny>,
        path: &'static str,
    ) -> Result<Self, ProjectionError> {
        let name = path.rsplit('.').next().unwrap_or(path);
        match snapshot.getattr(name) {
            Ok(value) => Ok(Self::new(path, value)),
            Err(error) if error.is_instance_of::<PyAttributeError>(snapshot.py()) => {
                match Self::missing_field(snapshot, name) {
                    Ok(true) => Err(ProjectionError::InternalSchemaFailure(format!(
                        "{path}: missing snapshot field"
                    ))),
                    _ => Err(error.into()),
                }
            }
            Err(error) => Err(error.into()),
        }
    }

    fn missing_field(snapshot: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
        let py = snapshot.py();
        let object = py.import("builtins")?.getattr("object")?;
        let missing = object.call0()?;
        let lookup = py.import("inspect")?.getattr("getattr_static")?;
        let declared = lookup.call1((snapshot, name, &missing))?;
        let fallback = lookup.call1((snapshot.get_type(), "__getattr__", &missing))?;
        let getter = lookup.call1((snapshot.get_type(), "__getattribute__"))?;
        Ok(declared.is(&missing)
            && fallback.is(&missing)
            && getter.is(object.getattr("__getattribute__")?))
    }

    fn expected(&self, expected: &'static str) -> Result<String, ProjectionError> {
        Ok(format!(
            "{}: expected {expected}, got {}",
            self.path,
            self.value.get_type().name()?
        ))
    }

    fn invalid(&self, expected: &'static str) -> ProjectionError {
        match self.expected(expected) {
            Ok(message) => ProjectionError::InvalidConfiguration(message),
            Err(error) => error,
        }
    }

    pub(crate) fn truthy(&self) -> Result<Truthy, ProjectionError> {
        Ok(Truthy(self.value.is_truthy()?))
    }

    pub(crate) fn exact_true(&self) -> ExactTrue {
        ExactTrue(self.value.is(PyBool::new(self.value.py(), true)))
    }

    pub(crate) fn strict_string(&self) -> Result<String, ProjectionError> {
        let value = self
            .value
            .cast::<PyString>()
            .map_err(|_| self.invalid("a string"))?;
        Ok(value.to_str()?.to_owned())
    }

    pub(crate) fn schema_string(&self) -> Result<String, ProjectionError> {
        if !self.value.is_instance_of::<PyString>() {
            return Err(ProjectionError::InternalSchemaFailure(
                self.expected("a string")?,
            ));
        }
        self.strict_string()
    }

    pub(crate) fn schema_bool(&self) -> Result<bool, ProjectionError> {
        if !self.value.is_instance_of::<PyBool>() {
            return Err(ProjectionError::InternalSchemaFailure(
                self.expected("a Boolean")?,
            ));
        }
        Ok(self.exact_true().0)
    }

    pub(crate) fn str_bool(&self) -> Result<StrBool, ProjectionError> {
        if self.value.is_none() {
            return Ok(StrBool(None));
        }
        Ok(StrBool(parse_str_bool(&self.strict_string()?)))
    }

    pub(crate) fn optional_strict_string(&self) -> Result<OptionalStrictString, ProjectionError> {
        if self.value.is_none() {
            return Ok(OptionalStrictString(None));
        }
        self.strict_string().map(Some).map(OptionalStrictString)
    }

    pub(crate) fn falsy_optional_string(&self) -> Result<FalsyOptionalString, ProjectionError> {
        if !self.truthy()?.0 {
            return Ok(FalsyOptionalString(None));
        }
        self.strict_string().map(Some).map(FalsyOptionalString)
    }

    pub(crate) fn tuning_string(&self) -> Result<TuningString, ProjectionError> {
        if !self.truthy()?.0 || !self.value.is_instance_of::<PyString>() {
            return Ok(TuningString(None));
        }
        self.strict_string().map(Some).map(TuningString)
    }

    pub(crate) fn string_collection(&self) -> Result<StringCollection, ProjectionError> {
        if !self.truthy()?.0 {
            return Ok(StringCollection(Vec::new()));
        }
        if self.value.is_instance_of::<PyString>() {
            return self
                .strict_string()
                .map(|value| StringCollection(vec![value]));
        }
        let values = self
            .value
            .try_iter()?
            .filter_map(|item| {
                let member = match item {
                    Ok(value) => Self::new(self.path, value),
                    Err(error) => return Some(Err(error.into())),
                };
                match member.truthy() {
                    Ok(Truthy(false)) => None,
                    Ok(Truthy(true)) => Some(member.strict_string()),
                    Err(error) => Some(Err(error)),
                }
            })
            .collect::<Result<Vec<_>, ProjectionError>>()?;
        Ok(StringCollection(values))
    }

    pub(crate) fn host_collection(&self) -> Result<StringCollection, ProjectionError> {
        let values = self
            .string_collection()?
            .0
            .into_iter()
            .map(|host| litellm_http::media::normalize_host(&host))
            .collect::<BTreeSet<_>>();
        Ok(StringCollection(values.into_iter().collect()))
    }

    pub(crate) fn ssl_verify(&self) -> Result<SslVerifyInput, ProjectionError> {
        if self.value.is_none() {
            return Ok(SslVerifyInput(None));
        }
        if self.value.is_instance_of::<PyBool>() {
            return Ok(SslVerifyInput(Some(if self.exact_true().0 {
                SslVerify::Enabled
            } else {
                SslVerify::Disabled
            })));
        }
        if self.value.is_instance_of::<PyString>() {
            let parsed = match self.str_bool()?.0 {
                Some(true) => SslVerify::Enabled,
                Some(false) => SslVerify::Disabled,
                None => SslVerify::CaBundle(self.strict_string()?.into()),
            };
            return Ok(SslVerifyInput(Some(parsed)));
        }
        let context = self.value.py().import("ssl")?.getattr("SSLContext")?;
        if self.value.is_instance(&context)? {
            return Err(ProjectionError::UnsupportedLiveObject(self.expected(
                "a Boolean, Boolean string, CA path, or None; live SSLContext is unsupported",
            )?));
        }
        Err(self.invalid("a Boolean, Boolean string, CA path, or None"))
    }
}

#[cfg(test)]
mod tests;
