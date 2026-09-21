use litellm_secrets_types::{AccessMode, KeyManagementSettings, KeyManagementSystem, SecretValue};
use pyo3::{prelude::*, types::PyAny};
use serde_json::Value;

use crate::coercion::{Field, ProjectionError};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SecretManagerSnapshot {
    pub(crate) system: Option<KeyManagementSystem>,
    pub(crate) premium_user: bool,
    pub(crate) settings: KeyManagementSettings,
}

pub(crate) fn project(value: &Bound<'_, PyAny>) -> Result<SecretManagerSnapshot, ProjectionError> {
    let system = parse_optional_system(&Field::read(value, "secret_manager.system")?)?;
    let access_mode = parse_access_mode(&Field::read(value, "secret_manager.access_mode")?)?;
    let hosted_keys = Field::read(value, "secret_manager.hosted_keys")?
        .optional_string_collection()?
        .map(|collection| collection.0);
    let replica_regions = Field::read(value, "secret_manager.replica_regions")?
        .optional_string_collection()?
        .map(|collection| collection.0);
    let settings = KeyManagementSettings {
        hosted_keys,
        store_virtual_keys: Some(
            Field::read(value, "secret_manager.store_virtual_keys")?
                .truthy()?
                .0,
        ),
        prefix_for_stored_virtual_keys: Field::read(
            value,
            "secret_manager.prefix_for_stored_virtual_keys",
        )?
        .strict_string()?,
        access_mode,
        primary_secret_name: Field::read(value, "secret_manager.primary_secret_name")?
            .falsy_optional_string()?
            .0,
        kms_key_id: Field::read(value, "secret_manager.kms_key_id")?
            .falsy_optional_string()?
            .0,
        custom_secret_manager: Field::read(value, "secret_manager.custom_secret_manager")?
            .falsy_optional_string()?
            .0,
        aws_region_name: Field::read(value, "secret_manager.aws_region_name")?
            .falsy_optional_string()?
            .0,
        aws_role_name: Field::read(value, "secret_manager.aws_role_name")?
            .falsy_optional_string()?
            .0,
        aws_session_name: Field::read(value, "secret_manager.aws_session_name")?
            .falsy_optional_string()?
            .0,
        aws_external_id: Field::read(value, "secret_manager.aws_external_id")?
            .falsy_optional_string()?
            .0
            .map(SecretValue::new),
        aws_profile_name: Field::read(value, "secret_manager.aws_profile_name")?
            .falsy_optional_string()?
            .0,
        aws_web_identity_token: Field::read(value, "secret_manager.aws_web_identity_token")?
            .falsy_optional_string()?
            .0
            .map(SecretValue::new),
        aws_sts_endpoint: Field::read(value, "secret_manager.aws_sts_endpoint")?
            .falsy_optional_string()?
            .0,
        replica_regions,
        ..KeyManagementSettings::default()
    };
    Ok(SecretManagerSnapshot {
        system,
        premium_user: Field::read(value, "secret_manager.premium_user")?
            .truthy()?
            .0,
        settings,
    })
}

fn parse_optional_system(
    field: &Field<'_>,
) -> Result<Option<KeyManagementSystem>, ProjectionError> {
    let Some(value) = field.falsy_optional_string()?.0 else {
        return Ok(None);
    };
    serde_json::from_value(Value::String(value))
        .map(Some)
        .map_err(|error| {
            ProjectionError::InvalidConfiguration(format!("secret manager system: {error}"))
        })
}

fn parse_access_mode(field: &Field<'_>) -> Result<AccessMode, ProjectionError> {
    let value = field.strict_string()?;
    serde_json::from_value(Value::String(value)).map_err(|error| {
        ProjectionError::InvalidConfiguration(format!("secret manager access mode: {error}"))
    })
}

#[cfg(test)]
mod tests {
    use pyo3::{
        prelude::*,
        types::{PyDict, PyTuple},
    };

    use super::project;

    fn snapshot<'py>(
        py: Python<'py>,
        system: &str,
        access_mode: &str,
        premium_user: Bound<'py, PyAny>,
        hosted_keys: Bound<'py, PyAny>,
    ) -> Bound<'py, PyAny> {
        let locals = PyDict::new(py);
        locals.set_item("system", system).unwrap();
        locals.set_item("access_mode", access_mode).unwrap();
        locals.set_item("premium_user", premium_user).unwrap();
        locals.set_item("hosted_keys", hosted_keys).unwrap();
        py.run(
            cr#"
from dataclasses import dataclass
from types import SimpleNamespace

@dataclass(frozen=True, slots=True)
class SecretManager:
    system: object
    access_mode: object
    hosted_keys: object
    primary_secret_name: object
    premium_user: object
    store_virtual_keys: object
    prefix_for_stored_virtual_keys: object
    kms_key_id: object
    custom_secret_manager: object
    aws_region_name: object
    aws_role_name: object
    aws_session_name: object
    aws_external_id: object
    aws_profile_name: object
    aws_web_identity_token: object
    aws_sts_endpoint: object
    replica_regions: object

root = SimpleNamespace(secret_manager=SecretManager(
    system=system,
    access_mode=access_mode,
    hosted_keys=hosted_keys,
    primary_secret_name=None,
    premium_user=premium_user,
    store_virtual_keys=False,
    prefix_for_stored_virtual_keys="litellm/",
    kms_key_id=None,
    custom_secret_manager=None,
    aws_region_name=None,
    aws_role_name=None,
    aws_session_name=None,
    aws_external_id=None,
    aws_profile_name=None,
    aws_web_identity_token=None,
    aws_sts_endpoint=None,
    replica_regions=None,
))
"#,
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
        locals
            .get_item("root")
            .unwrap()
            .unwrap()
            .getattr("secret_manager")
            .unwrap()
    }

    #[rstest::rstest]
    #[case::string_true(Some("true"), false, true)]
    #[case::string_one(Some("1"), false, true)]
    #[case::true_value(None, true, true)]
    #[case::false_value(None, false, false)]
    #[case::string_false(Some("false"), false, true)]
    fn python_compatible_boolean_coercion(
        #[case] string_value: Option<&str>,
        #[case] bool_value: bool,
        #[case] expected: bool,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let premium_user = match string_value {
                Some(value) => value.into_pyobject(py).unwrap().into_any(),
                None => bool_value.into_pyobject(py).unwrap().to_owned().into_any(),
            };
            let hosted_keys = PyTuple::new(py, ["ONE"]).unwrap().into_any();
            let projected = project(&snapshot(
                py,
                "local",
                "read_only",
                premium_user,
                hosted_keys,
            ))
            .unwrap();
            assert_eq!(projected.premium_user, expected);
        });
    }

    #[test]
    fn unknown_system_is_rejected() {
        Python::initialize();
        Python::attach(|py| {
            let error = project(&snapshot(
                py,
                "unknown",
                "read_only",
                false.into_pyobject(py).unwrap().to_owned().into_any(),
                PyTuple::empty(py).into_any(),
            ))
            .unwrap_err();
            let error: PyErr = error.into();
            assert!(error.is_instance_of::<pyo3::exceptions::PyValueError>(py));
        });
    }
}
