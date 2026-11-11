use litellm_config::{GeneralSettings, JwtAuthSettings, RolePermissions};
use litellm_gateway_auth_core::{HostFacts, Settings};
use litellm_host_python::from_py;
use litellm_secret_value::SecretValue;
use pyo3::{
    prelude::*,
    types::{PyBool, PyDict, PySet, PyString},
};

/// `general_settings.get(key)`. The proxy's settings are a dict or its `SettingsStore` mapping.
fn get<'py>(general: &Bound<'py, PyAny>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    let value = general.call_method1("get", (key,))?;
    Ok((!value.is_none()).then_some(value))
}

/// `general_settings.get(key, False) is True`, the check the proxy makes on each auth flag.
fn is_true(general: &Bound<'_, PyAny>, key: &str) -> PyResult<Option<bool>> {
    let py = general.py();
    Ok(get(general, key)?.map(|value| value.is(PyBool::new(py, true))))
}

fn routes(general: &Bound<'_, PyAny>, key: &str) -> PyResult<Option<Box<[String]>>> {
    get(general, key)?.map(|value| value.extract::<Vec<String>>().map(Into::into)).transpose()
}

/// `general_settings.role_permissions`: `RoleBasedPermissions` models or the dicts the config holds.
fn role_permissions(general: &Bound<'_, PyAny>) -> PyResult<Option<Box<[RolePermissions]>>> {
    let Some(value) = get(general, "role_permissions")? else {
        return Ok(None);
    };
    let py = general.py();
    value
        .try_iter()?
        .map(|item| {
            let item = item?;
            match item.hasattr("model_dump")? {
                true => {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("mode", "json")?;
                    from_py(&item.call_method("model_dump", (), Some(&kwargs))?)
                }
                false => from_py(&item),
            }
        })
        .collect::<PyResult<Vec<_>>>()
        .map(|permissions| Some(permissions.into()))
}

/// The validated `LiteLLM_JWTAuth`. `custom_validate` is a callable only Python can run, so the
/// core only learns that one is set.
pub(crate) fn jwt_auth(litellm_jwtauth: &Bound<'_, PyAny>) -> PyResult<JwtAuthSettings> {
    let py = litellm_jwtauth.py();
    let kwargs = PyDict::new(py);
    kwargs.set_item("mode", "json")?;
    kwargs.set_item("exclude", PySet::new(py, [PyString::new(py, "custom_validate")])?)?;
    let dumped = litellm_jwtauth.call_method("model_dump", (), Some(&kwargs))?;
    let custom_validate = litellm_jwtauth.getattr("custom_validate")?;
    Ok(JwtAuthSettings {
        custom_validate: (!custom_validate.is_none()).then(|| custom_validate.repr().map(|r| r.to_string())).transpose()?,
        ..from_py(&dumped)?
    })
}

/// Reads the live `general_settings` dict through PyO3 into the typed config the gateway loads
/// from YAML, so both hosts hand the core the same `GeneralSettings`.
fn general_settings(general: &Bound<'_, PyAny>, litellm_jwtauth: Option<&Bound<'_, PyAny>>) -> PyResult<GeneralSettings> {
    Ok(GeneralSettings {
        enable_jwt_auth: is_true(general, "enable_jwt_auth")?,
        enable_oauth2_auth: is_true(general, "enable_oauth2_auth")?,
        enable_oauth2_proxy_auth: is_true(general, "enable_oauth2_proxy_auth")?,
        allow_user_auth: is_true(general, "allow_user_auth")?,
        allow_requests_on_db_unavailable: get(general, "allow_requests_on_db_unavailable")?
            .map(|value| from_py(&value))
            .transpose()?
            .unwrap_or_default(),
        public_routes: routes(general, "public_routes")?,
        role_permissions: role_permissions(general)?,
        enforce_rbac: Some(get(general, "enforce_rbac")?.map(|value| value.is_truthy()).transpose()?.unwrap_or(false)),
        litellm_jwtauth: litellm_jwtauth.map(jwt_auth).transpose()?,
        ..GeneralSettings::default()
    })
}

#[pyclass(frozen, name = "RustAuthSettings", module = "litellm.rust_bridge._native")]
pub(crate) struct AuthSettings(pub(crate) Settings);

#[pymethods]
impl AuthSettings {
    #[staticmethod]
    #[pyo3(signature = (
        general_settings,
        litellm_jwtauth,
        *,
        master_key,
        premium,
        has_db,
        ui_login,
        proxy_budget,
        metrics_require_auth,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_general_settings(
        general_settings: &Bound<'_, PyAny>,
        litellm_jwtauth: Option<&Bound<'_, PyAny>>,
        master_key: Option<String>,
        premium: bool,
        has_db: bool,
        ui_login: bool,
        proxy_budget: bool,
        metrics_require_auth: bool,
    ) -> PyResult<Self> {
        let general = self::general_settings(general_settings, litellm_jwtauth)?;
        let facts = HostFacts {
            master_key: master_key.map(SecretValue::new),
            premium,
            has_db,
            ui_login,
            proxy_budget,
            metrics_require_auth,
        };
        Ok(Self(Settings::new(&general, facts, &|name| std::env::var(name).ok())))
    }
}
