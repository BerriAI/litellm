use litellm_config::{GeneralSettings, JwtAuthSettings, RolePermissions};
use litellm_gateway_auth_core::{HostFacts, MasterKey, Settings};
use litellm_host_python::from_py;
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

/// A route list as `route_in_additonal_public_routes` read it: a value that is not a list of
/// routes never fails the request, and only its string entries count.
fn routes(general: &Bound<'_, PyAny>, key: &str) -> PyResult<Option<Box<[String]>>> {
    let Some(value) = get(general, key)? else {
        return Ok(None);
    };
    if value.is_instance_of::<PyString>() {
        return Ok(None);
    }
    let Ok(entries) = value.try_iter() else {
        return Ok(None);
    };
    Ok(Some(
        entries
            .filter_map(|entry| entry.ok()?.extract::<String>().ok())
            .collect(),
    ))
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
    kwargs.set_item(
        "exclude",
        PySet::new(py, [PyString::new(py, "custom_validate")])?,
    )?;
    let dumped = litellm_jwtauth.call_method("model_dump", (), Some(&kwargs))?;
    let custom_validate = litellm_jwtauth.getattr("custom_validate")?;
    Ok(JwtAuthSettings {
        custom_validate: (!custom_validate.is_none())
            .then(|| custom_validate.repr().map(|r| r.to_string()))
            .transpose()?,
        ..from_py(&dumped)?
    })
}

/// Reads the live `general_settings` dict through PyO3 into the typed config the gateway loads
/// from YAML, so both hosts hand the core the same `GeneralSettings`.
fn general_settings(
    general: &Bound<'_, PyAny>,
    litellm_jwtauth: Option<&Bound<'_, PyAny>>,
) -> PyResult<GeneralSettings> {
    Ok(GeneralSettings {
        enable_jwt_auth: is_true(general, "enable_jwt_auth")?,
        enable_oauth2_auth: is_true(general, "enable_oauth2_auth")?,
        enable_oauth2_proxy_auth: is_true(general, "enable_oauth2_proxy_auth")?,
        allow_user_auth: is_true(general, "allow_user_auth")?,
        public_routes: routes(general, "public_routes")?,
        role_permissions: role_permissions(general)?,
        enforce_rbac: Some(
            get(general, "enforce_rbac")?
                .map(|value| value.is_truthy())
                .transpose()?
                .unwrap_or(false),
        ),
        litellm_jwtauth: litellm_jwtauth.map(jwt_auth).transpose()?,
        ..GeneralSettings::default()
    })
}

#[pyclass(
    frozen,
    name = "RustAuthSettings",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct AuthSettings(pub(crate) Settings);

/// The host facts Python passes in, read from a frozen dataclass (`core.HostFacts`). `route_groups`
/// is the one [`HostFacts`] field it leaves out, since the bridge reads `LiteLLMRoutes` itself.
#[derive(FromPyObject)]
struct Facts {
    #[pyo3(attribute)]
    master_key: Option<String>,
    #[pyo3(attribute)]
    premium: bool,
    #[pyo3(attribute)]
    has_db: bool,
    #[pyo3(attribute)]
    ui_login: bool,
    #[pyo3(attribute)]
    proxy_budget: bool,
    #[pyo3(attribute)]
    metrics_require_auth: bool,
}

#[pymethods]
impl AuthSettings {
    #[new]
    fn new(
        general_settings: &Bound<'_, PyAny>,
        litellm_jwtauth: Option<&Bound<'_, PyAny>>,
        facts: Facts,
    ) -> PyResult<Self> {
        let general = self::general_settings(general_settings, litellm_jwtauth)?;
        let route_groups = super::route_groups::route_groups(general_settings.py())?.clone();
        let facts = HostFacts {
            master_key: facts.master_key.map(MasterKey::new),
            premium: facts.premium,
            has_db: facts.has_db,
            ui_login: facts.ui_login,
            proxy_budget: facts.proxy_budget,
            metrics_require_auth: facts.metrics_require_auth,
            route_groups,
        };
        Ok(Self(Settings::new(&general, facts, &|name| {
            std::env::var(name).ok()
        })))
    }

    #[getter]
    fn jwt_leeway_seconds(&self) -> u64 {
        self.0.jwt.leeway_seconds
    }
}
