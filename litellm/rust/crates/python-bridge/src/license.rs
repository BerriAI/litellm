use std::time::Duration;

use litellm_error::Report;
use litellm_http::{ClientVariant, HttpClientConfig};
use litellm_license::{LICENSE_SERVER, PublicKey, Verifier};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(
    frozen,
    name = "LicenseVerifier",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct LicenseVerifier {
    verifier: Verifier,
    http: HttpClientConfig,
}

#[pymethods]
impl LicenseVerifier {
    #[new]
    #[pyo3(signature = (*, timeout_seconds, public_key_pem = None, license_server = None))]
    fn new(
        py: Python<'_>,
        timeout_seconds: f64,
        public_key_pem: Option<&str>,
        license_server: Option<String>,
    ) -> PyResult<Self> {
        let public_key = match public_key_pem {
            Some(pem) => PublicKey::from_pem(pem)
                .map_err(|error| PyValueError::new_err(Report::new(&error).to_string()))?,
            None => PublicKey::litellm(),
        };
        let timeout = Duration::try_from_secs_f64(timeout_seconds).map_err(|_| {
            PyValueError::new_err("timeout_seconds must be a non-negative number of seconds")
        })?;
        Ok(Self {
            verifier: Verifier::new(
                public_key,
                license_server.unwrap_or_else(|| LICENSE_SERVER.to_owned()),
                timeout,
            ),
            http: crate::http::settings_config(py, true)?,
        })
    }

    /// Runs on a throwaway runtime like the cost map fetch, since the proxy checks its license at import, before it
    /// may fork workers
    fn entitlements(&self, py: Python<'_>, license: Option<String>) -> PyResult<Entitlements> {
        let Some(license) = license else {
            return Ok(Entitlements(litellm_license::Entitlements::Community));
        };
        let logger = crate::logger::capture(py);
        let entitlements = litellm_host_python::release_gil(py, || {
            logger.scope(|| {
                litellm_host_python::run_isolated(self.verifier.entitlements(&license, || {
                    crate::http::pool().unpooled(&self.http, ClientVariant::Provider)
                }))
            })
        })?;
        Ok(Entitlements(entitlements))
    }
}

#[pyclass(frozen, name = "Entitlements", module = "litellm.rust_bridge._native")]
pub(crate) struct Entitlements(litellm_license::Entitlements);

#[pymethods]
impl Entitlements {
    #[classattr]
    #[allow(non_snake_case)]
    fn COMMUNITY() -> Self {
        Self(litellm_license::Entitlements::Community)
    }

    /// The message for a limit that a license granting `feature` would lift
    #[staticmethod]
    fn missing_feature_error(feature: &str) -> PyResult<String> {
        Ok(
            Report::new(&litellm_license::Error::MissingFeature(parse_feature(
                feature,
            )?))
            .to_string(),
        )
    }

    fn is_enterprise(&self) -> bool {
        self.0.is_enterprise()
    }

    fn grants(&self, feature: &str) -> PyResult<bool> {
        Ok(self.0.grants(parse_feature(feature)?))
    }

    /// The signed claims as a dict, or `None` without a signed license
    #[getter]
    fn license(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.0
            .license()
            .map(|license| litellm_host_python::to_py(py, license))
            .transpose()
    }

    #[getter]
    fn max_users(&self) -> Option<i64> {
        self.0.max_users()
    }

    fn is_over_user_limit(&self, users: i64) -> bool {
        self.0.is_over_user_limit(users)
    }

    fn is_over_team_limit(&self, teams: i64) -> bool {
        self.0.is_over_team_limit(teams)
    }
}

fn parse_feature(feature: &str) -> PyResult<litellm_license::Feature> {
    feature
        .parse()
        .map_err(|_| PyValueError::new_err(format!("unknown license feature: {feature}")))
}
