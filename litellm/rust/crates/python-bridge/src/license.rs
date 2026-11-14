use std::time::Duration;

use litellm_error::Report;
use litellm_http::{ClientVariant, HttpClientConfig};
use litellm_license::{LICENSE_SERVER, PublicKey};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(
    frozen,
    name = "LicenseVerifier",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct LicenseVerifier {
    public_key: PublicKey,
    server: String,
    timeout: Duration,
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
            public_key,
            server: license_server.unwrap_or_else(|| LICENSE_SERVER.to_owned()),
            timeout,
            http: crate::http::settings_config(py, true)?,
        })
    }

    /// A license the bundled key signed carries its claims. Any other license is enterprise only when the license
    /// server vouches for it
    fn entitlements(&self, py: Python<'_>, license: Option<String>) -> PyResult<Entitlements> {
        let Some(license) = license else {
            return Ok(Entitlements(litellm_license::Entitlements::Community));
        };
        let error = match self
            .public_key
            .verify(&license, jiff::Zoned::now().datetime())
        {
            Ok(signed) => {
                return Ok(Entitlements(litellm_license::Entitlements::Enterprise(
                    Some(signed),
                )));
            }
            Err(error) => error,
        };
        crate::logger::capture(py).scope(|| {
            litellm_tracing::debug!("license not verified locally: {}", Report::new(&error));
        });
        Ok(Entitlements(if self.ask_server(py, license)? {
            litellm_license::Entitlements::Enterprise(None)
        } else {
            litellm_license::Entitlements::Community
        }))
    }
}

impl LicenseVerifier {
    /// Runs on a throwaway runtime like the cost map fetch, since the proxy checks its license at import, before it
    /// may fork workers
    fn ask_server(&self, py: Python<'_>, license: String) -> PyResult<bool> {
        let (config, server, timeout) = (self.http.clone(), self.server.clone(), self.timeout);
        let logger = crate::logger::capture(py);
        litellm_host_python::release_gil(py, move || {
            logger.scope(|| {
                let client = crate::http::pool()
                    .unpooled(&config, ClientVariant::Provider)
                    .map_err(crate::http::client_error)?;
                let verdict = litellm_host_python::run_isolated(litellm_license::verify_remote(
                    &client, &server, &license, timeout,
                ))?;
                Ok(verdict.unwrap_or_else(|error| {
                    litellm_tracing::error!(
                        "license not verified by the license server: {}",
                        Report::new(&error)
                    );
                    false
                }))
            })
        })
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

    #[staticmethod]
    fn remedy(feature: &str) -> PyResult<String> {
        Ok(parse_feature(feature)?.remedy())
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
