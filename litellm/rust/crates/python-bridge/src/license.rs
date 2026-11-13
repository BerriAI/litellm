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

    /// `None` when the key did not sign this license or it expired
    fn verify_offline(&self, py: Python<'_>, license: &str) -> Option<License> {
        self.public_key
            .verify(license, jiff::Zoned::now().datetime())
            .map(License)
            .inspect_err(|error| {
                crate::logger::capture(py).scope(|| {
                    litellm_tracing::debug!("license not verified locally: {}", Report::new(error));
                });
            })
            .ok()
    }

    /// Runs on a throwaway runtime like the cost map fetch, since the proxy checks its license at import, before it
    /// may fork workers
    fn verify_remote(&self, py: Python<'_>, license: String) -> PyResult<bool> {
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

#[pyclass(frozen, name = "License", module = "litellm.rust_bridge._native")]
pub(crate) struct License(litellm_license::License);

#[pymethods]
impl License {
    #[getter]
    fn expiration_date(&self) -> String {
        self.0.expiration_date.to_string()
    }

    #[getter]
    fn user_id(&self) -> Option<&str> {
        self.0.user_id.as_deref()
    }

    #[getter]
    fn allowed_features(&self) -> Vec<String> {
        self.0.allowed_features.clone()
    }

    #[getter]
    fn max_users(&self) -> Option<i64> {
        self.0.max_users
    }

    #[getter]
    fn max_teams(&self) -> Option<i64> {
        self.0.max_teams
    }

    fn grants(&self, feature: &str) -> bool {
        self.0.grants(feature)
    }

    fn is_over_user_limit(&self, users: i64) -> bool {
        self.0.is_over_user_limit(users)
    }

    fn is_over_team_limit(&self, teams: i64) -> bool {
        self.0.is_over_team_limit(teams)
    }
}
