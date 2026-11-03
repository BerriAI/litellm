use std::{ffi::OsString, net::SocketAddr, str::FromStr, time::Duration};

use axum::http::Uri;
use serde::Deserialize;

use crate::Error;

const UPSTREAM: &str = "LITELLM_FORWARD_UPSTREAM";

/// The Python proxy's private origin. Only a loopback IPv4 address parses, so nothing can point
/// the forwarder at another host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Upstream(Uri);

impl Upstream {
    pub(crate) fn uri(&self) -> &Uri {
        &self.0
    }
}

impl FromStr for Upstream {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self, Error> {
        let uri: Uri = value.parse().map_err(Error::UpstreamUri)?;
        let address = uri
            .authority()
            .filter(|_| uri.scheme_str() == Some("http"))
            .filter(|authority| !authority.as_str().contains('@'))
            .filter(|_| uri.path() == "/" && uri.query().is_none())
            .and_then(|authority| authority.as_str().parse::<SocketAddr>().ok());
        match address {
            Some(SocketAddr::V4(address)) if address.ip().is_loopback() => Ok(Self(uri)),
            _ => Err(Error::Upstream),
        }
    }
}

#[derive(Deserialize)]
struct Environment {
    litellm_forward_upstream: String,
    litellm_forward_host: String,
    litellm_forward_port: u16,
    litellm_forward_upstream_keepalive: f64,
    litellm_forward_shutdown_drain: f64,
}

/// Everything forward mode runs with. The Python supervisor resolves each value from the settings
/// Python actually runs with, so none of them is guessed or defaulted here.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    pub upstream: Upstream,
    pub host: String,
    pub port: u16,
    pub upstream_keepalive: Duration,
    pub shutdown_drain: Duration,
}

impl Settings {
    /// `None` when `LITELLM_FORWARD_UPSTREAM` is unset, which means the gateway is not forwarding.
    pub fn from_env() -> Result<Option<Self>, Error> {
        Self::from_vars(std::env::vars_os())
    }

    pub fn from_vars(
        variables: impl IntoIterator<Item = (OsString, OsString)>,
    ) -> Result<Option<Self>, Error> {
        let variables: Vec<(String, String)> = variables
            .into_iter()
            .filter_map(|(key, value)| Some((key.into_string().ok()?, value.into_string().ok()?)))
            .collect();
        if !variables.iter().any(|(key, _)| key == UPSTREAM) {
            return Ok(None);
        }
        let environment: Environment = envy::from_iter(variables).map_err(|error| match error {
            envy::Error::MissingValue(field) => Error::MissingSetting(field.to_uppercase()),
            envy::Error::Custom(message) => Error::InvalidSetting(message),
        })?;
        Ok(Some(Self {
            upstream: environment.litellm_forward_upstream.parse()?,
            host: environment.litellm_forward_host,
            port: environment.litellm_forward_port,
            upstream_keepalive: positive_seconds(
                "LITELLM_FORWARD_UPSTREAM_KEEPALIVE",
                environment.litellm_forward_upstream_keepalive,
            )?,
            shutdown_drain: seconds(
                "LITELLM_FORWARD_SHUTDOWN_DRAIN",
                environment.litellm_forward_shutdown_drain,
            )?,
        }))
    }
}

fn seconds(variable: &str, value: f64) -> Result<Duration, Error> {
    Duration::try_from_secs_f64(value).map_err(|_| {
        Error::InvalidSetting(format!(
            "{variable} must be a non-negative number of seconds, got {value}"
        ))
    })
}

/// The keep-alive also bounds the wait for a client's first request, so zero would close every
/// client connection before it could send one.
fn positive_seconds(variable: &str, value: f64) -> Result<Duration, Error> {
    match seconds(variable, value)? {
        Duration::ZERO => Err(Error::InvalidSetting(format!(
            "{variable} must be a positive number of seconds, got {value}"
        ))),
        duration => Ok(duration),
    }
}
