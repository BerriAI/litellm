use litellm_core_utils::{
    settings::{Selected, SettingSource},
    url_utils::ApiUrl,
};
use litellm_router_types::LitellmParams;
use serde::{Deserialize, Serialize};

use super::auth::{AuthScheme, Headers};

/// Defaults a host reads from its own runtime configuration. They are kept apart from a
/// call's explicit arguments so a provider resolver can rank them below every explicit
/// spelling and above the environment.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeDefaults {
    pub api_base: Option<String>,
    pub vertex_project: Option<String>,
    pub vertex_location: Option<String>,
}

#[derive(Clone, Copy)]
pub struct ConnectionInputs<'a> {
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub litellm_params: &'a LitellmParams,
    pub runtime_defaults: &'a RuntimeDefaults,
}

pub struct PreparedConnection {
    pub url: String,
    pub headers: Headers,
    pub auth: AuthScheme,
    pub api_base_source: SettingSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("api_base is not a valid base URL")]
pub struct InvalidApiBase;

impl From<InvalidApiBase> for crate::Error {
    fn from(error: InvalidApiBase) -> Self {
        Self::InvalidRequest(crate::ErrorDetail::invalid("api_base", error))
    }
}

pub fn validated_api_base(selected: Selected<String>) -> Result<Selected<String>, InvalidApiBase> {
    ApiUrl::parse(&selected.value)
        .map(|_| selected)
        .map_err(|_| InvalidApiBase)
}

pub fn complete_url(base: &str, path: &[&str]) -> Result<String, InvalidApiBase> {
    ApiUrl::parse(base)
        .and_then(|url| url.complete_path(path))
        .map(ApiUrl::into_string)
        .map_err(|_| InvalidApiBase)
}

pub fn path_segments(path: &str) -> impl Iterator<Item = &str> {
    path.split('/').filter(|segment| !segment.is_empty())
}
