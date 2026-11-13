use std::collections::BTreeMap;

use litellm_python_compat::lenient::{IsTrue, Truthy};
use litellm_secret_value::SecretValue;
use litellm_starlette::RouteTemplate;
use serde::Deserialize;
use serde_with::serde_as;

use crate::{Object, Spelled};

/// A `general_settings.pass_through_endpoints` entry: a proxy `path` forwarded to `target`. Each
/// field is read the way `initialize_pass_through_endpoints` reads the config dict.
#[serde_as]
#[derive(Clone, Debug, Deserialize)]
pub struct PassThroughEndpoint {
    pub id: Option<String>,
    pub path: RouteTemplate,
    /// Registration skips an endpoint without one.
    pub target: Option<SecretValue>,
    /// Upstream credentials and `os.environ/` references, so only the keys are printed.
    #[serde(default)]
    pub headers: Object,
    #[serde(default)]
    pub default_query_params: Object,
    /// `endpoint.get("include_subpath", False) is True`
    #[serde_as(as = "IsTrue")]
    #[serde(default)]
    pub include_subpath: bool,
    pub cost_per_request: Option<Spelled<f64>>,
    pub timeout: Option<Spelled<f64>>,
    /// `endpoint.get("auth", True) is True`, which decides whether a request needs a key.
    /// Registration still reads `str(auth).lower() == "true"`, until both share one rule.
    #[serde_as(as = "IsTrue")]
    #[serde(default = "auth_default")]
    pub auth: bool,
    pub methods: Option<Box<[String]>>,
    pub guardrails: Option<BTreeMap<String, Option<PassThroughGuardrail>>>,
    #[serde_as(as = "Truthy")]
    #[serde(default)]
    pub forward_headers: bool,
    #[serde_as(as = "Truthy")]
    #[serde(default)]
    pub merge_query_params: bool,
    /// `"langfuse"` reads the key out of a Basic credential.
    pub custom_auth_parser: Option<String>,
    #[serde(flatten)]
    pub additional_fields: Object,
}

const fn auth_default() -> bool {
    true
}

/// `PassThroughGuardrailSettings`: which request and response fields one guardrail reads.
#[derive(Clone, Debug, Deserialize)]
pub struct PassThroughGuardrail {
    pub request_fields: Option<Box<[String]>>,
    pub response_fields: Option<Box<[String]>>,
    #[serde(flatten)]
    pub additional_fields: Object,
}
