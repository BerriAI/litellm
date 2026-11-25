//! The request body check. A caller who could set a deployment's endpoint, identity or credentials
//! from the body could send the proxy's own secrets to a host of their choosing, so each of those
//! parameters is refused wherever a body can carry it unless an admin granted it.
//!
//! This is a port of the Python `is_request_body_safe`. Each function keeps the name and the place
//! in the call order its Python original had, so the two read side by side. It asks nothing of the
//! process it runs in: [`needs`] names the model groups and stored credentials a body can reach,
//! the caller reads them, and [`check`] decides on what it was handed.

use std::{
    borrow::Cow,
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
};

use super::{
    destination::url_destination_allowed,
    form::form_metadata,
    grant::{MAX_OPT_IN_FALLBACK_MODELS, Reachable},
    names::{
        API_KEY, BUCKET_PARAMS, CREDENTIAL_NAME, FALLBACK_FIELDS, METADATA_KEYS,
        NESTED_CONFIG_KEYS, Names, ROUTER_SETTINGS_OVERRIDE, VERTEX_TARGET_PARAMS, is_content_key,
        is_metadata_key,
    },
    policy::{Deferred, Policy, allowlist},
    surface::Surface,
    value::{CallerValue, Concealed, Params},
};
use crate::error::body::{Rejection, ServerOwnedKind};

const USER_CONFIG: &str = "user_config";
const LITELLM_PARAMS: &str = "litellm_params";
const CHATGPT_AUTH_FILE: &str = "chatgpt_auth_file";
const EXTRA_BODY: &str = "extra_body";

/// One request body and what its route says about it.
#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    pub body: &'a Params,
    pub surface: Surface,
    /// The model the request names, empty when it names none.
    pub model: &'a str,
    /// The model the route's path names, on routes that carry one there.
    pub route_model: Option<&'a str>,
    pub deferred: Deferred,
}

/// The values each stored credential a body names holds, as the caller read them. A name it
/// holds no entry for is a credential that does not exist.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Credentials(BTreeMap<String, Params>);

impl FromIterator<(String, Params)> for Credentials {
    fn from_iter<I: IntoIterator<Item = (String, Params)>>(entries: I) -> Self {
        Self(entries.into_iter().collect())
    }
}

/// What a caller reads before it calls [`check`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Needs {
    /// The model names the request can reach: its model, or the route's when it names none, the
    /// route's, and every fallback target the body names. The caller resolves them to the groups
    /// they reach through aliases, the router's fallbacks and the groups each deployment found
    /// routes under, and hands them to [`check`] as a [`Reachable`].
    pub models: BTreeSet<String>,
    /// Every `litellm_credential_name` the check reads, wherever the body carries one.
    pub credentials: BTreeSet<String>,
}

/// What `body` makes [`check`] read. It walks the body as [`check`] does, with every override
/// granted and no credential found, so it lists everything a check on the real data can read
/// before it stops.
pub fn needs(request: &Request<'_>, names: &Names, policy: &Policy) -> Needs {
    let read = RefCell::new(BTreeSet::new());
    let _ = Check {
        request,
        names,
        policy,
        deployments: Deployments::Granted,
        credentials: Lookup::Collect(&read),
    }
    .is_request_body_safe();
    Needs {
        models: requested_models(request).0,
        credentials: read.into_inner(),
    }
}

pub fn check(
    request: &Request<'_>,
    reachable: &Reachable,
    credentials: &Credentials,
    names: &Names,
    policy: &Policy,
) -> Result<(), Rejection> {
    Check {
        request,
        names,
        policy,
        deployments: Deployments::Resolved(reachable),
        credentials: Lookup::Read(credentials),
    }
    .is_request_body_safe()
}

/// The requested model names, and how many come from the body's fallbacks and the route.
fn requested_models(request: &Request<'_>) -> (BTreeSet<String>, usize) {
    let route_model = request.route_model.filter(|model| !model.is_empty());
    let root = without_route_named_vertex_target(request.body, request.surface);
    let also_requested: BTreeSet<&str> = request_fallback_targets(&root)
        .unwrap_or_default()
        .into_iter()
        .filter_map(fallback_target_model_name)
        .chain(route_model)
        .collect();
    let model = if request.model.is_empty() {
        route_model.unwrap_or_default()
    } else {
        request.model
    };
    let count = also_requested.len();
    let models = also_requested
        .into_iter()
        .chain([model])
        .map(str::to_owned)
        .collect();
    (models, count)
}

/// Where the check reads a deployment's opt-in from.
#[derive(Clone, Copy, Debug)]
enum Deployments<'a> {
    /// The groups the caller resolved from [`Needs::models`].
    Resolved(&'a Reachable),
    /// [`needs`]: every override is granted, so the walk goes on past it.
    Granted,
}

/// Where the check reads a stored credential from.
#[derive(Clone, Copy, Debug)]
enum Lookup<'a> {
    Read(&'a Credentials),
    /// [`needs`]: each name is recorded and found holding nothing.
    Collect(&'a RefCell<BTreeSet<String>>),
}

impl Lookup<'_> {
    fn values(&self, name: &str) -> Cow<'_, Params> {
        match self {
            Self::Read(credentials) => credentials
                .0
                .get(name)
                .map_or_else(|| Cow::Owned(Params::default()), Cow::Borrowed),
            Self::Collect(read) => {
                read.borrow_mut().insert(name.to_owned());
                Cow::Owned(Params::default())
            }
        }
    }
}

/// What a mapping of a body is. Each variant stands for one set of the keyword arguments the
/// Python `_check_banned_params` was called with, and the methods below are those arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Judged {
    /// The body itself, where a route may read some override names as its own fields.
    Root,
    /// Keyword arguments spread into the call: a nested config or a fallback target.
    CallParams,
    /// `extra_body`, which also may not set the call's fallbacks.
    ExtraBody,
    /// A fallback target's `extra_body`, judged for its fallbacks alone, since the target itself
    /// was judged as call parameters.
    FallbackExtraBody,
    /// Tags the observability callbacks read. Nothing resolves them, so one that looks like a
    /// secret reference is left alone, and the names providers alone read are no overrides.
    Metadata,
}

impl Judged {
    /// An MCP server route's token exchange settings are its own fields.
    fn mcp_server_fields_allowed(self, surface: Surface) -> bool {
        self == Self::Root && surface.manages_mcp_servers
    }

    fn route_owned_params(self, surface: Surface) -> &'static [&'static str] {
        match self {
            Self::Root => surface.owned,
            Self::CallParams | Self::ExtraBody | Self::FallbackExtraBody | Self::Metadata => &[],
        }
    }

    fn rejects_secret_references(self) -> bool {
        match self {
            Self::Root | Self::CallParams | Self::ExtraBody | Self::FallbackExtraBody => true,
            Self::Metadata => false,
        }
    }

    fn banned_params(self, names: &Names) -> &[String] {
        match self {
            Self::Root | Self::CallParams => names.overrides(),
            Self::ExtraBody => names.extra_body_overrides(),
            Self::FallbackExtraBody => names.fallbacks_only(),
            Self::Metadata => names.metadata_overrides(),
        }
    }
}

/// Who lets a caller set an override.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Admitted {
    /// The route reads the name as its own field.
    ByRoute,
    /// `allow_client_side_credentials` is on.
    ProxyWide,
    /// The value is on an allowlist in the proxy's settings.
    Allowlisted,
    /// A check that runs later decides, once it knows the caller's team.
    LeftToLaterCheck,
    /// Every deployment the request can reach opts in.
    ByDeployments,
}

struct Check<'a> {
    request: &'a Request<'a>,
    names: &'a Names,
    policy: &'a Policy,
    deployments: Deployments<'a>,
    credentials: Lookup<'a>,
}

type Checked = Result<(), Rejection>;

impl Check<'_> {
    fn is_request_body_safe(&self) -> Checked {
        let request = self.request;
        if request.body.contains("model_list") {
            return Err(Rejection::ModelList);
        }
        let root_body = without_route_named_vertex_target(request.body, request.surface);
        let fallback_targets = request_fallback_targets(&root_body)?;
        self.check_banned_params(&root_body, Judged::Root)?;
        for nested_key in NESTED_CONFIG_KEYS {
            if let Some(nested) = coerce_metadata_to_dict(request.body.get(nested_key)) {
                let judged = if nested_key == EXTRA_BODY {
                    Judged::ExtraBody
                } else {
                    Judged::CallParams
                };
                self.check_banned_params(&nested, judged)?;
            }
        }
        for metadata_key in METADATA_KEYS {
            if let Some(metadata) = coerce_metadata_to_dict(request.body.get(metadata_key)) {
                self.check_banned_params(&metadata, Judged::Metadata)?;
            }
            let prefix = format!("{metadata_key}[");
            if request.body.keys().any(|key| key.starts_with(&prefix)) {
                self.check_banned_params(&form_metadata(request.body, &prefix), Judged::Metadata)?;
            }
        }
        for target in fallback_targets.iter().copied() {
            match target {
                CallerValue::Map(target) => {
                    self.check_banned_params(target, Judged::CallParams)?;
                    let extra_body = coerce_metadata_to_dict(target.get(EXTRA_BODY))
                        .map(|extra_body| extra_body.only("fallbacks"))
                        .unwrap_or_default();
                    self.check_banned_params(&extra_body, Judged::FallbackExtraBody)?;
                    if let Some(target_model) = target.get("model").and_then(CallerValue::text) {
                        self.reject_url_valued_fallback_target(target_model)?;
                    }
                }
                CallerValue::Text(target) => self.reject_url_valued_fallback_target(target)?,
                CallerValue::Null
                | CallerValue::List(_)
                | CallerValue::TooDeep
                | CallerValue::Other => {}
            }
        }
        for user_config_params in iter_user_config_litellm_params(request.body) {
            self.reject_client_chosen_federation(&user_config_params)?;
        }
        if let Some(litellm_params) = coerce_metadata_to_dict(request.body.get(LITELLM_PARAMS)) {
            self.reject_request_credential_params(&litellm_params)?;
            if let Some(litellm_params_metadata) =
                coerce_metadata_to_dict(litellm_params.get("metadata"))
            {
                self.check_banned_params(&litellm_params_metadata, Judged::Metadata)?;
            }
        }
        for user_config_params in user_config_litellm_params(request.body) {
            self.reject_api_key_secret_reference(&user_config_params)?;
            reject_user_config_secret_references(&user_config_params)?;
        }
        if request.surface.model_call && !request.surface.manages_deployments {
            unlisted_secret_references(request.body, &fallback_targets)?;
        }
        Ok(())
    }

    fn check_banned_params(&self, body: &Params, judged: Judged) -> Checked {
        let surface = self.request.surface;
        let mcp_server_fields_allowed = judged.mcp_server_fields_allowed(surface);
        self.reject_server_owned_wif_params(body, |key| {
            mcp_server_fields_allowed && self.names.is_token_exchange(key)
        })?;
        if !surface.manages_deployments {
            self.reject_federated_credential_reference(body)?;
            if judged.rejects_secret_references() {
                self.reject_secret_references(body)?;
            }
        }
        for param in judged.banned_params(self.names) {
            let Some(value) = body.get(param) else {
                continue;
            };
            if self.admitted(judged, param, value)?.is_none() {
                return Err(banned_param_error(param));
            }
        }
        Ok(())
    }

    /// A federation or token-exchange setting decides whose identity the proxy presents upstream,
    /// and an `oidc/` api_key has the proxy mint its own token. No body sets either, with or
    /// without a client-side credential opt-in.
    fn reject_server_owned_wif_params(
        &self,
        body: &Params,
        exempt: impl Fn(&str) -> bool,
    ) -> Checked {
        self.reject_server_owned_wif_fields(body, exempt)?;
        if holds_api_key_secret_reference(body) {
            return Err(Rejection::ApiKeyOidcReference);
        }
        Ok(())
    }

    /// `reject_server_owned_wif_params` without the `oidc/` api_key check, for the deployment
    /// writes whose role gate already refuses a non-admin setting one. A form spells a repeated
    /// field `name[]`.
    fn reject_server_owned_wif_fields(
        &self,
        body: &Params,
        exempt: impl Fn(&str) -> bool,
    ) -> Checked {
        let submitted_names: Vec<&str> = body
            .keys()
            .filter(|key| !exempt(key))
            .map(|key| key.strip_suffix("[]").unwrap_or(key))
            .collect();
        let found = self
            .names
            .federation()
            .iter()
            .find(|param| submitted_names.contains(&param.as_str()));
        match found {
            Some(param) => Err(Rejection::ServerOwned {
                param: param.clone(),
                kind: if self.names.is_github_copilot(param) {
                    ServerOwnedKind::GitHubCopilot
                } else {
                    ServerOwnedKind::Federation
                },
            }),
            None => Ok(()),
        }
    }

    /// Naming a stored credential merges its values into the call, so a body naming a federated
    /// one picks a server-owned identity exactly as sending the federation fields inline would.
    fn reject_federated_credential_reference(&self, body: &Params) -> Checked {
        let Some(named) = credential_name(body) else {
            return Ok(());
        };
        let values = self.credentials.values(named);
        let field = self
            .names
            .federation()
            .iter()
            .map(String::as_str)
            .find(|name| values.get(name).is_some_and(|value| !value.is_null()))
            .or_else(|| holds_api_key_secret_reference(&values).then_some(API_KEY));
        match field {
            Some(field) => Err(Rejection::FederatedCredential {
                named: named.to_owned(),
                field: field.to_owned(),
            }),
            None => Ok(()),
        }
    }

    fn reject_secret_references(&self, body: &Params) -> Checked {
        reject_secret_reference_in(
            body.iter()
                .filter(|(param, _)| self.names.resolves_secrets(param)),
        )
    }

    /// The loop body of `check_banned_params`: the first of those who can let a caller set
    /// `param` that does, asked in the order the Python loop tried them. The deployments are
    /// asked last.
    fn admitted(
        &self,
        judged: Judged,
        param: &str,
        value: &CallerValue,
    ) -> Result<Option<Admitted>, Rejection> {
        let request = self.request;
        if judged.route_owned_params(request.surface).contains(&param) {
            return Ok(Some(Admitted::ByRoute));
        }
        if self.policy.allow_client_side_credentials {
            return Ok(Some(Admitted::ProxyWide));
        }
        if self.policy.admits(param, value) {
            return Ok(Some(Admitted::Allowlisted));
        }
        let defer_s3_buckets = request.deferred.s3_buckets && BUCKET_PARAMS.contains(&param);
        if defer_s3_buckets || request.deferred.model_checks {
            return Ok(Some(Admitted::LeftToLaterCheck));
        }
        let reachable = match self.deployments {
            Deployments::Granted => return Ok(Some(Admitted::ByDeployments)),
            Deployments::Resolved(reachable) => reachable,
        };
        if requested_models(request).1 > MAX_OPT_IN_FALLBACK_MODELS {
            return Err(Rejection::TooManyFallbackModels {
                limit: MAX_OPT_IN_FALLBACK_MODELS,
            });
        }
        Ok(reachable
            .allow(param, value)
            .then_some(Admitted::ByDeployments))
    }

    fn reject_url_valued_fallback_target(&self, value: &str) -> Checked {
        if url_destination_allowed(value, &self.policy.url_destination_allowed_hosts) {
            return Ok(());
        }
        Err(Rejection::UrlFallbackDestination {
            value: value.to_owned(),
        })
    }

    /// A caller-supplied router config builds its deployments from the body, so its
    /// `litellm_params` are held to the federation rules a configured deployment's are not.
    fn reject_client_chosen_federation(&self, litellm_params: &Params) -> Checked {
        self.reject_server_owned_wif_params(litellm_params, |_| false)?;
        if !self.request.surface.manages_deployments {
            self.reject_federated_credential_reference(litellm_params)?;
        }
        Ok(())
    }

    fn reject_request_credential_params(&self, litellm_params: &Params) -> Checked {
        if self.request.surface.manages_deployments {
            self.reject_server_owned_wif_fields(litellm_params, |key| key == CHATGPT_AUTH_FILE)?;
        } else {
            self.reject_server_owned_wif_params(litellm_params, |_| false)?;
            self.reject_federated_credential_reference(litellm_params)?;
        }
        for nested_params in nested_configs(litellm_params) {
            self.reject_api_key_secret_reference(&nested_params)?;
        }
        Ok(())
    }

    fn reject_api_key_secret_reference(&self, params: &Params) -> Checked {
        if holds_api_key_secret_reference(params) {
            return Err(Rejection::ApiKeyOidcReference);
        }
        let Some(named) = credential_name(params) else {
            return Ok(());
        };
        if holds_api_key_secret_reference(&self.credentials.values(named)) {
            return Err(Rejection::CredentialApiKeyOidcReference {
                named: named.to_owned(),
            });
        }
        Ok(())
    }
}

/// Every key of a model call outside its content can reach a provider as a parameter, and
/// providers resolve references in more parameters than any list names. So a reference anywhere a
/// parameter can sit is refused, listed or not.
fn unlisted_secret_references(body: &Params, fallback_targets: &[&CallerValue]) -> Checked {
    for (key, value) in body.iter() {
        if is_content_key(key)
            || is_metadata_key(key)
            || NESTED_CONFIG_KEYS.contains(&key)
            || FALLBACK_FIELDS.contains(&key)
            || key == USER_CONFIG
        {
            continue;
        }
        let concealed = match (key, value.map()) {
            (ROUTER_SETTINGS_OVERRIDE, Some(settings)) => {
                concealed_outside(settings, |name| FALLBACK_FIELDS.contains(&name))
            }
            (LITELLM_PARAMS, Some(params)) => concealed_outside(params, is_metadata_key),
            _ => value.concealed(),
        };
        if let Some(concealed) = concealed {
            return Err(concealing(key, concealed));
        }
    }
    for config in nested_configs(body) {
        first_concealing(&config)?;
    }
    for target in fallback_targets.iter().filter_map(|target| target.map()) {
        first_concealing(target)?;
    }
    // A caller-supplied router resolves references in its own settings, not only in its
    // deployments.
    if let Some(config) = coerce_metadata_to_dict(body.get(USER_CONFIG)) {
        first_concealing(&config)?;
    }
    Ok(())
}

/// A depth the host did not read is refused like a reference, since one could sit there unseen.
fn concealing(param: &str, concealed: Concealed) -> Rejection {
    match concealed {
        Concealed::SecretReference => secret_reference(param),
        Concealed::BelowReadDepth => Rejection::TooDeep {
            param: param.to_owned(),
        },
    }
}

fn first_concealing(params: &Params) -> Checked {
    match params.iter().find_map(|(param, value)| {
        value
            .concealed()
            .map(|concealed| concealing(param, concealed))
    }) {
        Some(rejection) => Err(rejection),
        None => Ok(()),
    }
}

/// What the entries `skipped` does not name conceal.
fn concealed_outside(params: &Params, skipped: impl Fn(&str) -> bool) -> Option<Concealed> {
    params
        .iter()
        .filter(|(name, _)| !skipped(name))
        .find_map(|(_, value)| value.concealed())
}

/// `iter_request_fallback_targets`: every target the body's fallback lists name, at the root and
/// in `router_settings_override`, through targets that carry fallbacks of their own.
fn request_fallback_targets(body: &Params) -> Result<Vec<&CallerValue>, Rejection> {
    let override_settings = body
        .get(ROUTER_SETTINGS_OVERRIDE)
        .and_then(CallerValue::map);
    let mut targets = Vec::new();
    for source in [Some(body), override_settings].into_iter().flatten() {
        for field in FALLBACK_FIELDS {
            collect_fallback_targets(source.get(field), 0, &mut targets)?;
        }
    }
    Ok(targets)
}

/// `_iter_fallback_targets`. A mapping whose values are all lists is a keyed entry, not a target.
fn collect_fallback_targets<'v>(
    value: Option<&'v CallerValue>,
    depth: usize,
    targets: &mut Vec<&'v CallerValue>,
) -> Result<(), Rejection> {
    if depth > 2 * MAX_FALLBACKS {
        return Err(Rejection::FallbackNestingTooDeep);
    }
    let Some(items) = value.and_then(CallerValue::list) else {
        return Ok(());
    };
    for item in items {
        match item {
            CallerValue::Text(_) => targets.push(item),
            CallerValue::Map(entry) => {
                let keyed = entry.iter().next().is_some()
                    && entry.iter().all(|(_, value)| value.list().is_some());
                if !keyed {
                    targets.push(item);
                }
                if entry.get("model").and_then(CallerValue::text).is_some() {
                    for field in FALLBACK_FIELDS {
                        collect_fallback_targets(entry.get(field), depth + 1, targets)?;
                    }
                } else {
                    for (_, target_list) in entry.iter() {
                        collect_fallback_targets(Some(target_list), depth + 1, targets)?;
                    }
                }
            }
            CallerValue::Null
            | CallerValue::List(_)
            | CallerValue::TooDeep
            | CallerValue::Other => {}
        }
    }
    Ok(())
}

/// `litellm.ROUTER_MAX_FALLBACKS`.
const MAX_FALLBACKS: usize = 5;

/// The `litellm_params` of a caller-supplied router config: its defaults, then each
/// deployment's.
fn iter_user_config_litellm_params(body: &Params) -> Vec<Params> {
    let Some(config) = coerce_metadata_to_dict(body.get(USER_CONFIG)) else {
        return Vec::new();
    };
    let defaults = coerce_metadata_to_dict(config.get("default_litellm_params"));
    let model_list = config
        .get("model_list")
        .and_then(CallerValue::list)
        .unwrap_or_default();
    let deployments = model_list
        .iter()
        .filter_map(CallerValue::map)
        .filter_map(|deployment| coerce_metadata_to_dict(deployment.get("litellm_params")));
    defaults
        .into_iter()
        .chain(deployments)
        .map(Cow::into_owned)
        .collect()
}

/// The `litellm_params` of every deployment and search tool a caller-supplied router config
/// names, then its defaults.
fn user_config_litellm_params(body: &Params) -> Vec<Params> {
    let Some(user_config) = coerce_metadata_to_dict(body.get(USER_CONFIG)) else {
        return Vec::new();
    };
    let entries: Vec<Params> = ["model_list", "search_tools"]
        .into_iter()
        .flat_map(|key| mappings_in(user_config.get(key)))
        .filter_map(|entry| {
            coerce_metadata_to_dict(entry.get("litellm_params")).map(Cow::into_owned)
        })
        .collect();
    entries
        .into_iter()
        .chain(
            coerce_metadata_to_dict(user_config.get("default_litellm_params")).map(Cow::into_owned),
        )
        .collect()
}

/// The router resolves every field of a caller-supplied deployment the way it resolves a
/// configured one, so a reference in any of them reads a proxy secret for the caller.
fn reject_user_config_secret_references(litellm_params: &Params) -> Checked {
    reject_secret_reference_in(litellm_params.iter())?;
    for nested in nested_configs(litellm_params) {
        reject_secret_reference_in(nested.iter())?;
    }
    for parameters in data_source_parameters(litellm_params) {
        reject_secret_reference_in(parameters.iter())?;
    }
    Ok(())
}

fn data_source_parameters(litellm_params: &Params) -> Vec<Cow<'_, Params>> {
    mappings_in(litellm_params.get("dataSources"))
        .into_iter()
        .filter_map(|source| match source {
            Cow::Borrowed(source) => coerce_metadata_to_dict(source.get("parameters")),
            Cow::Owned(source) => coerce_metadata_to_dict(source.get("parameters"))
                .map(|found| Cow::Owned(found.into_owned())),
        })
        .collect()
}

fn nested_configs(params: &Params) -> Vec<Cow<'_, Params>> {
    NESTED_CONFIG_KEYS
        .into_iter()
        .filter_map(|key| coerce_metadata_to_dict(params.get(key)))
        .collect()
}

fn mappings_in(value: Option<&CallerValue>) -> Vec<Cow<'_, Params>> {
    value
        .and_then(CallerValue::list)
        .unwrap_or_default()
        .iter()
        .filter_map(|entry| coerce_metadata_to_dict(Some(entry)))
        .collect()
}

/// A mapping, or the one a JSON string decodes to, since forms and `extra_body` carry metadata
/// and configs as text the proxy decodes after this check.
fn coerce_metadata_to_dict(value: Option<&CallerValue>) -> Option<Cow<'_, Params>> {
    match value? {
        CallerValue::Map(params) => Some(Cow::Borrowed(params)),
        CallerValue::Text(text) => Params::decode(text).map(Cow::Owned),
        CallerValue::Null | CallerValue::List(_) | CallerValue::TooDeep | CallerValue::Other => {
            None
        }
    }
}

fn without_route_named_vertex_target(request_body: &Params, surface: Surface) -> Cow<'_, Params> {
    if surface.names_vertex_target {
        Cow::Owned(request_body.without(&VERTEX_TARGET_PARAMS))
    } else {
        Cow::Borrowed(request_body)
    }
}

fn credential_name(params: &Params) -> Option<&str> {
    params
        .get(CREDENTIAL_NAME)
        .and_then(CallerValue::text)
        .filter(|named| !named.is_empty())
}

fn holds_api_key_secret_reference(params: &Params) -> bool {
    params
        .get(API_KEY)
        .is_some_and(CallerValue::is_oidc_reference)
}

fn banned_param_error(param: &str) -> Rejection {
    Rejection::Override {
        param: param.to_owned(),
        allowlist: allowlist(param),
    }
}

fn secret_reference(param: &str) -> Rejection {
    Rejection::SecretReference {
        param: param.to_owned(),
    }
}

fn reject_secret_reference_in<'a>(
    params: impl IntoIterator<Item = (&'a str, &'a CallerValue)>,
) -> Result<(), Rejection> {
    match params
        .into_iter()
        .find(|(_, value)| value.is_secret_reference())
    {
        Some((param, _)) => Err(secret_reference(param)),
        None => Ok(()),
    }
}

fn fallback_target_model_name(target: &CallerValue) -> Option<&str> {
    match target {
        CallerValue::Text(name) => Some(name),
        CallerValue::Map(params) => params.get("model").and_then(CallerValue::text),
        CallerValue::Null | CallerValue::List(_) | CallerValue::TooDeep | CallerValue::Other => {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use serde_json::{Value, json};

    use super::{
        super::{
            grant::{Deployment, MAX_OPT_IN_FALLBACK_MODELS, OptIn},
            names::HostTables,
        },
        *,
    };
    use crate::error::body::Allowlist;

    const CHAT: &str = "/v1/chat/completions";

    fn params(json: &Value) -> Params {
        match CallerValue::from(json) {
            CallerValue::Map(params) => params,
            CallerValue::Null
            | CallerValue::Text(_)
            | CallerValue::List(_)
            | CallerValue::TooDeep
            | CallerValue::Other => Params::default(),
        }
    }

    /// The proxy the caller reads for the check: a router whose one group, `azure-group`, has one
    /// deployment that opts in to `opt_in`, and a stored credential named `federated`.
    #[derive(Default)]
    struct Proxy {
        opt_in: Option<Vec<&'static str>>,
        federated_credential: Value,
    }

    impl Proxy {
        /// What a caller resolves the requested names to: each name's group, empty for a name the
        /// router does not serve, and nothing without a router.
        fn reachable(&self, needs: &Needs) -> Reachable {
            let Some(opt_in) = &self.opt_in else {
                return Reachable::default();
            };
            let group = |model: &str| -> Vec<Deployment> {
                (model == "azure-group")
                    .then(|| Deployment {
                        model_name: "azure-group".to_owned(),
                        provider: Some("azure".to_owned()),
                        opt_in: Some(OptIn {
                            params: opt_in.iter().map(|param| (*param).to_owned()).collect(),
                            api_base_patterns: Vec::new(),
                        }),
                    })
                    .into_iter()
                    .collect()
            };
            Reachable {
                groups: needs.models.iter().map(|model| group(model)).collect(),
            }
        }

        fn credentials(&self, needs: &Needs) -> Credentials {
            needs
                .credentials
                .iter()
                .filter(|name| *name == "federated")
                .map(|name| (name.clone(), params(&self.federated_credential)))
                .collect()
        }
    }

    #[fixture]
    fn names() -> Names {
        let owned = |names: &[&str]| names.iter().map(|name| (*name).to_owned()).collect();
        Names::new(HostTables {
            callback_params: owned(&["langfuse_host"]),
            blocked_callback_params: Vec::new(),
            pricing_params: owned(&["input_cost_per_token"]),
            federation_params: owned(&[
                "anthropic_issuer_url",
                "token_exchange_endpoint",
                "github_copilot_app_id",
                "chatgpt_auth_file",
            ]),
            token_exchange_params: owned(&["token_exchange_endpoint"]),
            github_copilot_params: owned(&["github_copilot_app_id"]),
            upstream_bound_params: owned(&["api_key", "aws_access_key_id", "organization"]),
        })
    }

    struct Case<'a> {
        route: Option<&'a str>,
        route_model: Option<&'a str>,
        body: Value,
        policy: Policy,
        deferred: Deferred,
        proxy: Proxy,
    }

    impl Case<'_> {
        fn new(body: Value) -> Self {
            Self {
                route: Some(CHAT),
                route_model: None,
                body,
                policy: Policy::default(),
                deferred: Deferred::default(),
                proxy: Proxy::default(),
            }
        }

        fn with_request<T>(&self, judge: impl FnOnce(&Request<'_>) -> T) -> T {
            let body = params(&self.body);
            judge(&Request {
                body: &body,
                surface: Surface::of(self.route),
                model: self.body["model"].as_str().unwrap_or_default(),
                route_model: self.route_model,
                deferred: self.deferred,
            })
        }

        /// The check as a caller runs it: what it needs, read from the proxy, then the check.
        fn run(&self, names: &Names) -> Result<(), Rejection> {
            self.with_request(|request| {
                let needs = needs(request, names, &self.policy);
                check(
                    request,
                    &self.proxy.reachable(&needs),
                    &self.proxy.credentials(&needs),
                    names,
                    &self.policy,
                )
            })
        }

        /// Who lets the caller set the root's `param`.
        fn admitted(&self, names: &Names, param: &str) -> Option<Admitted> {
            self.with_request(|request| {
                let needs = needs(request, names, &self.policy);
                let reachable = self.proxy.reachable(&needs);
                let credentials = self.proxy.credentials(&needs);
                let check = Check {
                    request,
                    names,
                    policy: &self.policy,
                    deployments: Deployments::Resolved(&reachable),
                    credentials: Lookup::Read(&credentials),
                };
                let value = request.body.get(param)?;
                check.admitted(Judged::Root, param, value).ok().flatten()
            })
        }
    }

    fn on(route: &'static str, body: Value) -> Case<'static> {
        Case {
            route: Some(route),
            ..Case::new(body)
        }
    }

    fn open(body: Value) -> Case<'static> {
        Case {
            policy: Policy {
                allow_client_side_credentials: true,
                ..Policy::default()
            },
            ..Case::new(body)
        }
    }

    fn routed(opt_in: &[&'static str], body: Value) -> Case<'static> {
        Case {
            proxy: Proxy {
                opt_in: Some(opt_in.to_vec()),
                ..Proxy::default()
            },
            ..Case::new(body)
        }
    }

    fn refused(param: &str) -> Result<(), Rejection> {
        Err(Rejection::Override {
            param: param.to_owned(),
            allowlist: Allowlist::None,
        })
    }

    fn reference(param: &str) -> Result<(), Rejection> {
        Err(secret_reference(param))
    }

    #[rstest]
    #[case::endpoint(json!({"model": "gpt-4o", "api_base": "https://attacker.example"}), "api_base")]
    #[case::endpoint_beside_the_callers_own_key(
        json!({"model": "gpt-4o", "api_key": "sk-own", "api_base": "https://attacker.example"}),
        "api_base"
    )]
    #[case::entra_credential(json!({"model": "gpt-4o", "client_id": "caller"}), "client_id")]
    #[case::callback_destination(json!({"model": "gpt-4o", "langfuse_host": "https://attacker.example"}), "langfuse_host")]
    #[case::pricing(json!({"model": "gpt-4o", "input_cost_per_token": 0}), "input_cost_per_token")]
    #[case::in_extra_body(json!({"model": "gpt-4o", "extra_body": {"api_base": "https://attacker.example"}}), "api_base")]
    #[case::in_an_embedding_config(
        json!({"model": "gpt-4o", "litellm_embedding_config": {"api_base": "https://attacker.example"}}),
        "api_base"
    )]
    #[case::in_extra_body_sent_as_text(
        json!({"model": "gpt-4o", "extra_body": "{\"azure_ad_token\": \"caller\"}"}),
        "azure_ad_token"
    )]
    #[case::in_metadata(json!({"model": "gpt-4o", "metadata": {"langfuse_host": "https://attacker.example"}}), "langfuse_host")]
    #[case::in_metadata_sent_as_text(
        json!({"model": "gpt-4o", "litellm_metadata": "{\"langfuse_host\": \"https://attacker.example\"}"}),
        "langfuse_host"
    )]
    #[case::in_litellm_params_metadata(
        json!({"model": "gpt-4o", "litellm_params": {"metadata": {"langfuse_host": "https://attacker.example"}}}),
        "langfuse_host"
    )]
    fn an_override_is_refused_wherever_a_body_carries_it(
        names: Names,
        #[case] body: Value,
        #[case] param: &str,
    ) {
        assert_eq!(Case::new(body).run(&names), refused(param));
    }

    #[rstest]
    #[case::the_route(on("/introspect", json!({"client_id": "gateway"})), "client_id", Some(Admitted::ByRoute))]
    #[case::the_proxy_wide_opt_in(
        open(json!({"model": "gpt-4o", "api_base": "https://own.example"})),
        "api_base",
        Some(Admitted::ProxyWide)
    )]
    #[case::an_allowlist(
        Case {
            policy: Policy { s3_buckets: vec!["team-bucket".to_owned()], ..Policy::default() },
            ..Case::new(json!({"model": "gpt-4o", "s3_bucket_name": "team-bucket"}))
        },
        "s3_bucket_name",
        Some(Admitted::Allowlisted)
    )]
    #[case::the_handler_that_knows_the_teams_buckets(
        Case {
            deferred: Deferred { s3_buckets: true, ..Deferred::default() },
            ..Case::new(json!({"model": "gpt-4o", "s3_bucket_name": "team-bucket"}))
        },
        "s3_bucket_name",
        Some(Admitted::LeftToLaterCheck)
    )]
    #[case::the_check_that_runs_once_the_team_is_known(
        Case {
            deferred: Deferred { model_checks: true, ..Deferred::default() },
            ..Case::new(json!({"model": "azure-group", "client_id": "caller"}))
        },
        "client_id",
        Some(Admitted::LeftToLaterCheck)
    )]
    #[case::every_reachable_deployment(
        routed(&["client_id"], json!({"model": "azure-group", "client_id": "caller"})),
        "client_id",
        Some(Admitted::ByDeployments)
    )]
    #[case::the_route_before_the_proxy_wide_opt_in(
        Case { route: Some("/introspect"), ..open(json!({"client_id": "gateway"})) },
        "client_id",
        Some(Admitted::ByRoute)
    )]
    #[case::the_proxy_wide_opt_in_before_the_deployments(
        Case {
            policy: Policy { allow_client_side_credentials: true, ..Policy::default() },
            ..routed(&["client_id"], json!({"model": "azure-group", "client_id": "caller"}))
        },
        "client_id",
        Some(Admitted::ProxyWide)
    )]
    #[case::bucket_deferral_covers_buckets_alone(
        Case {
            deferred: Deferred { s3_buckets: true, ..Deferred::default() },
            ..Case::new(json!({"model": "gpt-4o", "api_base": "https://own.example"}))
        },
        "api_base",
        None
    )]
    #[case::no_one(Case::new(json!({"model": "gpt-4o", "api_base": "https://own.example"})), "api_base", None)]
    fn an_override_is_admitted_by_the_first_who_allows_it(
        names: Names,
        #[case] case: Case<'static>,
        #[case] param: &str,
        #[case] expected: Option<Admitted>,
    ) {
        assert_eq!(case.admitted(&names, param), expected);
    }

    #[rstest]
    fn a_form_spelled_metadata_entry_is_judged_as_metadata(names: Names) {
        let case = Case::new(
            json!({"model": "whisper-1", "metadata[langfuse_host]": "https://attacker.example"}),
        );

        assert_eq!(case.run(&names), refused("langfuse_host"));
    }

    #[rstest]
    #[case::dict_target(json!([{"model": "gpt-4o", "api_base": "https://attacker.example"}]), refused("api_base"))]
    #[case::url_valued_name(
        json!(["https://attacker.example/v1"]),
        Err(Rejection::UrlFallbackDestination { value: "https://attacker.example/v1".to_owned() })
    )]
    #[case::url_valued_dict_model(
        json!([{"model": "https://attacker.example/v1"}]),
        Err(Rejection::UrlFallbackDestination { value: "https://attacker.example/v1".to_owned() })
    )]
    #[case::model_names(json!(["gpt-4o-mini", {"model": "gpt-4o"}]), Ok(()))]
    fn a_fallback_target_is_held_to_the_rules_of_the_request(
        names: Names,
        #[case] fallback_targets: Value,
        #[case] expected: Result<(), Rejection>,
    ) {
        let case = Case::new(json!({"model": "gpt-4o", "fallbacks": fallback_targets}));

        assert_eq!(case.run(&names), expected);
    }

    #[rstest]
    #[case::in_extra_body(json!({"model": "azure-group", "extra_body": {"fallbacks": ["backup"]}}))]
    #[case::in_a_fallback_targets_extra_body(
        json!({"model": "azure-group", "fallbacks": [{"model": "azure-group", "extra_body": {"fallbacks": ["backup"]}}]})
    )]
    fn fallbacks_set_through_extra_body_need_a_grant(
        names: Names,
        #[case] body: Value,
        #[values(None, Some("fallbacks"))] granted: Option<&'static str>,
    ) {
        let case = routed(&granted.into_iter().collect::<Vec<_>>(), body);

        let expected = match granted {
            Some(_) => Ok(()),
            None => refused("fallbacks"),
        };
        assert_eq!(case.run(&names), expected);
    }

    #[rstest]
    #[case::at_the_limit(10, Ok(()))]
    #[case::past_it(11, Err(Rejection::FallbackNestingTooDeep))]
    fn fallbacks_nested_in_targets_are_refused_past_the_validation_depth(
        names: Names,
        #[case] depth: usize,
        #[case] expected: Result<(), Rejection>,
    ) {
        let nested = (0..depth).fold(
            json!(["gpt-4o-mini"]),
            |inner, _| json!([{"model": "gpt-4o", "fallbacks": inner}]),
        );
        let case = Case::new(json!({"model": "gpt-4o", "fallbacks": nested}));

        assert_eq!(case.run(&names), expected);
    }

    #[rstest]
    fn a_caller_is_told_every_model_and_credential_the_body_can_reach(names: Names) {
        let case = Case {
            route_model: Some("route-model"),
            ..Case::new(json!({
                "model": "gpt-4o",
                "api_base": "https://own.example",
                "fallbacks": ["backup", {"gpt-4o": ["keyed"]}],
                "router_settings_override": {"context_window_fallbacks": [{"model": "bigger"}]},
                "litellm_credential_name": "root-credential",
                "litellm_params": {"extra_body": {"litellm_credential_name": "nested-credential"}},
            }))
        };

        let needs = case.with_request(|request| needs(request, &names, &Policy::default()));

        assert_eq!(
            needs,
            Needs {
                models: ["backup", "bigger", "gpt-4o", "keyed", "route-model"]
                    .map(str::to_owned)
                    .into(),
                credentials: ["nested-credential", "root-credential"]
                    .map(str::to_owned)
                    .into(),
            }
        );
    }

    #[rstest]
    fn a_body_never_carries_a_model_list(names: Names) {
        let case = open(json!({"model": "gpt-4o", "model_list": []}));

        assert_eq!(case.run(&names), Err(Rejection::ModelList));
    }

    #[rstest]
    #[case::proxy_wide(open(json!({"model": "gpt-4o", "api_base": "https://own.example", "langfuse_host": "h"})))]
    #[case::deployment_opt_in(routed(&["client_id"], json!({"model": "azure-group", "client_id": "caller"})))]
    #[case::listed_bucket(Case {
        policy: Policy { s3_buckets: vec!["team-bucket".to_owned()], ..Policy::default() },
        ..Case::new(json!({"model": "gpt-4o", "s3_bucket_name": "team-bucket"}))
    })]
    #[case::listed_endpoint_url(Case {
        policy: Policy { s3_endpoint_urls: vec!["https://s3.own.example".to_owned()], ..Policy::default() },
        ..Case::new(json!({"model": "gpt-4o", "s3_endpoint_url": "https://s3.own.example"}))
    })]
    #[case::bucket_left_to_the_handler(Case {
        deferred: Deferred { s3_buckets: true, ..Deferred::default() },
        ..Case::new(json!({"model": "gpt-4o", "s3_output_bucket_name": "any"}))
    })]
    #[case::model_check_left_until_the_team_is_known(Case {
        deferred: Deferred { model_checks: true, ..Deferred::default() },
        ..Case::new(json!({"model": "gpt-4o", "api_base": "https://own.example"}))
    })]
    fn an_admins_grant_lets_an_override_through(names: Names, #[case] case: Case<'static>) {
        assert_eq!(case.run(&names), Ok(()));
    }

    #[rstest]
    #[case::another_param(routed(&["client_id"], json!({"model": "azure-group", "client_id": "caller", "client_secret": "s"})), "client_secret")]
    #[case::another_group(routed(&["client_id"], json!({"model": "other-group", "client_id": "caller"})), "client_id")]
    #[case::no_router(Case::new(json!({"model": "azure-group", "client_id": "caller"})), "client_id")]
    fn a_deployment_opt_in_covers_only_its_own_group_and_param(
        names: Names,
        #[case] case: Case<'static>,
        #[case] param: &str,
    ) {
        assert_eq!(case.run(&names), refused(param));
    }

    #[rstest]
    #[case::unlisted_bucket(
        Policy { s3_buckets: vec!["team-bucket".to_owned()], ..Policy::default() },
        "s3_bucket_name",
        Allowlist::Buckets
    )]
    #[case::unlisted_endpoint_url(Policy::default(), "s3_endpoint_url", Allowlist::EndpointUrls)]
    fn a_storage_override_outside_its_allowlist_is_refused_naming_the_setting(
        names: Names,
        #[case] policy: Policy,
        #[case] param: &str,
        #[case] allowlist: Allowlist,
    ) {
        let case = Case {
            policy,
            ..Case::new(json!({"model": "gpt-4o", param: "other"}))
        };

        assert_eq!(
            case.run(&names),
            Err(Rejection::Override {
                param: param.to_owned(),
                allowlist
            })
        );
    }

    #[rstest]
    #[case::entra_credential(open(json!({"model": "azure-group", "client_id": "os.environ/LITELLM_MASTER_KEY", "tenant_id": "t"})), "client_id")]
    #[case::oidc(open(json!({"model": "azure-group", "azure_ad_token": "oidc/env/LITELLM_MASTER_KEY"})), "azure_ad_token")]
    #[case::upstream_bound(open(json!({"model": "azure-group", "aws_access_key_id": "os.environ/LITELLM_MASTER_KEY"})), "aws_access_key_id")]
    #[case::in_extra_body(open(json!({"model": "azure-group", "extra_body": {"client_secret": "os.environ/LITELLM_MASTER_KEY"}})), "client_secret")]
    #[case::deployment_opted_in(
        routed(&["client_id"], json!({"model": "azure-group", "client_id": "os.environ/LITELLM_MASTER_KEY"})),
        "client_id"
    )]
    fn no_grant_lets_a_caller_have_a_secret_reference_resolved(
        names: Names,
        #[case] case: Case<'static>,
        #[case] param: &str,
    ) {
        assert_eq!(case.run(&names), reference(param));
    }

    #[rstest]
    #[case::root(json!({"model": "gpt-4o", "api_key": "oidc/google/aud"}))]
    #[case::extra_body(json!({"model": "gpt-4o", "extra_body": {"api_key": "oidc/google/aud"}}))]
    #[case::litellm_params_extra_body(json!({"model": "gpt-4o", "litellm_params": {"extra_body": {"api_key": "oidc/google/aud"}}}))]
    fn an_oidc_api_key_is_refused_as_the_proxys_own_identity(names: Names, #[case] body: Value) {
        assert_eq!(open(body).run(&names), Err(Rejection::ApiKeyOidcReference));
    }

    #[rstest]
    #[case::unlisted_param(json!({"model": "gpt-4o", "some_provider_param": "os.environ/LITELLM_MASTER_KEY"}), "some_provider_param")]
    #[case::below_an_unlisted_param(
        json!({"model": "gpt-4o", "provider_options": {"auth": ["plain", {"token": "oidc/env/LITELLM_MASTER_KEY"}]}}),
        "provider_options"
    )]
    #[case::unlisted_param_in_extra_body(json!({"model": "gpt-4o", "extra_body": {"some_provider_param": "os.environ/HOME"}}), "some_provider_param")]
    #[case::below_litellm_params(json!({"model": "gpt-4o", "litellm_params": {"anything": "os.environ/HOME"}}), "litellm_params")]
    #[case::below_router_settings(json!({"model": "gpt-4o", "router_settings_override": {"timeout": "os.environ/HOME"}}), "router_settings_override")]
    #[case::in_a_fallback_target(json!({"model": "gpt-4o", "fallbacks": [{"model": "gpt-4o", "some_provider_param": "os.environ/HOME"}]}), "some_provider_param")]
    fn a_model_call_carries_no_secret_reference_outside_its_content(
        names: Names,
        #[case] body: Value,
        #[case] param: &str,
    ) {
        assert_eq!(open(body).run(&names), reference(param));
    }

    #[rstest]
    #[case::connection_setting(json!({"redis_password": "os.environ/LITELLM_MASTER_KEY"}), "redis_password")]
    #[case::below_a_setting(json!({"cache_kwargs": {"password": "oidc/env/LITELLM_MASTER_KEY"}}), "cache_kwargs")]
    #[case::beside_a_deployment(
        json!({
            "model_list": [{"litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-own"}}],
            "redis_url": "os.environ/LITELLM_MASTER_KEY",
        }),
        "redis_url"
    )]
    #[case::sent_as_text(json!("{\"redis_password\": \"os.environ/LITELLM_MASTER_KEY\"}"), "redis_password")]
    fn a_caller_supplied_router_carries_no_secret_reference_in_any_setting(
        names: Names,
        #[case] user_config: Value,
        #[case] param: &str,
    ) {
        let case = open(json!({"model": "caller-model", "user_config": user_config}));

        assert_eq!(case.run(&names), reference(param));
    }

    #[rstest]
    fn a_caller_supplied_router_may_carry_the_callers_own_settings(names: Names) {
        let case = open(json!({
            "model": "caller-model",
            "user_config": {
                "model_list": [{"litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-own"}}],
                "redis_host": "cache.own.example",
                "redis_password": "own-password",
            },
        }));

        assert_eq!(case.run(&names), Ok(()));
    }

    #[rstest]
    #[case::provider_param("some_provider_param", Err(Rejection::TooDeep { param: "some_provider_param".to_owned() }))]
    #[case::metadata("metadata", Ok(()))]
    #[case::content("messages", Ok(()))]
    fn a_parameter_nested_below_the_depth_a_host_reads_is_refused_where_a_reference_would_be(
        names: Names,
        #[case] key: &str,
        #[case] expected: Result<(), Rejection>,
    ) {
        let body = Params::from_iter([
            ("model".to_owned(), CallerValue::Text("gpt-4o".to_owned())),
            (
                key.to_owned(),
                CallerValue::List(vec![CallerValue::TooDeep]),
            ),
        ]);
        let request = Request {
            body: &body,
            surface: Surface::of(Some(CHAT)),
            model: "gpt-4o",
            route_model: None,
            deferred: Deferred::default(),
        };

        assert_eq!(
            check(
                &request,
                &Reachable::default(),
                &Credentials::default(),
                &names,
                &Policy::default()
            ),
            expected
        );
    }

    #[rstest]
    #[case::prompt(open(json!({"model": "gpt-4o", "prompt": "os.environ/PATH is what in Python?"})))]
    #[case::message(open(json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "os.environ/HOME"}]})))]
    #[case::tool_schema(open(json!({"model": "gpt-4o", "tools": [{"function": {"parameters": {"default": "os.environ/OPENAI_API_KEY"}}}]})))]
    #[case::litellm_params_metadata_tag(open(json!({"model": "gpt-4o", "litellm_params": {"metadata": {"note": "os.environ/HOME"}}})))]
    #[case::mentioned_later_in_an_unlisted_param(open(json!({"model": "gpt-4o", "some_provider_param": "read os.environ/HOME"})))]
    #[case::fallback_model_name(open(json!({"model": "gpt-4o", "fallbacks": ["os.environ/HOME"]})))]
    #[case::fallback_model_name_in_router_settings(
        open(json!({"model": "gpt-4o", "router_settings_override": {"fallbacks": [{"gpt-4o": ["os.environ/HOME"]}]}}))
    )]
    #[case::key_management_route(on("/key/generate", json!({"key_alias": "os.environ/HOME"})))]
    #[case::credential_route(on("/credentials", json!({"credential_values": {"api_key": "os.environ/AZURE_API_KEY"}})))]
    #[case::metadata_tag(open(json!({"model": "gpt-4o", "metadata": {"client_id": "os.environ/HOME", "note": "oidc/azure/team-a"}})))]
    #[case::mentioned_later(open(json!({"model": "gpt-4o", "client_id": "read os.environ/HOME"})))]
    #[case::callers_own_key(open(json!({"model": "gpt-4o", "api_key": "sk-caller-own-key"})))]
    fn text_that_no_provider_resolves_may_look_like_a_secret_reference(
        names: Names,
        #[case] case: Case<'static>,
    ) {
        assert_eq!(case.run(&names), Ok(()));
    }

    #[rstest]
    #[case::deployment_secret_reference(on("/model/new", json!({"model_name": "azure-group", "api_key": "os.environ/AZURE_API_KEY"})), Ok(()))]
    #[case::deployment_update_by_id(on("/model/abc/update", json!({"litellm_params": {"api_key": "os.environ/AZURE_API_KEY"}})), Ok(()))]
    #[case::transcription_session_secret(on("/v1/realtime/transcription_sessions", json!({"client_secret": {"expires_after": {}}})), Ok(()))]
    #[case::transcription_session_other_entra(on("/v1/realtime/transcription_sessions", json!({"client_id": "caller"})), refused("client_id"))]
    #[case::mcp_oauth_client(on("/v1/mcp/server", json!({"client_id": "id", "client_secret": "secret"})), Ok(()))]
    #[case::mcp_token_exchange(on("/v1/mcp/server", json!({"token_exchange_endpoint": "https://idp.example/token"})), Ok(()))]
    #[case::mcp_other_entra(on("/v1/mcp/server", json!({"tenant_id": "t"})), refused("tenant_id"))]
    #[case::mcp_other_federation(
        on("/v1/mcp/server", json!({"anthropic_issuer_url": "https://idp.example"})),
        Err(Rejection::ServerOwned { param: "anthropic_issuer_url".to_owned(), kind: ServerOwnedKind::Federation })
    )]
    #[case::deployment_chatgpt_auth_file(on("/model/new", json!({"litellm_params": {"chatgpt_auth_file": "auth.json"}})), Ok(()))]
    #[case::deployment_other_federation(
        on("/model/new", json!({"litellm_params": {"anthropic_issuer_url": "https://idp.example"}})),
        Err(Rejection::ServerOwned { param: "anthropic_issuer_url".to_owned(), kind: ServerOwnedKind::Federation })
    )]
    #[case::introspection_client(on("/introspect", json!({"client_id": "id", "client_secret": "secret"})), Ok(()))]
    #[case::introspection_secret_reference(on("/introspect", json!({"client_id": "os.environ/LITELLM_MASTER_KEY"})), reference("client_id"))]
    #[case::pass_through_entra(on("/azure/openai/deployments/d/chat/completions", json!({"client_id": "caller"})), refused("client_id"))]
    #[case::pass_through_tenant(on("/vertex_ai/v1/projects/p", json!({"tenant_id": "t"})), refused("tenant_id"))]
    #[case::vertex_live_target(on("/vertex_ai/live", json!({"vertex_project": "p", "vertex_location": "l"})), Ok(()))]
    #[case::vertex_live_other_override(on("/vertex_ai/live", json!({"vertex_project": "p", "api_base": "https://attacker.example"})), refused("api_base"))]
    fn a_route_decides_which_names_are_its_own_fields(
        names: Names,
        #[case] case: Case<'static>,
        #[case] expected: Result<(), Rejection>,
    ) {
        assert_eq!(case.run(&names), expected);
    }

    #[rstest]
    #[case::root(json!({"model": "gpt-4o", "anthropic_issuer_url": "https://idp.example"}), "anthropic_issuer_url", ServerOwnedKind::Federation)]
    #[case::form_list_spelling(json!({"model": "gpt-4o", "anthropic_issuer_url[]": "https://idp.example"}), "anthropic_issuer_url", ServerOwnedKind::Federation)]
    #[case::token_exchange_outside_mcp(json!({"model": "gpt-4o", "token_exchange_endpoint": "https://idp.example"}), "token_exchange_endpoint", ServerOwnedKind::Federation)]
    #[case::github_copilot(json!({"model": "gpt-4o", "github_copilot_app_id": 1}), "github_copilot_app_id", ServerOwnedKind::GitHubCopilot)]
    #[case::in_litellm_params(json!({"model": "gpt-4o", "litellm_params": {"anthropic_issuer_url": "https://idp.example"}}), "anthropic_issuer_url", ServerOwnedKind::Federation)]
    #[case::in_a_user_config_deployment(
        json!({"model": "m", "user_config": {"model_list": [{"litellm_params": {"anthropic_issuer_url": "https://idp.example"}}]}}),
        "anthropic_issuer_url",
        ServerOwnedKind::Federation
    )]
    fn a_federation_setting_is_refused_despite_a_proxy_wide_opt_in(
        names: Names,
        #[case] body: Value,
        #[case] param: &str,
        #[case] kind: ServerOwnedKind,
    ) {
        assert_eq!(
            open(body).run(&names),
            Err(Rejection::ServerOwned {
                param: param.to_owned(),
                kind
            })
        );
    }

    #[rstest]
    #[case::federation_field(json!({"anthropic_issuer_url": "https://idp.example"}), "anthropic_issuer_url")]
    #[case::oidc_api_key(json!({"api_key": "oidc/google/aud"}), "api_key")]
    fn a_body_cannot_name_a_federated_credential(
        names: Names,
        #[case] federated_credential: Value,
        #[case] field: &str,
    ) {
        let case = Case {
            proxy: Proxy {
                federated_credential,
                ..Proxy::default()
            },
            ..open(json!({"model": "gpt-4o", "litellm_credential_name": "federated"}))
        };

        assert_eq!(
            case.run(&names),
            Err(Rejection::FederatedCredential {
                named: "federated".to_owned(),
                field: field.to_owned()
            })
        );
    }

    #[rstest]
    fn a_caller_supplied_deployment_cannot_name_a_federated_credential(names: Names) {
        let case = Case {
            proxy: Proxy {
                federated_credential: json!({"anthropic_issuer_url": "https://idp.example"}),
                ..Proxy::default()
            },
            ..open(json!({
                "model": "caller-model",
                "user_config": {"model_list": [{"litellm_params": {"litellm_credential_name": "federated"}}]},
            }))
        };

        assert_eq!(
            case.run(&names),
            Err(Rejection::FederatedCredential {
                named: "federated".to_owned(),
                field: "anthropic_issuer_url".to_owned()
            })
        );
    }

    #[rstest]
    fn a_body_may_name_a_credential_that_holds_plain_values(names: Names) {
        let case = Case {
            proxy: Proxy {
                federated_credential: json!({"api_key": "sk-stored", "anthropic_issuer_url": null}),
                ..Proxy::default()
            },
            ..open(json!({"model": "gpt-4o", "litellm_credential_name": "federated"}))
        };

        assert_eq!(case.run(&names), Ok(()));
    }

    #[rstest]
    #[case::listed_field(json!({"model_list": [{"litellm_params": {"model": "openai/gpt-4o", "api_key": "os.environ/LITELLM_MASTER_KEY"}}]}), "api_key")]
    #[case::unlisted_field(json!({"model_list": [{"litellm_params": {"model": "openai/gpt-4o", "anything": "os.environ/LITELLM_MASTER_KEY"}}]}), "anything")]
    #[case::defaults(json!({"default_litellm_params": {"organization": "os.environ/LITELLM_MASTER_KEY"}}), "organization")]
    #[case::search_tool(json!({"search_tools": [{"litellm_params": {"api_key": "os.environ/LITELLM_MASTER_KEY"}}]}), "api_key")]
    #[case::nested_config(json!({"model_list": [{"litellm_params": {"extra_body": {"token": "os.environ/LITELLM_MASTER_KEY"}}}]}), "token")]
    #[case::data_source(json!({"model_list": [{"litellm_params": {"dataSources": [{"parameters": {"key": "os.environ/LITELLM_MASTER_KEY"}}]}}]}), "key")]
    #[case::entry_sent_as_text(json!({"model_list": ["{\"litellm_params\": {\"api_key\": \"os.environ/LITELLM_MASTER_KEY\"}}"]}), "api_key")]
    fn every_field_of_a_caller_supplied_deployment_is_checked_for_a_reference(
        names: Names,
        #[case] user_config: Value,
        #[case] param: &str,
    ) {
        let case = open(json!({"model": "caller-model", "user_config": user_config}));

        assert_eq!(case.run(&names), reference(param));
    }

    #[rstest]
    fn a_caller_supplied_deployment_may_carry_the_callers_own_credentials(names: Names) {
        let case = open(json!({
            "model": "caller-model",
            "user_config": {"model_list": [{"litellm_params": {
                "model": "openai/gpt-4o", "api_base": "https://own.example", "api_key": "sk-own"
            }}]},
        }));

        assert_eq!(case.run(&names), Ok(()));
    }

    #[rstest]
    #[case::at_the_limit(MAX_OPT_IN_FALLBACK_MODELS, refused("api_base"))]
    #[case::past_the_limit(
        MAX_OPT_IN_FALLBACK_MODELS + 1,
        Err(Rejection::TooManyFallbackModels { limit: MAX_OPT_IN_FALLBACK_MODELS })
    )]
    fn fallback_models_are_checked_up_to_the_limit_and_refused_past_it(
        names: Names,
        #[case] count: usize,
        #[case] expected: Result<(), Rejection>,
    ) {
        let models: Vec<String> = (0..count).map(|index| format!("model-{index}")).collect();
        let case = Case::new(
            json!({"model": "gpt-4o", "api_base": "https://own.example", "fallbacks": models}),
        );

        assert_eq!(case.run(&names), expected);
    }

    #[rstest]
    #[case::another_group(Some("other-group"), refused("client_id"))]
    #[case::the_same_group(Some("azure-group"), Ok(()))]
    #[case::no_model(Some(""), Ok(()))]
    fn a_model_named_in_the_route_must_opt_in_too(
        names: Names,
        #[case] route_model: Option<&'static str>,
        #[case] expected: Result<(), Rejection>,
    ) {
        let case = Case {
            route_model,
            ..routed(
                &["client_id"],
                json!({"model": "azure-group", "client_id": "caller"}),
            )
        };

        assert_eq!(case.run(&names), expected);
    }

    /// Where a body can carry the parameters of a call.
    #[derive(Clone, Copy, Debug)]
    enum Place {
        Root,
        ExtraBody,
        ExtraBodyAsText,
        EmbeddingConfig,
        Metadata,
        LitellmMetadataAsText,
        FallbackTarget,
        LitellmParams,
        LitellmParamsExtraBody,
        LitellmParamsMetadata,
        UserConfigDeployment,
        UserConfigDefaults,
        UserConfigSearchTool,
    }

    /// `carried` at `place`, on a proxy that lets callers bring their own credentials and holds
    /// a federated credential named `federated`.
    fn placed(place: Place, carried: &Value) -> Case<'static> {
        let with_model = |model: &str| {
            let mut params = carried.clone();
            params["model"] = json!(model);
            params
        };
        let body = match place {
            Place::Root => with_model("gpt-4o"),
            Place::ExtraBody => json!({"model": "gpt-4o", "extra_body": carried}),
            Place::ExtraBodyAsText => json!({"model": "gpt-4o", "extra_body": carried.to_string()}),
            Place::EmbeddingConfig => {
                json!({"model": "gpt-4o", "litellm_embedding_config": carried})
            }
            Place::Metadata => json!({"model": "gpt-4o", "metadata": carried}),
            Place::LitellmMetadataAsText => {
                json!({"model": "gpt-4o", "litellm_metadata": carried.to_string()})
            }
            Place::FallbackTarget => {
                json!({"model": "gpt-4o", "fallbacks": [with_model("backup")]})
            }
            Place::LitellmParams => json!({"model": "gpt-4o", "litellm_params": carried}),
            Place::LitellmParamsExtraBody => {
                json!({"model": "gpt-4o", "litellm_params": {"extra_body": carried}})
            }
            Place::LitellmParamsMetadata => {
                json!({"model": "gpt-4o", "litellm_params": {"metadata": carried}})
            }
            Place::UserConfigDeployment => {
                json!({"model": "m", "user_config": {"model_list": [{"litellm_params": carried}]}})
            }
            Place::UserConfigDefaults => {
                json!({"model": "m", "user_config": {"default_litellm_params": carried}})
            }
            Place::UserConfigSearchTool => {
                json!({"model": "m", "user_config": {"search_tools": [{"litellm_params": carried}]}})
            }
        };
        Case {
            proxy: Proxy {
                federated_credential: json!({"anthropic_issuer_url": "https://idp.example"}),
                ..Proxy::default()
            },
            ..open(body)
        }
    }

    #[derive(Clone, Copy)]
    enum Verdict {
        Accepted,
        /// Refused with the refusal `carried` pairs with what was sent.
        Refused,
        /// Refused for a secret reference, named by the root key it sits below.
        RefusedBelow(&'static str),
    }
    use Verdict::{Accepted as ACCEPTED, Refused as REFUSED};
    const BELOW_LITELLM_PARAMS: Verdict = Verdict::RefusedBelow("litellm_params");

    /// What a caller must never set, and the refusal each earns where it is refused.
    fn carried() -> [(Value, Rejection); 5] {
        [
            (
                json!({"anthropic_issuer_url": "https://idp.example"}),
                Rejection::ServerOwned {
                    param: "anthropic_issuer_url".to_owned(),
                    kind: ServerOwnedKind::Federation,
                },
            ),
            (
                json!({"api_key": "oidc/google/aud"}),
                Rejection::ApiKeyOidcReference,
            ),
            (
                json!({"litellm_credential_name": "federated"}),
                Rejection::FederatedCredential {
                    named: "federated".to_owned(),
                    field: "anthropic_issuer_url".to_owned(),
                },
            ),
            (
                json!({"api_key": "os.environ/LITELLM_MASTER_KEY"}),
                secret_reference("api_key"),
            ),
            (
                json!({"anything": "os.environ/LITELLM_MASTER_KEY"}),
                secret_reference("anything"),
            ),
        ]
    }

    // Columns, in the order of `carried`: a federation setting, an `oidc/` api_key, the name of a
    // federated credential, a reference in a parameter providers resolve, a reference in any other.
    #[rstest]
    #[case::root(Place::Root, [REFUSED, REFUSED, REFUSED, REFUSED, REFUSED])]
    #[case::extra_body(Place::ExtraBody, [REFUSED, REFUSED, REFUSED, REFUSED, REFUSED])]
    #[case::extra_body_as_text(Place::ExtraBodyAsText, [REFUSED, REFUSED, REFUSED, REFUSED, REFUSED])]
    #[case::embedding_config(Place::EmbeddingConfig, [REFUSED, REFUSED, REFUSED, REFUSED, REFUSED])]
    #[case::fallback_target(Place::FallbackTarget, [REFUSED, REFUSED, REFUSED, REFUSED, REFUSED])]
    #[case::metadata(Place::Metadata, [REFUSED, REFUSED, REFUSED, ACCEPTED, ACCEPTED])]
    #[case::litellm_metadata_as_text(Place::LitellmMetadataAsText, [REFUSED, REFUSED, REFUSED, ACCEPTED, ACCEPTED])]
    #[case::litellm_params(Place::LitellmParams, [REFUSED, REFUSED, REFUSED, BELOW_LITELLM_PARAMS, BELOW_LITELLM_PARAMS])]
    #[case::litellm_params_metadata(Place::LitellmParamsMetadata, [REFUSED, REFUSED, REFUSED, ACCEPTED, ACCEPTED])]
    #[case::litellm_params_extra_body(
        Place::LitellmParamsExtraBody,
        [ACCEPTED, REFUSED, ACCEPTED, BELOW_LITELLM_PARAMS, BELOW_LITELLM_PARAMS]
    )]
    #[case::user_config_deployment(Place::UserConfigDeployment, [REFUSED, REFUSED, REFUSED, REFUSED, REFUSED])]
    #[case::user_config_defaults(Place::UserConfigDefaults, [REFUSED, REFUSED, REFUSED, REFUSED, REFUSED])]
    #[case::user_config_search_tool(Place::UserConfigSearchTool, [ACCEPTED, REFUSED, ACCEPTED, REFUSED, REFUSED])]
    fn each_place_a_body_carries_parameters_is_judged_by_its_own_rules(
        names: Names,
        #[case] place: Place,
        #[case] verdicts: [Verdict; 5],
    ) {
        let judged: Vec<Result<(), Rejection>> = carried()
            .iter()
            .map(|(params, _)| placed(place, params).run(&names))
            .collect();
        let expected: Vec<Result<(), Rejection>> = carried()
            .into_iter()
            .zip(verdicts)
            .map(|((_, rejection), verdict)| match verdict {
                Verdict::Accepted => Ok(()),
                Verdict::Refused => Err(rejection),
                Verdict::RefusedBelow(key) => reference(key),
            })
            .collect();

        assert_eq!(judged, expected);
    }

    #[rstest]
    #[case::override_param(
        Err(Rejection::Override { param: "s3_bucket_name".to_owned(), allowlist: Allowlist::Buckets }),
        "Rejected Request: s3_bucket_name is not allowed in request body. Clientside passthrough requires explicit \
admin opt-in via either `general_settings.allow_client_side_credentials = true` (proxy-wide) or \
`configurable_clientside_auth_params` on the deployment in your proxy config.yaml. Relevant Issue: \
https://huntr.com/bounties/4001e1a2-7b7a-4776-a3ae-e6692ec3d997 To allow specific values only, list them in \
`general_settings.allowed_client_side_s3_buckets` or in the team's `metadata.allowed_s3_buckets`."
    )]
    #[case::secret_reference(
        reference("client_id"),
        "Rejected Request: client_id holds a secret reference, which only a deployment or credential a proxy admin \
configures can carry."
    )]
    #[case::server_owned(
        Err(Rejection::ServerOwned { param: "github_copilot_app_id".to_owned(), kind: ServerOwnedKind::GitHubCopilot }),
        "Rejected Request: github_copilot_app_id is a server-owned GitHub Copilot authentication parameter and cannot \
be set in a request body. A proxy admin configures it on the deployment or on a stored credential."
    )]
    fn a_rejection_reads_as_the_proxy_has_always_worded_it(
        #[case] rejection: Result<(), Rejection>,
        #[case] expected: &str,
    ) {
        assert_eq!(
            rejection.map_err(|rejection| rejection.to_string()),
            Err(expected.to_owned())
        );
    }
}
