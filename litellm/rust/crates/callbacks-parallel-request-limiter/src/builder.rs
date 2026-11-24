use std::{cmp::Ordering, collections::BTreeMap};

use litellm_python_compat::truthy::json_truthy;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::{
    arithmetic::{ArithmeticError, AuthFacts, auth_descriptors},
    descriptor::{PROJECT_ITPM, PROJECT_OTPM, PTU_TEAM, python_text},
    ptu::Ceiling,
    python::{hash_error, stringified, type_name},
    state::RateLimitedModel,
};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BuildError {
    #[error(transparent)]
    Arithmetic(#[from] ArithmeticError),
    #[error("'<' not supported between the limits of one {0} descriptor")]
    Incomparable(String),
    #[error("the key limits do not decode: {0}")]
    Auth(String),
    #[error("{exception}: {message}")]
    Python { exception: String, message: String },
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct AuthMetadata {
    pub key_model_rpm_limit: Value,
    pub key_model_tpm_limit: Value,
    pub key_tag_rpm_limit: Value,
    pub model_max_budget: Value,
    pub team_model_rpm_limit: Value,
    pub team_model_tpm_limit: Value,
    pub organization_model_rpm_limit: Value,
    pub organization_model_tpm_limit: Value,
    pub project_model_rpm_limit: Value,
    pub project_model_tpm_limit: Value,
    pub project_model_itpm_limit: Value,
    pub project_model_otpm_limit: Value,
    pub end_user_model_rpm_limit: Value,
    pub end_user_model_tpm_limit: Value,
    /// Read from the typed `key_limits` record, not the metadata input.
    #[serde(skip)]
    pub customer_scopes: Option<Vec<CustomerScope>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CustomerScope {
    pub key: String,
    pub entity_id: String,
    pub rpm_limit: Value,
    pub tpm_limit: Value,
    pub model_rpm_limit: Value,
    pub model_tpm_limit: Value,
    /// A customer organization listed to borrow idle capacity (`borrowing_customer_org_model_limits`).
    pub borrows: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct AgentPolicy {
    pub rpm_limit: Value,
    pub tpm_limit: Value,
    pub session_rpm_limit: Value,
    pub session_tpm_limit: Value,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct DeploymentLimits {
    pub rpm: Option<i64>,
    pub tpm: Option<i64>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct Lookups {
    pub candidates: Vec<String>,
    pub groups: BTreeMap<String, Option<String>>,
    pub deployment_limits: BTreeMap<String, DeploymentLimits>,
    pub borrows: bool,
    pub borrows_key: bool,
    pub dynamic_covered: Vec<String>,
    pub model_has_failures: bool,
    pub ptu_ceiling: Option<Ceiling>,
    pub tag_limits: Vec<(Value, Value, Value)>,
    #[serde(skip)]
    pub managed_agent_policy: Option<AgentPolicy>,
    #[serde(skip)]
    pub invoked_agent_policy: Option<AgentPolicy>,
    pub registry_agents: Vec<(Value, Option<AgentPolicy>)>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct RequestFacts {
    pub tags: Vec<Value>,
    pub resolved_agent_id: Value,
    pub resolved_agent_error: Option<(String, String)>,
    pub session_id: Option<String>,
    pub session_error: Option<(String, String)>,
}

pub struct Request<'a> {
    pub auth: &'a Map<String, Value>,
    pub rpm_limit_type: Option<&'a str>,
    pub tpm_limit_type: Option<&'a str>,
    pub metadata: &'a AuthMetadata,
    pub lookups: &'a Lookups,
    pub request: &'a RequestFacts,
    pub requested_model: Option<&'a str>,
    pub window_size: i64,
}

pub struct Built {
    pub descriptors: Vec<Value>,
    pub ptu_ceiling: Option<Ceiling>,
}

fn truthy(value: &Value) -> bool {
    if let Some(found) = stringified(value) {
        return found.truthy;
    }
    json_truthy(value)
}

fn text(value: &Value) -> String {
    python_text(value).unwrap_or_else(|| value.to_string())
}

fn limit_in(limits: &Value, model: &RateLimitedModel) -> Result<Value, BuildError> {
    let limits = match limits {
        Value::Null => return Ok(Value::Null),
        Value::Object(limits) => limits,
        other => {
            return Err(BuildError::Python {
                exception: "AttributeError".to_owned(),
                message: format!("'{}' object has no attribute 'get'", type_name(other)),
            });
        }
    };
    Ok(match limits.get(&model.requested) {
        Some(found) if !found.is_null() => found.clone(),
        _ => limits.get(&model.group).cloned().unwrap_or(Value::Null),
    })
}

fn descriptor(key: &str, value: Value, requests: Value, tokens: Value, window_size: i64) -> Value {
    json!({
        "key": key,
        "value": value,
        "rate_limit": {
            "requests_per_unit": requests,
            "tokens_per_unit": tokens,
            "window_size": window_size,
        },
    })
}

fn compare(left: &Value, right: &Value) -> Option<Ordering> {
    let number = |value: &Value| match value {
        Value::Bool(flag) => Some(f64::from(u8::from(*flag))),
        Value::Number(number) => number.as_f64(),
        _ => None,
    };
    match (left, right) {
        (Value::String(left), Value::String(right)) => Some(left.cmp(right)),
        _ => number(left)?.partial_cmp(&number(right)?),
    }
}

fn minimum(descriptors: &[&Value], metric: &str) -> Result<Value, BuildError> {
    let mut least: Option<&Value> = None;
    for descriptor in descriptors {
        let Some(limit) = descriptor["rate_limit"]
            .get(metric)
            .filter(|limit| !limit.is_null())
        else {
            continue;
        };
        least = Some(match least {
            None => limit,
            Some(current) => match compare(limit, current) {
                Some(Ordering::Less) => limit,
                Some(_) => current,
                None => {
                    return Err(BuildError::Incomparable(text(&descriptor["key"])));
                }
            },
        });
    }
    Ok(least.cloned().unwrap_or(Value::Null))
}

fn identity(descriptor: &Value) -> (Value, Value) {
    (descriptor["key"].clone(), descriptor["value"].clone())
}

fn raised(exception: &str, message: impl Into<String>) -> BuildError {
    BuildError::Python {
        exception: exception.to_owned(),
        message: message.into(),
    }
}

fn hashed(values: &[&Value]) -> Result<(), BuildError> {
    match values.iter().find_map(|value| hash_error(value)) {
        Some(message) => Err(raised("TypeError", message)),
        None => Ok(()),
    }
}

pub fn coalesce(descriptors: Vec<Value>) -> Result<Vec<Value>, BuildError> {
    hashed(
        &descriptors
            .iter()
            .map(|descriptor| &descriptor["value"])
            .collect::<Vec<_>>(),
    )?;
    let mut identities: Vec<(Value, Value)> = Vec::new();
    for descriptor in &descriptors {
        let found = identity(descriptor);
        if !identities.contains(&found) {
            identities.push(found);
        }
    }
    if identities.len() == descriptors.len() {
        return Ok(descriptors);
    }
    identities
        .into_iter()
        .map(|found| {
            let group: Vec<&Value> = descriptors
                .iter()
                .filter(|descriptor| identity(descriptor) == found)
                .collect();
            let first = group[0];
            if group.len() == 1 {
                return Ok(first.clone());
            }
            Ok(json!({
                "key": first["key"],
                "value": first["value"],
                "rate_limit": {
                    "requests_per_unit": minimum(&group, "requests_per_unit")?,
                    "tokens_per_unit": minimum(&group, "tokens_per_unit")?,
                    "max_parallel_requests": minimum(&group, "max_parallel_requests")?,
                    "window_size": first["rate_limit"].get("window_size").cloned().unwrap_or(Value::Null),
                },
            }))
        })
        .collect()
}

impl Lookups {
    pub fn rate_limited_model(&self, requested: Option<&str>) -> Option<RateLimitedModel> {
        let requested = requested.filter(|requested| !requested.is_empty())?;
        let group = self
            .groups
            .get(requested)
            .and_then(|group| group.as_deref())
            .filter(|group| !group.is_empty())
            .unwrap_or(requested);
        Some(RateLimitedModel {
            requested: requested.to_owned(),
            group: group.to_owned(),
        })
    }
}

impl Request<'_> {
    fn attribute(&self, name: &str) -> &Value {
        self.auth.get(name).unwrap_or(&Value::Null)
    }

    fn model(&self, candidate: &str) -> Option<RateLimitedModel> {
        self.lookups.rate_limited_model(Some(candidate))
    }

    fn key_own_model_limit(&self, metadata_limit: &Value, budget_key: &str) -> Value {
        if truthy(metadata_limit) {
            return metadata_limit.clone();
        }
        let Value::Object(budgets) = &self.metadata.model_max_budget else {
            return Value::Null;
        };
        let limits: Map<String, Value> = budgets
            .iter()
            .filter_map(|(model, budget)| {
                let limit = budget.as_object()?.get(budget_key)?;
                (!limit.is_null()).then(|| (model.clone(), limit.clone()))
            })
            .collect();
        if limits.is_empty() {
            Value::Null
        } else {
            Value::Object(limits)
        }
    }

    fn key_own_rpm(&self) -> Value {
        self.key_own_model_limit(&self.metadata.key_model_rpm_limit, "rpm_limit")
    }

    fn key_own_tpm(&self) -> Value {
        self.key_own_model_limit(&self.metadata.key_model_tpm_limit, "tpm_limit")
    }

    fn key_model_limit(
        &self,
        own: Value,
        team: &Value,
        default: Option<i64>,
        group: &str,
    ) -> Value {
        if !own.is_null() {
            return own;
        }
        if !team.is_null() {
            return team.clone();
        }
        default.map_or(Value::Null, |limit| json!({ group: limit }))
    }

    fn inherited_team_limit(
        &self,
        model: &RateLimitedModel,
        team: &Value,
        own: Value,
    ) -> Result<Value, BuildError> {
        let team_limit = limit_in(team, model)?;
        if team_limit.is_null() || !limit_in(&own, model)?.is_null() {
            return Ok(Value::Null);
        }
        Ok(team_limit)
    }

    fn inherited_team_rpm(&self, model: &RateLimitedModel) -> Result<Value, BuildError> {
        self.inherited_team_limit(
            model,
            &self.metadata.team_model_rpm_limit,
            self.key_own_rpm(),
        )
    }

    fn inherited_team_tpm(&self, model: &RateLimitedModel) -> Result<Value, BuildError> {
        self.inherited_team_limit(
            model,
            &self.metadata.team_model_tpm_limit,
            self.key_own_tpm(),
        )
    }

    fn borrowing(&self, model: &RateLimitedModel) -> Result<(bool, bool), BuildError> {
        if !self.lookups.borrows || self.attribute("team_id").is_null() {
            return Ok((false, false));
        }
        let rpm = !self.inherited_team_rpm(model)?.is_null();
        let tpm = !self.inherited_team_tpm(model)?.is_null();
        if (!rpm && !tpm) || !self.lookups.dynamic_covered.contains(&model.group) {
            return Ok((false, false));
        }
        Ok((rpm, tpm))
    }

    /// The key's own per-model limits are shares the dynamic limiter enforces at saturation when keys borrow
    /// idle capacity (`borrowing_key_model_limits`).
    fn key_borrowing(&self, model: &RateLimitedModel) -> Result<(bool, bool), BuildError> {
        if !self.lookups.borrows_key || self.attribute("api_key").is_null() {
            return Ok((false, false));
        }
        let rpm = !limit_in(&self.key_own_rpm(), model)?.is_null();
        let tpm = !limit_in(&self.key_own_tpm(), model)?.is_null();
        if (!rpm && !tpm) || !self.lookups.dynamic_covered.contains(&model.group) {
            return Ok((false, false));
        }
        Ok((rpm, tpm))
    }

    fn model_per_key(&self, candidate: &str) -> Result<Option<Value>, BuildError> {
        let Some(model) = self.model(candidate) else {
            return Ok(None);
        };
        let (team_rpm, team_tpm) = self.borrowing(&model)?;
        let (key_rpm, key_tpm) = self.key_borrowing(&model)?;
        let (borrows_rpm, borrows_tpm) = (team_rpm || key_rpm, team_tpm || key_tpm);
        let defaults = self
            .lookups
            .deployment_limits
            .get(&model.group)
            .cloned()
            .unwrap_or_default();
        let tokens = if borrows_tpm {
            Value::Null
        } else {
            limit_in(
                &self.key_model_limit(
                    self.key_own_tpm(),
                    &self.metadata.team_model_tpm_limit,
                    defaults.tpm,
                    &model.group,
                ),
                &model,
            )?
        };
        let requests = if borrows_rpm {
            Value::Null
        } else {
            limit_in(
                &self.key_model_limit(
                    self.key_own_rpm(),
                    &self.metadata.team_model_rpm_limit,
                    defaults.rpm,
                    &model.group,
                ),
                &model,
            )?
        };
        if tokens.is_null() && requests.is_null() {
            return Ok(None);
        }
        Ok(Some(descriptor(
            "model_per_key",
            Value::String(format!(
                "{}:{}",
                text(self.attribute("api_key")),
                model.group
            )),
            requests,
            tokens,
            self.window_size,
        )))
    }

    fn model_per_end_user(&self, candidate: &str) -> Result<Option<Value>, BuildError> {
        let Some(model) = self.model(candidate) else {
            return Ok(None);
        };
        let end_user = self.attribute("end_user_id");
        if !truthy(end_user) {
            return Ok(None);
        }
        let tokens = limit_in(&self.metadata.end_user_model_tpm_limit, &model)?;
        let requests = limit_in(&self.metadata.end_user_model_rpm_limit, &model)?;
        if tokens.is_null() && requests.is_null() {
            return Ok(None);
        }
        Ok(Some(descriptor(
            "model_per_end_user",
            Value::String(format!("{}:{}", text(end_user), model.group)),
            requests,
            tokens,
            self.window_size,
        )))
    }

    fn customer_scopes(&self) -> Vec<Value> {
        self.metadata
            .customer_scopes
            .iter()
            .flatten()
            .filter(|scope| !scope.rpm_limit.is_null() || !scope.tpm_limit.is_null())
            .map(|scope| {
                descriptor(
                    &scope.key,
                    Value::String(scope.entity_id.clone()),
                    scope.rpm_limit.clone(),
                    scope.tpm_limit.clone(),
                    self.window_size,
                )
            })
            .collect()
    }

    fn customer_models(&self, candidate: &str) -> Result<Vec<Value>, BuildError> {
        let (Some(model), Some(scopes)) = (self.model(candidate), &self.metadata.customer_scopes)
        else {
            return Ok(Vec::new());
        };
        let covered = self.lookups.dynamic_covered.contains(&model.group);
        let mut found = Vec::new();
        for scope in scopes {
            let mut tokens = limit_in(&scope.model_tpm_limit, &model)?;
            let mut requests = limit_in(&scope.model_rpm_limit, &model)?;
            // A borrowing customer organization's per-model limits are shares the dynamic limiter enforces at
            // saturation (`borrowing_customer_org_model_limits`).
            if covered && scope.borrows {
                tokens = Value::Null;
                requests = Value::Null;
            }
            if tokens.is_null() && requests.is_null() {
                continue;
            }
            found.push(descriptor(
                &format!("model_per_{}", scope.key),
                Value::String(format!("{}:{}", scope.entity_id, model.group)),
                requests,
                tokens,
                self.window_size,
            ));
        }
        Ok(found)
    }

    fn unique_tags(&self) -> Vec<&Value> {
        let mut tags: Vec<&Value> = Vec::new();
        for tag in &self.request.tags {
            if !tags.contains(&tag) {
                tags.push(tag);
            }
        }
        tags
    }

    fn tags_per_key(&self) -> Vec<Value> {
        let api_key = self.attribute("api_key");
        let Value::Object(limits) = &self.metadata.key_tag_rpm_limit else {
            return Vec::new();
        };
        if !truthy(api_key) || limits.is_empty() {
            return Vec::new();
        }
        self.unique_tags()
            .into_iter()
            .filter_map(|tag| {
                let limit = limits.get(tag.as_str()?).filter(|limit| !limit.is_null())?;
                Some(descriptor(
                    "tag_per_key",
                    Value::String(format!("{}:{}", text(api_key), text(tag))),
                    limit.clone(),
                    Value::Null,
                    self.window_size,
                ))
            })
            .collect()
    }

    fn model_per_team(&self, candidate: &str) -> Result<Option<Value>, BuildError> {
        let Some(model) = self.model(candidate) else {
            return Ok(None);
        };
        let (borrows_rpm, borrows_tpm) = self.borrowing(&model)?;
        let requests = if borrows_rpm {
            Value::Null
        } else {
            self.inherited_team_rpm(&model)?
        };
        let tokens = if borrows_tpm {
            Value::Null
        } else {
            self.inherited_team_tpm(&model)?
        };
        if requests.is_null() && tokens.is_null() {
            return Ok(None);
        }
        Ok(Some(descriptor(
            "model_per_team",
            Value::String(format!(
                "{}:{}",
                text(self.attribute("team_id")),
                model.group
            )),
            requests,
            tokens,
            self.window_size,
        )))
    }

    fn team_ptu(&self) -> Option<(Value, Ceiling)> {
        self.lookups.rate_limited_model(self.requested_model)?;
        let team = self.attribute("team_id");
        if !truthy(team) {
            return None;
        }
        let ceiling = self.lookups.ptu_ceiling.clone()?;
        Some((
            descriptor(
                PTU_TEAM,
                Value::String(format!("{}:{}", text(team), ceiling.model_group)),
                Value::Null,
                json!(ceiling.tpm_limit),
                self.window_size,
            ),
            ceiling,
        ))
    }

    fn agent_policy(&self, agent_id: &Value) -> Option<&AgentPolicy> {
        let selected = if agent_id == self.attribute("agent_id") {
            &self.lookups.managed_agent_policy
        } else {
            &self.lookups.invoked_agent_policy
        };
        match selected {
            Some(policy) => Some(policy),
            None => self
                .lookups
                .registry_agents
                .iter()
                .find(|(id, _)| id == agent_id)
                .and_then(|(_, policy)| policy.as_ref()),
        }
    }

    fn agents(&self) -> Result<Vec<Value>, BuildError> {
        if let Some((exception, message)) = &self.request.resolved_agent_error {
            return Err(raised(exception, message.clone()));
        }
        let resolved = self.request.resolved_agent_id.clone();
        let invoked = self.attribute("invoked_agent_id").clone();
        hashed(&[&resolved, &invoked])?;
        let ids: Vec<Value> = if resolved == invoked {
            vec![resolved]
        } else {
            vec![resolved, invoked]
        };
        let mut descriptors = Vec::new();
        for agent_id in ids.iter().filter(|agent_id| !agent_id.is_null()) {
            descriptors.extend(self.agent_descriptors(agent_id)?);
        }
        Ok(descriptors)
    }

    fn agent_descriptors(&self, agent_id: &Value) -> Result<Vec<Value>, BuildError> {
        let Some(policy) = self.agent_policy(agent_id) else {
            return Ok(Vec::new());
        };
        let mut descriptors = Vec::new();
        if !policy.rpm_limit.is_null() || !policy.tpm_limit.is_null() {
            descriptors.push(descriptor(
                "agent",
                agent_id.clone(),
                policy.rpm_limit.clone(),
                policy.tpm_limit.clone(),
                self.window_size,
            ));
        }
        let session_limited =
            !policy.session_rpm_limit.is_null() || !policy.session_tpm_limit.is_null();
        if session_limited && let Some((exception, message)) = &self.request.session_error {
            return Err(raised(exception, message.clone()));
        }
        if session_limited && let Some(session) = &self.request.session_id {
            descriptors.push(descriptor(
                "agent_session",
                Value::String(format!("{}:{session}", text(agent_id))),
                policy.session_rpm_limit.clone(),
                policy.session_tpm_limit.clone(),
                self.window_size,
            ));
        }
        Ok(descriptors)
    }

    fn scoped_models(&self) -> Vec<RateLimitedModel> {
        if self.requested_model.is_none_or(str::is_empty) {
            return Vec::new();
        }
        self.lookups
            .candidates
            .iter()
            .filter_map(|candidate| self.model(candidate))
            .collect()
    }

    fn project_models(&self) -> Result<Vec<Value>, BuildError> {
        let project = self.attribute("project_id");
        coalesce(
            self.scoped_models()
                .iter()
                .map(|model| {
                    let tokens = limit_in(&self.metadata.project_model_tpm_limit, model)?;
                    let requests = limit_in(&self.metadata.project_model_rpm_limit, model)?;
                    Ok((!tokens.is_null() || !requests.is_null()).then(|| {
                        descriptor(
                            "model_per_project",
                            Value::String(format!("{}:{}", text(project), model.group)),
                            requests,
                            tokens,
                            self.window_size,
                        )
                    }))
                })
                .filter_map(Result::transpose)
                .collect::<Result<_, BuildError>>()?,
        )
    }

    fn project_io(&self) -> Result<Vec<Value>, BuildError> {
        let project = self.attribute("project_id");
        if project.is_null() {
            return Ok(Vec::new());
        }
        let mut found = Vec::new();
        for model in self.scoped_models() {
            let value = format!("{}:{}", text(project), model.group);
            let itpm = limit_in(&self.metadata.project_model_itpm_limit, &model)?;
            let otpm = limit_in(&self.metadata.project_model_otpm_limit, &model)?;
            for (key, limit) in [(PROJECT_ITPM, itpm), (PROJECT_OTPM, otpm)] {
                if !limit.is_null() {
                    found.push(descriptor(
                        key,
                        Value::String(value.clone()),
                        Value::Null,
                        limit,
                        self.window_size,
                    ));
                }
            }
        }
        coalesce(found)
    }

    fn organization_models(&self) -> Result<Vec<Value>, BuildError> {
        let organization = self.attribute("org_id");
        coalesce(
            self.scoped_models()
                .iter()
                .map(|model| {
                    let tokens = limit_in(&self.metadata.organization_model_tpm_limit, model)?;
                    let requests = limit_in(&self.metadata.organization_model_rpm_limit, model)?;
                    Ok((!tokens.is_null() || !requests.is_null()).then(|| {
                        descriptor(
                            "model_per_organization",
                            Value::String(format!("{}:{}", text(organization), model.group)),
                            requests,
                            tokens,
                            self.window_size,
                        )
                    }))
                })
                .filter_map(Result::transpose)
                .collect::<Result<_, BuildError>>()?,
        )
    }

    fn tags_from_database(&self) -> Vec<Value> {
        self.unique_tags()
            .into_iter()
            .filter_map(|tag| {
                let (_, requests, tokens) = self
                    .lookups
                    .tag_limits
                    .iter()
                    .find(|(limited, _, _)| limited == tag)?;
                Some(descriptor(
                    "tag",
                    tag.clone(),
                    requests.clone(),
                    tokens.clone(),
                    self.window_size,
                ))
            })
            .collect()
    }

    fn auth_facts(&self) -> Result<AuthFacts, BuildError> {
        let mut facts: AuthFacts = serde_json::from_value(Value::Object(self.auth.clone()))
            .map_err(|error| BuildError::Auth(error.to_string()))?;
        facts.rpm_limit_type = self.rpm_limit_type.map(str::to_owned);
        facts.tpm_limit_type = self.tpm_limit_type.map(str::to_owned);
        facts.model_has_failures = self.lookups.model_has_failures;
        facts.window_size = json!(self.window_size);
        Ok(facts)
    }

    pub fn build(&self) -> Result<Built, BuildError> {
        let auth = auth_descriptors(&self.auth_facts()?)?;
        let mut descriptors = auth.leading;
        descriptors.extend(self.customer_scopes());
        for candidate in &self.lookups.candidates {
            descriptors.extend(self.model_per_key(candidate)?);
            descriptors.extend(self.model_per_end_user(candidate)?);
            descriptors.extend(self.customer_models(candidate)?);
        }
        descriptors.extend(self.tags_per_key());
        for candidate in &self.lookups.candidates {
            descriptors.extend(self.model_per_team(candidate)?);
        }
        let ptu = self.team_ptu();
        let ptu_ceiling = ptu.as_ref().map(|(_, ceiling)| ceiling.clone());
        descriptors.extend(ptu.map(|(descriptor, _)| descriptor));
        descriptors.extend(self.agents()?);
        let mut descriptors = if self.lookups.candidates.len() == 1 {
            descriptors
        } else {
            coalesce(descriptors)?
        };
        descriptors.extend(self.project_models()?);
        descriptors.extend(self.project_io()?);
        descriptors.extend(auth.organization);
        descriptors.extend(self.organization_models()?);
        descriptors.extend(self.tags_from_database());
        Ok(Built {
            descriptors,
            ptu_ceiling,
        })
    }
}

/// The descriptor keys a router fallback target reserves: every limit scoped to one model.
pub const FALLBACK_TARGET_KEYS: [&str; 15] = [
    "model_per_key",
    "model_per_team",
    "model_per_end_user",
    "model_per_organization",
    "model_per_project",
    PROJECT_ITPM,
    PROJECT_OTPM,
    "model_per_customer_org",
    "model_per_customer_team",
    "model_per_customer_member",
    "model_per_app",
    "model_per_app_customer_org",
    "model_per_app_customer_member",
    "model_per_app_group",
    "model_per_app_group_customer_org",
];
const MODEL_SCOPED_PREFIX: &str = "model_per_";
const SKILL_HUB_PUBLISH: &str = "skill_hub_publish";
const SKILL_HUB_PUBLISH_RPM: i64 = 10;
const SKILL_HUB_PUBLISH_WINDOW_SECONDS: i64 = 60;

/// The MCP server a tool call reaches, with the key and team limits configured for it.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct McpScope {
    pub server_name: Option<String>,
    pub server_id: Value,
    pub server_rpm: Value,
    pub key_rpm_limit: Value,
    pub team_rpm_limit: Value,
    pub source_team_rpm_limits: Vec<(String, Value)>,
}

/// One scope a batch is charged per day instead of per minute.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct TpdLimit {
    pub key: String,
    pub value: Value,
    pub limit: Value,
}

/// Which descriptors an entry point outside the V2 pre-call checks.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Selection {
    /// Every limit the request's pre-call would check.
    #[default]
    Request,
    /// The per-model limits a router fallback target reserves.
    FallbackTarget,
    /// The token windows one WebSocket turn is checked against.
    SocketTurn {
        #[serde(default)]
        agent_windows: Vec<Value>,
        #[serde(default)]
        project_io: bool,
    },
    /// The project ITPM and OTPM quotas of the requested model.
    ProjectIo,
    /// The limits a proxy sub-call inside an admitted request is checked against.
    SubCall,
    /// The limits a batch submission is charged against.
    Batch {
        #[serde(default)]
        tpd_limits: Vec<TpdLimit>,
    },
    /// The key, team and server limits of one MCP server.
    Mcp(Box<McpScope>),
    /// One caller's Skills Hub publishing budget.
    SkillHubPublish { value: String },
}

fn member(limits: &Value, name: &str) -> Result<Value, BuildError> {
    match limits {
        Value::Null => Ok(Value::Null),
        Value::Object(limits) => Ok(limits.get(name).cloned().unwrap_or(Value::Null)),
        other => Err(raised(
            "AttributeError",
            format!("'{}' object has no attribute 'get'", type_name(other)),
        )),
    }
}

fn rate_limit_member(descriptor: &Value, name: &str) -> Value {
    descriptor["rate_limit"]
        .get(name)
        .cloned()
        .unwrap_or(Value::Null)
}

fn keyed(descriptor: &Value, keys: &[&str]) -> bool {
    descriptor["key"]
        .as_str()
        .is_some_and(|key| keys.contains(&key))
}

/// Each descriptor's token window alone, for the turns of a socket that took its slot and RPM at
/// admission. PTU budgets are charged by their own path, and so is project IO unless the turns
/// charge it themselves, so those are left out.
pub fn socket_turn_tpm_windows(
    built: Vec<Value>,
    agent_windows: &[Value],
    project_io: bool,
) -> Vec<Value> {
    let uncounted: &[&str] = if project_io {
        &[PTU_TEAM]
    } else {
        &[PROJECT_ITPM, PROJECT_OTPM, PTU_TEAM]
    };
    let built_scopes: Vec<(Value, Value)> = built.iter().map(identity).collect();
    built
        .iter()
        .chain(
            agent_windows
                .iter()
                .filter(|window| !built_scopes.contains(&identity(window))),
        )
        .filter(|descriptor| !keyed(descriptor, uncounted))
        .filter_map(|descriptor| {
            let tokens = rate_limit_member(descriptor, "tokens_per_unit");
            (!tokens.is_null()).then(|| {
                json!({
                    "key": descriptor["key"],
                    "value": descriptor["value"],
                    "rate_limit": {
                        "tokens_per_unit": tokens,
                        "window_size": rate_limit_member(descriptor, "window_size"),
                    },
                })
            })
        })
        .collect()
}

/// A sub-call runs inside its caller's admitted request, so it takes no parallel slot and only the
/// RPM of a limit on the model itself gates it.
pub fn sub_call_descriptor(descriptor: &Value) -> Value {
    if descriptor["rate_limit"].is_null() {
        return descriptor.clone();
    }
    let caller_wide = !descriptor["key"]
        .as_str()
        .is_some_and(|key| key.starts_with(MODEL_SCOPED_PREFIX));
    let requests = if caller_wide {
        Value::Null
    } else {
        rate_limit_member(descriptor, "requests_per_unit")
    };
    json!({
        "key": descriptor["key"],
        "value": descriptor["value"],
        "rate_limit": {
            "requests_per_unit": requests,
            "tokens_per_unit": rate_limit_member(descriptor, "tokens_per_unit"),
            "window_size": rate_limit_member(descriptor, "window_size"),
        },
    })
}

/// A scope with a daily token limit is charged against its own `_tpd` descriptor instead of its
/// per-minute one, because a batch's rows never share a minute with the submission.
pub fn batch_descriptors(descriptors: Vec<Value>, tpd_limits: &[TpdLimit]) -> Vec<Value> {
    let mut limited: Vec<&TpdLimit> = Vec::new();
    for tpd in tpd_limits
        .iter()
        .filter(|tpd| truthy(&tpd.value) && !tpd.limit.is_null())
    {
        match limited.iter().position(|seen| seen.key == tpd.key) {
            Some(index) => limited[index] = tpd,
            None => limited.push(tpd),
        }
    }
    if limited.is_empty() {
        return descriptors;
    }
    let keys: Vec<&str> = limited.iter().map(|&tpd| tpd.key.as_str()).collect();
    descriptors
        .into_iter()
        .filter(|descriptor| !keyed(descriptor, &keys))
        .chain(limited.into_iter().map(|tpd| {
            json!({
                "key": format!("{}{}", tpd.key, crate::batch::TPD_SUFFIX),
                "value": tpd.value,
                "rate_limit": {
                    "requests_per_unit": Value::Null,
                    "tokens_per_unit": tpd.limit,
                    "window_size": crate::batch::TPD_WINDOW_SECONDS,
                },
            })
        }))
        .collect()
}

pub fn skill_hub_publish_descriptor(value: &str) -> Value {
    json!({
        "key": SKILL_HUB_PUBLISH,
        "value": value,
        "rate_limit": {
            "requests_per_unit": SKILL_HUB_PUBLISH_RPM,
            "window_size": SKILL_HUB_PUBLISH_WINDOW_SECONDS,
        },
    })
}

impl Request<'_> {
    /// The auth, customer, per-model, tag, team, PTU and agent limits, without the project,
    /// organization and database tag limits `build` appends. The organization's own descriptors come
    /// back separately so a caller can place them.
    pub fn base(&self) -> Result<(Vec<Value>, Vec<Value>), BuildError> {
        let auth = auth_descriptors(&self.auth_facts()?)?;
        let mut descriptors = auth.leading;
        descriptors.extend(self.customer_scopes());
        for candidate in &self.lookups.candidates {
            descriptors.extend(self.model_per_key(candidate)?);
            descriptors.extend(self.model_per_end_user(candidate)?);
            descriptors.extend(self.customer_models(candidate)?);
        }
        descriptors.extend(self.tags_per_key());
        for candidate in &self.lookups.candidates {
            descriptors.extend(self.model_per_team(candidate)?);
        }
        descriptors.extend(self.team_ptu().map(|(descriptor, _)| descriptor));
        descriptors.extend(self.agents()?);
        let descriptors = if self.lookups.candidates.len() == 1 {
            descriptors
        } else {
            coalesce(descriptors)?
        };
        Ok((descriptors, auth.organization))
    }

    fn mcp(&self, scope: &McpScope) -> Result<Vec<Value>, BuildError> {
        let mut found = Vec::new();
        if let Some(server) = scope.server_name.as_deref().filter(|name| !name.is_empty()) {
            let api_key = self.attribute("api_key");
            if truthy(api_key) && truthy(&scope.key_rpm_limit) {
                let limit = member(&scope.key_rpm_limit, server)?;
                if !limit.is_null() {
                    found.push(descriptor(
                        "mcp_per_key",
                        Value::String(format!("{}:{server}", text(api_key))),
                        limit,
                        Value::Null,
                        self.window_size,
                    ));
                }
            }
            let team = self.attribute("team_id");
            let source = |team_id: &Value| {
                scope
                    .source_team_rpm_limits
                    .iter()
                    .find(|(source, _)| team_id.as_str() == Some(source.as_str()))
                    .map_or(Value::Null, |(_, limit)| limit.clone())
            };
            let mut team_ids: Vec<Value> = Vec::new();
            for team_id in truthy(team).then(|| team.clone()).into_iter().chain(
                scope
                    .source_team_rpm_limits
                    .iter()
                    .map(|(team_id, _)| Value::String(team_id.clone())),
            ) {
                if !team_ids.contains(&team_id) {
                    team_ids.push(team_id);
                }
            }
            for team_id in team_ids {
                let limit = if &team_id == team && truthy(&scope.team_rpm_limit) {
                    scope.team_rpm_limit.clone()
                } else {
                    source(&team_id)
                };
                if !truthy(&team_id) || !truthy(&limit) {
                    continue;
                }
                let server_limit = member(&limit, server)?;
                if server_limit.is_null() {
                    continue;
                }
                found.push(descriptor(
                    "mcp_per_team",
                    Value::String(format!("{}:{server}", text(&team_id))),
                    server_limit,
                    Value::Null,
                    self.window_size,
                ));
            }
        }
        if !scope.server_rpm.is_null() {
            found.push(descriptor(
                "mcp_server",
                scope.server_id.clone(),
                scope.server_rpm.clone(),
                Value::Null,
                self.window_size,
            ));
        }
        Ok(found)
    }

    /// The descriptors one entry point outside the V2 pre-call checks, built by the same rules.
    pub fn select(&self, selection: &Selection) -> Result<Vec<Value>, BuildError> {
        match selection {
            Selection::Request => Ok(self.build()?.descriptors),
            Selection::FallbackTarget => Ok(self
                .build()?
                .descriptors
                .into_iter()
                .filter(|descriptor| keyed(descriptor, &FALLBACK_TARGET_KEYS))
                .collect()),
            Selection::SocketTurn {
                agent_windows,
                project_io,
            } => Ok(socket_turn_tpm_windows(
                self.build()?.descriptors,
                agent_windows,
                *project_io,
            )),
            Selection::ProjectIo => self.project_io(),
            Selection::SubCall => {
                let (mut descriptors, organization) = self.base()?;
                descriptors.extend(self.project_models()?);
                descriptors.extend(organization);
                descriptors.extend(self.organization_models()?);
                Ok(descriptors.iter().map(sub_call_descriptor).collect())
            }
            Selection::Batch { tpd_limits } => Ok(batch_descriptors(self.base()?.0, tpd_limits)),
            Selection::Mcp(scope) => self.mcp(scope),
            Selection::SkillHubPublish { value } => Ok(vec![skill_hub_publish_descriptor(value)]),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::app("app")]
    #[case::organization("app_customer_org")]
    #[case::member("app_customer_member")]
    fn app_limits_emit_scope_and_model_descriptors(#[case] owner: &str) {
        let metadata = AuthMetadata {
            customer_scopes: Some(vec![CustomerScope {
                key: owner.into(),
                entity_id: "identity".into(),
                rpm_limit: json!(2),
                tpm_limit: json!(20),
                model_rpm_limit: json!({"model": 1}),
                model_tpm_limit: json!({"model": 10}),
                ..Default::default()
            }]),
            ..Default::default()
        };
        let lookups = lookups(&["model"]);
        let descriptors = built(json!({}), metadata, &lookups, "model");
        assert!(descriptors.contains(&descriptor(
            owner,
            json!("identity"),
            json!(2),
            json!(20),
            60
        )));
        assert!(descriptors.contains(&descriptor(
            &format!("model_per_{owner}"),
            json!("identity:model"),
            json!(1),
            json!(10),
            60,
        )));
    }

    fn lookups(candidates: &[&str]) -> Lookups {
        Lookups {
            candidates: candidates.iter().map(|name| (*name).to_owned()).collect(),
            ..Lookups::default()
        }
    }

    fn built(auth: Value, metadata: AuthMetadata, lookups: &Lookups, model: &str) -> Vec<Value> {
        let Value::Object(auth) = auth else {
            unreachable!("the auth facts are an object")
        };
        Request {
            auth: &auth,
            rpm_limit_type: None,
            tpm_limit_type: None,
            metadata: &metadata,
            lookups,
            request: &RequestFacts::default(),
            requested_model: Some(model),
            window_size: 60,
        }
        .build()
        .unwrap()
        .descriptors
    }

    #[rstest]
    fn a_key_model_limit_falls_back_to_the_team_map_then_the_deployment_default() {
        let mut found = lookups(&["gpt-4o"]);
        found.deployment_limits.insert(
            "gpt-4o".to_owned(),
            DeploymentLimits {
                rpm: Some(9),
                tpm: None,
            },
        );
        let descriptors = built(
            json!({"api_key": "k", "team_id": "t"}),
            AuthMetadata {
                team_model_tpm_limit: json!({"gpt-4o": "20"}),
                ..AuthMetadata::default()
            },
            &found,
            "gpt-4o",
        );
        assert_eq!(
            descriptors,
            vec![
                descriptor(
                    "model_per_key",
                    json!("k:gpt-4o"),
                    json!(9),
                    json!("20"),
                    60
                ),
                descriptor(
                    "model_per_team",
                    json!("t:gpt-4o"),
                    json!(null),
                    json!("20"),
                    60
                ),
            ]
        );
    }

    #[rstest]
    fn a_key_owning_the_model_limit_drops_the_inherited_team_limit() {
        let descriptors = built(
            json!({"api_key": "k", "team_id": "t"}),
            AuthMetadata {
                key_model_tpm_limit: json!({"gpt-4o": 5}),
                team_model_tpm_limit: json!({"gpt-4o": 20}),
                ..AuthMetadata::default()
            },
            &lookups(&["gpt-4o"]),
            "gpt-4o",
        );
        assert_eq!(
            descriptors,
            vec![descriptor(
                "model_per_key",
                json!("k:gpt-4o"),
                json!(null),
                json!(5),
                60
            )]
        );
    }

    #[rstest]
    fn latest_alias_candidates_coalesce_to_the_smallest_limit() {
        let descriptors = built(
            json!({"api_key": "k"}),
            AuthMetadata {
                key_model_rpm_limit: json!({"m": 7, "m-2": 3}),
                ..AuthMetadata::default()
            },
            &Lookups {
                groups: BTreeMap::from([
                    ("m".to_owned(), Some("g".to_owned())),
                    ("m-2".to_owned(), Some("g".to_owned())),
                ]),
                ..lookups(&["m", "m-2"])
            },
            "m",
        );
        assert_eq!(
            descriptors,
            vec![
                json!({"key": "model_per_key", "value": "k:g", "rate_limit": {
                "requests_per_unit": 3, "tokens_per_unit": null,
                "max_parallel_requests": null, "window_size": 60}})
            ]
        );
    }

    #[rstest]
    #[case::numbers(json!(10), json!(9), Ok(json!(9)))]
    #[case::texts_compare_like_python(json!("10"), json!("9"), Ok(json!("10")))]
    #[case::mixed_raise_like_python(json!("10"), json!(9), Err(BuildError::Incomparable("team".to_owned())))]
    fn the_merged_limit_is_python_min(
        #[case] first: Value,
        #[case] second: Value,
        #[case] expected: Result<Value, BuildError>,
    ) {
        let merged = coalesce(vec![
            descriptor("team", json!("t"), first, json!(null), 60),
            descriptor("team", json!("t"), second, json!(null), 60),
        ]);
        assert_eq!(
            merged.map(|merged| merged[0]["rate_limit"]["requests_per_unit"].clone()),
            expected
        );
    }

    #[rstest]
    fn agent_limits_use_the_keys_managed_policy_and_the_request_session() {
        let Value::Object(auth) = json!({"api_key": "k", "agent_id": "a1"}) else {
            unreachable!("the auth facts are an object")
        };
        let found = Lookups {
            managed_agent_policy: Some(AgentPolicy {
                rpm_limit: json!(4),
                session_tpm_limit: json!(40),
                ..AgentPolicy::default()
            }),
            ..lookups(&[""])
        };
        let request = RequestFacts {
            resolved_agent_id: json!("a1"),
            session_id: Some("s".to_owned()),
            ..RequestFacts::default()
        };
        let descriptors = Request {
            auth: &auth,
            rpm_limit_type: None,
            tpm_limit_type: None,
            metadata: &AuthMetadata::default(),
            lookups: &found,
            request: &request,
            requested_model: None,
            window_size: 60,
        }
        .build()
        .unwrap()
        .descriptors;
        assert_eq!(
            descriptors,
            vec![
                descriptor("agent", json!("a1"), json!(4), json!(null), 60),
                descriptor("agent_session", json!("a1:s"), json!(null), json!(40), 60),
            ]
        );
    }
}
