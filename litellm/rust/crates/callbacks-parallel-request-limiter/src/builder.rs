use std::{cmp::Ordering, collections::BTreeMap};

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
    pub customer_scopes: Option<Vec<CustomerScope>>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct CustomerScope {
    pub key: String,
    pub entity_id: String,
    pub rpm_limit: Value,
    pub tpm_limit: Value,
    pub model_rpm_limit: Value,
    pub model_tpm_limit: Value,
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
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|float| float != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(fields) => !fields.is_empty(),
    }
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

    fn model_per_key(&self, candidate: &str) -> Result<Option<Value>, BuildError> {
        let Some(model) = self.model(candidate) else {
            return Ok(None);
        };
        let (borrows_rpm, borrows_tpm) = self.borrowing(&model)?;
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
        let mut found = Vec::new();
        for scope in scopes {
            let tokens = limit_in(&scope.model_tpm_limit, &model)?;
            let requests = limit_in(&scope.model_rpm_limit, &model)?;
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

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

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
