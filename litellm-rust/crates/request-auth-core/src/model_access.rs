use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{Error, patterns};

#[derive(Clone, Default, Deserialize)]
pub struct Provider {
    pub stripped_model: String,
    pub provider: String,
}

#[derive(Clone, Default, Deserialize)]
pub struct ModelRecord {
    pub access_groups: Vec<String>,
    pub provider: Option<Provider>,
    pub deployment_team: Option<String>,
    pub deployment_public_name: Option<String>,
}

#[derive(Clone, Default, Deserialize)]
pub struct Registry {
    pub present: bool,
    pub names: Vec<String>,
    pub aliases: BTreeMap<String, String>,
    pub records: BTreeMap<String, ModelRecord>,
    pub compaction_parents: BTreeMap<String, String>,
}

#[derive(Clone, Default, Deserialize)]
pub struct Aliases {
    pub team: BTreeMap<String, String>,
    pub key: BTreeMap<String, String>,
    #[serde(rename = "global_")]
    pub global: BTreeMap<String, String>,
}

#[derive(Clone, Deserialize)]
pub struct Facts {
    pub requested: Vec<String>,
    pub grants: Vec<String>,
    pub team_id: Option<String>,
    pub aliases: Aliases,
    pub registry: Registry,
    pub object_type: ObjectType,
    pub fallback_depth: usize,
    pub max_depth: usize,
    pub is_list: bool,
}

#[derive(Clone, Copy, Deserialize, Serialize, Debug, PartialEq, Eq, strum::VariantArray)]
#[serde(rename_all = "snake_case")]
pub enum ObjectType {
    User,
    Customer,
    Team,
    Key,
    Org,
    Project,
    Agent,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Allow,
    Deny {
        model: String,
        grants: Vec<String>,
        object_type: ObjectType,
    },
    Depth {
        model: Option<String>,
    },
    NoDefaultModels,
    NoAgentModels,
    EmptyAgentCeiling,
    PersonalTeam {
        model: String,
    },
    ReadGroups,
}

impl Decision {
    pub fn allowed(&self) -> bool {
        matches!(self, Self::Allow)
    }
}

fn unique(values: impl Iterator<Item = String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    values.filter(|value| seen.insert(value.clone())).collect()
}

pub fn resolve_key_grants(
    grants: &[String],
    team_id: Option<&str>,
    team_grants: &[String],
) -> Vec<String> {
    if grants.iter().any(|grant| grant == "all-team-models") {
        return if team_id.is_none() {
            Vec::new()
        } else {
            team_grants.to_vec()
        };
    }
    grants.to_vec()
}

fn expanded_grants(facts: &Facts) -> Vec<String> {
    if !facts.grants.iter().any(|grant| grant == "all-team-models")
        || facts.team_id.is_none()
        || !facts.registry.present
    {
        return facts.grants.clone();
    }
    let others = facts
        .grants
        .iter()
        .filter(|grant| grant.as_str() != "all-team-models")
        .cloned();
    if facts.registry.names.is_empty() {
        let values: Vec<_> = others.collect();
        return if values.is_empty() {
            facts.grants.clone()
        } else {
            values
        };
    }
    unique(others.chain(facts.registry.names.iter().cloned()))
}

fn helper(model: &str, facts: &Facts, aliases: bool) -> Result<bool, Error> {
    let grants = expanded_grants(facts);
    let record = facts.registry.records.get(model);
    let groups = record.map_or(&[][..], |record| record.access_groups.as_slice());
    if grants.iter().any(|grant| groups.contains(grant)) {
        return Ok(true);
    }
    let filtered: Vec<_> = grants
        .iter()
        .filter(|grant| !groups.contains(grant))
        .collect();
    if aliases && facts.aliases.team.contains_key(model) {
        return Ok(true);
    }
    for grant in &filtered {
        if patterns::model_pattern(model, grant)? {
            return Ok(true);
        }
    }
    if let Some(provider) = record.and_then(|record| record.provider.as_ref())
        && (provider.stripped_model != model || !model.contains('/'))
    {
        for grant in &filtered {
            if patterns::model_pattern(
                &format!("{}/{}", provider.provider, provider.stripped_model),
                grant,
            )? {
                return Ok(true);
            }
        }
    }
    Ok((filtered.is_empty() && grants.is_empty())
        || filtered.iter().any(|grant| {
            matches!(grant.as_str(), "*" | "all-proxy-models") || grant.as_str() == model
        }))
}

pub fn candidate_names(
    model: &str,
    aliases: &Aliases,
    registry: &Registry,
    team_id: Option<&str>,
) -> (Vec<String>, Vec<String>) {
    let after_team = aliases.team.get(model).map_or(model, String::as_str);
    let after_key = aliases
        .key
        .get(after_team)
        .map_or(after_team, String::as_str);
    let after_global = aliases
        .global
        .get(after_key)
        .map_or(after_key, String::as_str);
    let dispatched = aliases
        .key
        .get(after_global)
        .map_or(after_global, String::as_str);
    if after_key != after_team || dispatched != after_global {
        return (vec![dispatched.to_owned()], vec![]);
    }
    let candidates = std::iter::once(model.to_owned())
        .chain(registry.compaction_parents.get(model).cloned())
        .chain(
            aliases
                .global
                .get(model)
                .or_else(|| registry.aliases.get(model))
                .filter(|target| !target.is_empty())
                .cloned(),
        )
        .collect();
    let Some(team) = team_id else {
        return (candidates, vec![]);
    };
    if !registry.present || !model.starts_with(&format!("model_name_{team}_")) {
        return (candidates, vec![]);
    }
    let record = registry.records.get(model);
    let public = record
        .filter(|record| record.deployment_team.as_deref() == Some(team))
        .and_then(|record| record.deployment_public_name.clone())
        .filter(|name| !name.is_empty());
    let legacy = aliases
        .team
        .iter()
        .filter(|(_, target)| target.as_str() == model)
        .map(|(alias, _)| alias.clone());
    (candidates, public.into_iter().chain(legacy).collect())
}

pub fn live_team_target(model: &str, facts: &Facts) -> String {
    let Some(target) = facts.aliases.team.get(model) else {
        return model.to_owned();
    };
    let team = facts.team_id.as_deref().unwrap_or("None");
    if facts.registry.present
        && target.starts_with(&format!("model_name_{team}_"))
        && !facts.registry.names.contains(target)
    {
        return model.to_owned();
    }
    target.clone()
}

pub fn check(facts: &Facts) -> Result<Decision, Error> {
    if facts.fallback_depth >= facts.max_depth {
        return Ok(Decision::Depth { model: None });
    }
    for model in &facts.requested {
        if facts.is_list && facts.fallback_depth + 1 >= facts.max_depth {
            return Ok(Decision::Depth {
                model: Some(model.clone()),
            });
        }
        let (candidates, owning_names) = candidate_names(
            model,
            &facts.aliases,
            &facts.registry,
            facts.team_id.as_deref(),
        );
        let mut allowed = false;
        for candidate in candidates {
            if helper(&candidate, facts, true)? {
                allowed = true;
                break;
            }
        }
        if !allowed {
            for candidate in owning_names {
                if helper(&candidate, facts, false)? {
                    allowed = true;
                    break;
                }
            }
        }
        if !allowed {
            return Ok(Decision::Deny {
                model: model.clone(),
                grants: facts.grants.clone(),
                object_type: facts.object_type,
            });
        }
    }
    Ok(Decision::Allow)
}

#[derive(Deserialize)]
pub enum Request {
    Object(Facts),
    Grouped {
        facts: Facts,
        group_ids: Vec<String>,
        group_models: Option<Vec<String>>,
        merge: bool,
    },
    Helper {
        model: String,
        facts: Facts,
    },
    User {
        facts: Facts,
        present: bool,
    },
    Customer(Facts),
    Agent {
        facts: Facts,
        managed_grants: Option<Vec<String>>,
        ceilings: Vec<Vec<String>>,
    },
    Personal {
        requested: Vec<String>,
        teams: Vec<PersonalTeam>,
    },
    KeyGrants {
        grants: Vec<String>,
        team_id: Option<String>,
        team_grants: Vec<String>,
    },
    Candidates(Facts),
    PersonalApplies {
        enabled: bool,
        token_present: bool,
        team_id: Option<String>,
        user_id: Option<String>,
        master: bool,
        role: Option<String>,
        api_key_present: bool,
        inheritance: bool,
    },
    AssignedGroups {
        groups: Vec<AssignedGroup>,
        team_id: Option<String>,
        key_id: Option<String>,
    },
}

#[derive(Deserialize)]
pub struct AssignedGroup {
    pub assigned_team_ids: Vec<String>,
    pub assigned_key_ids: Vec<String>,
    pub models: Vec<String>,
}

#[derive(Deserialize)]
pub struct PersonalTeam {
    pub blocked: bool,
    pub available: bool,
    pub team: Facts,
    pub member: Option<Facts>,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum Response {
    Decision(Decision),
    Models(Vec<String>),
    Predicate(bool),
}

pub fn decide(request: Request) -> Result<Response, Error> {
    let decision = match request {
        Request::Object(facts) => check(&facts)?,
        Request::Grouped {
            facts,
            group_ids,
            group_models,
            merge,
        } => {
            let primary = check(&facts)?;
            if primary.allowed()
                || !matches!(primary, Decision::Deny { .. })
                || group_ids.is_empty()
            {
                primary
            } else if let Some(models) = group_models {
                if models.is_empty() {
                    primary
                } else {
                    check(&Facts {
                        grants: if merge {
                            unique(facts.grants.iter().cloned().chain(models))
                        } else {
                            models
                        },
                        ..facts
                    })?
                }
            } else {
                Decision::ReadGroups
            }
        }
        Request::Helper { model, facts } => {
            return Ok(Response::Predicate(helper(&model, &facts, true)?));
        }
        Request::User { facts, present } => {
            if !present {
                Decision::Allow
            } else if facts
                .grants
                .iter()
                .any(|grant| grant == "no-default-models")
            {
                Decision::NoDefaultModels
            } else {
                check(&facts)?
            }
        }
        Request::Customer(facts) => {
            for name in &facts.requested {
                let target = live_team_target(name, &facts);
                if target != *name && facts.grants.contains(name) {
                    continue;
                }
                let customer = Facts {
                    requested: vec![target],
                    is_list: false,
                    team_id: None,
                    aliases: Aliases {
                        key: facts.aliases.key.clone(),
                        global: facts.aliases.global.clone(),
                        ..Default::default()
                    },
                    ..facts.clone()
                };
                let result = check(&customer)?;
                if !result.allowed() {
                    return Ok(Response::Decision(result));
                }
            }
            Decision::Allow
        }
        Request::Agent {
            facts,
            managed_grants,
            ceilings,
        } => {
            if facts.requested.is_empty()
                || (facts.requested.len() == 1 && !facts.is_list && facts.requested[0].is_empty())
            {
                return Ok(Response::Decision(Decision::Allow));
            }
            let dispatched = facts
                .requested
                .iter()
                .map(|name| live_team_target(name, &facts))
                .collect();
            let scope = Facts {
                requested: dispatched,
                aliases: Aliases {
                    key: facts.aliases.key.clone(),
                    global: facts.aliases.global.clone(),
                    ..Default::default()
                },
                object_type: ObjectType::Agent,
                ..facts
            };
            if let Some(grants) = managed_grants {
                if grants.is_empty() {
                    return Ok(Response::Decision(Decision::NoAgentModels));
                }
                let result = check(&Facts {
                    grants,
                    ..scope.clone()
                })?;
                if !result.allowed() {
                    return Ok(Response::Decision(result));
                }
            }
            for grants in ceilings {
                if grants.is_empty() {
                    return Ok(Response::Decision(Decision::EmptyAgentCeiling));
                }
                let result = check(&Facts {
                    grants,
                    ..scope.clone()
                })?;
                if !result.allowed() {
                    return Ok(Response::Decision(result));
                }
            }
            Decision::Allow
        }
        Request::Personal { requested, teams } => {
            for name in requested {
                let mut allowed = false;
                for team in &teams {
                    if team.blocked || !team.available {
                        continue;
                    }
                    if !check(&Facts {
                        requested: vec![name.clone()],
                        is_list: false,
                        ..team.team.clone()
                    })?
                    .allowed()
                    {
                        continue;
                    }
                    if let Some(member) = &team.member
                        && !check(&Facts {
                            requested: vec![name.clone()],
                            is_list: false,
                            ..member.clone()
                        })?
                        .allowed()
                    {
                        continue;
                    }
                    allowed = true;
                    break;
                }
                if !allowed {
                    return Ok(Response::Decision(Decision::PersonalTeam { model: name }));
                }
            }
            Decision::Allow
        }
        Request::KeyGrants {
            grants,
            team_id,
            team_grants,
        } => {
            return Ok(Response::Models(resolve_key_grants(
                &grants,
                team_id.as_deref(),
                &team_grants,
            )));
        }
        Request::Candidates(facts) => {
            return Ok(Response::Models(unique(facts.requested.iter().flat_map(
                |model| {
                    let (candidates, owned) = candidate_names(
                        model,
                        &facts.aliases,
                        &facts.registry,
                        facts.team_id.as_deref(),
                    );
                    candidates.into_iter().chain(owned)
                },
            ))));
        }
        Request::PersonalApplies {
            enabled,
            token_present,
            team_id,
            user_id,
            master,
            role,
            api_key_present,
            inheritance,
        } => {
            return Ok(Response::Predicate(
                enabled
                    && token_present
                    && team_id.is_none()
                    && user_id.is_some()
                    && !master
                    && role.as_deref() != Some("proxy_admin")
                    && (!inheritance || api_key_present),
            ));
        }
        Request::AssignedGroups {
            groups,
            team_id,
            key_id,
        } => {
            return Ok(Response::Models(unique(
                groups
                    .into_iter()
                    .filter(|group| {
                        team_id.as_ref().is_some_and(|id| {
                            !id.is_empty() && group.assigned_team_ids.contains(id)
                        }) || key_id
                            .as_ref()
                            .is_some_and(|id| !id.is_empty() && group.assigned_key_ids.contains(id))
                    })
                    .flat_map(|group| group.models),
            )));
        }
    };
    Ok(Response::Decision(decision))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::empty(vec![], true)]
    #[case::wildcard(vec!["*"], true)]
    #[case::proxy(vec!["all-proxy-models"], true)]
    #[case::other(vec!["other"], false)]
    #[case::exact(vec!["model"], true)]
    fn model_grants(#[case] grants: Vec<&str>, #[case] expected: bool) {
        let facts = Facts {
            requested: vec!["model".into()],
            grants: grants.into_iter().map(str::to_owned).collect(),
            team_id: None,
            aliases: Aliases::default(),
            registry: Registry::default(),
            object_type: ObjectType::Key,
            fallback_depth: 0,
            max_depth: 10,
            is_list: false,
        };
        assert_eq!(check(&facts).unwrap().allowed(), expected);
    }
}
