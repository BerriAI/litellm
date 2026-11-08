use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct BoundIdentity {
    pub active: bool,
    pub revision: String,
}

#[derive(Debug, Deserialize)]
pub struct Agent {
    pub agent_id: String,
    pub enabled: bool,
    pub identity_managed: bool,
    pub identity: Option<BoundIdentity>,
    pub execution_mode: String,
}

#[derive(Debug, Deserialize)]
pub struct Context {
    pub agent_id: String,
    pub binding_revision: Option<String>,
    pub mode: String,
    pub user_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind")]
pub enum ManagedFacts {
    Actor {
        agent: Agent,
        context: Option<Context>,
    },
    ActorLookup {
        agent_id: Option<String>,
        has_database: bool,
        registry_loaded: bool,
        registry_managed: bool,
        agent_loaded: bool,
        agent: Option<Agent>,
        retired: Option<bool>,
        context: Option<Context>,
        jwt_claims: bool,
        delegation_verified: bool,
        human_grants: Option<Vec<String>>,
        policy_assigned: bool,
    },
    UnionGrants {
        grants: Vec<Option<Vec<String>>>,
    },
    IntersectGrants {
        left: Option<Vec<String>>,
        right: Option<Vec<String>>,
    },
    DeclaredGrants {
        direct: Vec<String>,
        declared_groups: Vec<String>,
        unified_groups: Vec<String>,
        group_agents: Vec<String>,
        unified_agents: Vec<String>,
    },
}

#[derive(Debug, PartialEq, Serialize, strum::VariantArray)]
pub enum ManagedDenial {
    Disabled,
    MissingContext,
    ChangedIdentity,
    ExecutionMode,
    MissingHuman,
    DatabaseRequired,
    Removed,
    VerifiedIdentityRequired,
    HumanNotPermitted,
}

#[derive(Debug, PartialEq, Serialize, strum::VariantArray)]
pub enum ManagedAction {
    Allow,
    Deny,
    ReadRegistry,
    ReadAgent,
    ReadRetired,
    AssignPolicy,
    ReadHumanGrants,
    Grants,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct ManagedDecision {
    pub action: ManagedAction,
    pub denial: Option<ManagedDenial>,
    pub grants: Option<Vec<String>>,
}

fn action(action: ManagedAction) -> ManagedDecision {
    ManagedDecision {
        action,
        denial: None,
        grants: None,
    }
}

fn denied(denial: ManagedDenial) -> ManagedDecision {
    ManagedDecision {
        action: ManagedAction::Deny,
        denial: Some(denial),
        grants: None,
    }
}

fn actor(agent: &Agent, context: Option<&Context>) -> ManagedDecision {
    if !agent.enabled
        || !agent
            .identity
            .as_ref()
            .is_some_and(|identity| identity.active)
    {
        return denied(ManagedDenial::Disabled);
    }
    let Some(context) = context else {
        return denied(ManagedDenial::MissingContext);
    };
    if context.agent_id != agent.agent_id
        || agent
            .identity
            .as_ref()
            .is_some_and(|identity| context.binding_revision.as_ref() != Some(&identity.revision))
    {
        return denied(ManagedDenial::ChangedIdentity);
    }
    if agent.execution_mode != context.mode && agent.execution_mode != "both" {
        return denied(ManagedDenial::ExecutionMode);
    }
    if context.mode == "delegated"
        && !context
            .user_id
            .as_ref()
            .is_some_and(|user| !user.is_empty())
    {
        return denied(ManagedDenial::MissingHuman);
    }
    action(ManagedAction::Allow)
}

pub fn decide(facts: ManagedFacts) -> ManagedDecision {
    match facts {
        ManagedFacts::Actor { agent, context } => actor(&agent, context.as_ref()),
        ManagedFacts::ActorLookup {
            agent_id,
            has_database,
            registry_loaded,
            registry_managed,
            agent_loaded,
            agent,
            retired,
            context,
            jwt_claims,
            delegation_verified,
            human_grants,
            policy_assigned,
        } => {
            if agent_id.is_none() {
                return action(ManagedAction::Allow);
            }
            if !has_database {
                if !registry_loaded {
                    return action(ManagedAction::ReadRegistry);
                }
                return if context.is_some() || registry_managed {
                    denied(ManagedDenial::DatabaseRequired)
                } else {
                    action(ManagedAction::Allow)
                };
            }
            if !agent_loaded {
                return action(ManagedAction::ReadAgent);
            }
            let Some(agent) = agent else {
                return match retired {
                    None => action(ManagedAction::ReadRetired),
                    Some(retired) if retired || context.is_some() => denied(ManagedDenial::Removed),
                    Some(_) => action(ManagedAction::Allow),
                };
            };
            if !agent.identity_managed {
                return action(ManagedAction::Allow);
            }
            if jwt_claims && context.is_none() {
                return denied(ManagedDenial::VerifiedIdentityRequired);
            }
            let checked = actor(&agent, context.as_ref());
            if checked.denial.is_some() {
                return checked;
            }
            if !policy_assigned {
                return action(ManagedAction::AssignPolicy);
            }
            if context
                .as_ref()
                .is_some_and(|context| context.mode == "delegated")
                && !delegation_verified
            {
                return match human_grants {
                    None => action(ManagedAction::ReadHumanGrants),
                    Some(grants) if !grants.contains(&agent.agent_id) => {
                        denied(ManagedDenial::HumanNotPermitted)
                    }
                    Some(_) => action(ManagedAction::Allow),
                };
            }
            action(ManagedAction::Allow)
        }
        ManagedFacts::UnionGrants { grants } => {
            let grants = if !grants.is_empty() && grants.iter().all(Option::is_none) {
                None
            } else {
                Some(grants.into_iter().flatten().flatten().collect())
            };
            ManagedDecision {
                action: ManagedAction::Grants,
                denial: None,
                grants,
            }
        }
        ManagedFacts::IntersectGrants { left, right } => {
            let grants = match (left, right) {
                (None, right) => right,
                (left, None) => left,
                (Some(left), Some(right)) => {
                    Some(left.into_iter().filter(|id| right.contains(id)).collect())
                }
            };
            ManagedDecision {
                action: ManagedAction::Grants,
                denial: None,
                grants,
            }
        }
        ManagedFacts::DeclaredGrants {
            direct,
            declared_groups,
            unified_groups,
            group_agents,
            unified_agents,
        } => {
            let grants =
                if direct.is_empty() && declared_groups.is_empty() && unified_groups.is_empty() {
                    None
                } else {
                    Some(
                        direct
                            .into_iter()
                            .chain(group_agents)
                            .chain(unified_agents)
                            .collect(),
                    )
                };
            ManagedDecision {
                action: ManagedAction::Grants,
                denial: None,
                grants,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::empty(vec![], Some(vec![]))]
    #[case::unrestricted(vec![None], None)]
    #[case::mixed(vec![None, Some(vec!["a".into()])], Some(vec!["a".into()]))]
    fn union(#[case] grants: Vec<Option<Vec<String>>>, #[case] expected: Option<Vec<String>>) {
        assert_eq!(
            decide(ManagedFacts::UnionGrants { grants }).grants,
            expected
        );
    }
}
