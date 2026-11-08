use serde::{Deserialize, Serialize};

use crate::managed::Agent;

#[derive(Clone, Debug, Deserialize)]
pub struct Facts {
    pub registry_managed: bool,
    pub database_present: bool,
    pub target_loaded: bool,
    pub target: Option<Agent>,
    pub subject_present: bool,
    pub key_hash_present: bool,
    pub subject_managed: bool,
    pub session: bool,
    pub custom: bool,
    pub authority_loaded: bool,
    pub explicit_grants: Option<Vec<String>>,
    pub legacy_loaded: bool,
    pub legacy_grants: Option<Vec<String>>,
    pub target_ids: Option<Vec<String>>,
}

#[derive(Debug, PartialEq, Eq, Serialize, strum::VariantArray)]
pub enum Decision {
    ReadTarget,
    ReadAuthority,
    ReadExplicitGrants,
    ReadLegacyGrants,
    ReadTargetIds,
    Allow,
    Deny,
}

pub fn decide(f: &Facts) -> Decision {
    if f.registry_managed || f.database_present {
        if !f.target_loaded {
            return Decision::ReadTarget;
        }
        if f.target.is_none() && f.registry_managed {
            return Decision::Deny;
        }
        if let Some(target) = f.target.as_ref().filter(|target| target.identity_managed) {
            if !target.enabled
                || !target
                    .identity
                    .as_ref()
                    .is_some_and(|identity| identity.active)
                || !f.subject_present
            {
                return Decision::Deny;
            }
            if f.key_hash_present
                && !f.subject_managed
                && !f.session
                && !f.custom
                && !f.authority_loaded
            {
                return Decision::ReadAuthority;
            }
            return match f.explicit_grants.as_ref() {
                None => Decision::ReadExplicitGrants,
                Some(grants) if grants.contains(&target.agent_id) => Decision::Allow,
                Some(_) => Decision::Deny,
            };
        }
    }
    if !f.legacy_loaded {
        return Decision::ReadLegacyGrants;
    }
    let Some(grants) = f.legacy_grants.as_ref() else {
        return Decision::Allow;
    };
    match f.target_ids.as_ref() {
        None => Decision::ReadTargetIds,
        Some(ids) if ids.iter().any(|id| grants.contains(id)) => Decision::Allow,
        Some(_) => Decision::Deny,
    }
}

#[derive(Deserialize)]
pub struct Invocation {
    pub registered: Option<Agent>,
    pub has_store: bool,
    pub target_loaded: bool,
    pub target: Option<Agent>,
    pub actor_managed: bool,
    pub actor_id_present: bool,
    pub permissions: Option<Facts>,
    pub assigned: bool,
    pub fee_loaded: bool,
    pub fee: Option<f64>,
}

#[derive(Debug, PartialEq, Eq, Serialize, strum::VariantArray)]
pub enum InvocationAction {
    Complete,
    ReadTarget,
    ReadPermissions,
    Assign,
    ReadFee,
    Deny,
}

#[derive(Debug, PartialEq, Eq, Serialize, strum::VariantArray)]
pub enum InvocationDenial {
    DatabaseRequired,
    Removed,
    CallerNotPermitted,
    InvalidFee,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct InvocationPlan {
    pub action: InvocationAction,
    pub denial: Option<InvocationDenial>,
    pub use_target: bool,
    pub assign_billing: bool,
}

pub fn invocation(f: &Invocation) -> InvocationPlan {
    let action = |action, denial| InvocationPlan {
        action,
        denial,
        use_target: f.target.is_some(),
        assign_billing: !f.actor_id_present
            && f.target
                .as_ref()
                .or(f.registered.as_ref())
                .is_some_and(|agent| agent.identity_managed),
    };
    let Some(registered) = f.registered.as_ref() else {
        return action(InvocationAction::Complete, None);
    };
    let registered_managed = registered.identity_managed || registered.identity.is_some();
    if !f.has_store && registered_managed {
        return action(
            InvocationAction::Deny,
            Some(InvocationDenial::DatabaseRequired),
        );
    }
    if !f.target_loaded {
        return action(InvocationAction::ReadTarget, None);
    }
    if f.target.is_none() && registered_managed {
        return action(InvocationAction::Deny, Some(InvocationDenial::Removed));
    }
    let effective = f.target.as_ref().unwrap_or(registered);
    if !effective.identity_managed && !f.actor_managed {
        return action(InvocationAction::Complete, None);
    }
    match f.permissions.as_ref().map(decide) {
        None => return action(InvocationAction::ReadPermissions, None),
        Some(Decision::Allow) => (),
        Some(_) => {
            return action(
                InvocationAction::Deny,
                Some(InvocationDenial::CallerNotPermitted),
            );
        }
    }
    if !f.assigned {
        return action(InvocationAction::Assign, None);
    }
    if !f.fee_loaded {
        return action(InvocationAction::ReadFee, None);
    }
    if !f.fee.is_some_and(|fee| fee.is_finite() && fee >= 0.0) {
        return action(InvocationAction::Deny, Some(InvocationDenial::InvalidFee));
    }
    action(InvocationAction::Complete, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::free(Some(0.0), None)]
    #[case::priced(Some(1.0), None)]
    #[case::missing(None, Some(InvocationDenial::InvalidFee))]
    #[case::negative(Some(-1.0), Some(InvocationDenial::InvalidFee))]
    #[case::nan(Some(f64::NAN), Some(InvocationDenial::InvalidFee))]
    #[case::infinity(Some(f64::INFINITY), Some(InvocationDenial::InvalidFee))]
    fn invocation_fee_is_checked_after_assignment(
        #[case] fee: Option<f64>,
        #[case] denial: Option<InvocationDenial>,
    ) {
        let facts = Invocation {
            registered: Some(Agent {
                agent_id: "agent".into(),
                enabled: true,
                identity_managed: true,
                identity: None,
                execution_mode: "both".into(),
            }),
            has_store: true,
            target_loaded: true,
            target: Some(Agent {
                agent_id: "agent".into(),
                enabled: true,
                identity_managed: true,
                identity: None,
                execution_mode: "both".into(),
            }),
            actor_managed: false,
            actor_id_present: false,
            permissions: Some(Facts {
                registry_managed: false,
                database_present: false,
                target_loaded: false,
                target: None,
                subject_present: true,
                key_hash_present: false,
                subject_managed: false,
                session: false,
                custom: false,
                authority_loaded: false,
                explicit_grants: None,
                legacy_loaded: true,
                legacy_grants: None,
                target_ids: None,
            }),
            assigned: true,
            fee_loaded: true,
            fee,
        };
        assert_eq!(invocation(&facts).denial, denial);
        assert!(invocation(&facts).assign_billing);
        assert_eq!(
            invocation(&Invocation {
                assigned: false,
                ..facts
            })
            .action,
            InvocationAction::Assign
        );
    }

    #[rstest]
    #[case::no_database(false, false, false, Decision::ReadLegacyGrants)]
    #[case::database(false, true, false, Decision::ReadTarget)]
    #[case::managed_missing(true, true, true, Decision::Deny)]
    #[case::ordinary_missing(false, true, true, Decision::ReadLegacyGrants)]
    fn target_read_precedes_grants(
        #[case] registry_managed: bool,
        #[case] database_present: bool,
        #[case] target_loaded: bool,
        #[case] expected: Decision,
    ) {
        let facts = Facts {
            registry_managed,
            database_present,
            target_loaded,
            target: None,
            subject_present: false,
            key_hash_present: false,
            subject_managed: false,
            session: false,
            custom: false,
            authority_loaded: false,
            explicit_grants: None,
            legacy_loaded: false,
            legacy_grants: None,
            target_ids: None,
        };
        assert_eq!(decide(&facts), expected);
    }
}
