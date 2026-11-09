use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize)]
pub struct Membership {
    pub organization_id: Option<String>,
    pub role: Option<String>,
}

#[derive(Deserialize)]
pub struct Facts {
    pub route: String,
    pub user_present: bool,
    pub user_role: Option<String>,
    pub memberships: Option<Vec<Membership>>,
    pub organization_id: Option<String>,
    pub org_admin_routes: Vec<String>,
    #[serde(default)]
    pub organization_id_present: bool,
    #[serde(default)]
    pub identifier_pending: bool,
    #[serde(default)]
    pub hash_failure: Option<String>,
    #[serde(default)]
    pub request_pending: bool,
    #[serde(default)]
    pub memberships_pending: bool,
}

#[derive(Debug, PartialEq, Eq, Serialize, strum::VariantArray)]
pub enum Denial {
    CreateOrganization,
    NoMemberships,
    MissingOrganization,
    MissingRole,
    WrongRole,
    TeamMissingOrganization,
    TeamWrongRole,
}

#[derive(Debug, Default, PartialEq, Eq, Serialize)]
pub struct Decision {
    pub denial: Option<Denial>,
    pub organization_role: Option<String>,
    pub organization_ids: Vec<String>,
    pub read_identifier: bool,
    pub hash_failure: Option<String>,
    pub read_request: bool,
    pub read_memberships: bool,
}

pub fn decide(facts: Facts) -> Decision {
    if facts.user_present && facts.request_pending {
        return Decision {
            read_request: true,
            ..Decision::default()
        };
    }
    if facts.user_present
        && facts.user_role.as_deref() != Some("proxy_admin")
        && facts.route != "/organization/new"
        && (facts.org_admin_routes.contains(&facts.route) || facts.route == "/team/new")
        && facts.memberships_pending
    {
        return Decision {
            read_memberships: true,
            ..Decision::default()
        };
    }
    let memberships = facts.memberships.as_deref().unwrap_or(&[]);
    let ids = memberships
        .iter()
        .filter_map(|item| item.organization_id.clone())
        .collect();
    let role = memberships
        .iter()
        .rev()
        .find(|item| {
            item.organization_id.is_some() && item.organization_id == facts.organization_id
        })
        .and_then(|item| item.role.clone());
    let present = facts.organization_id_present || facts.organization_id.is_some();
    let needs_lookup = facts.user_present
        && facts.user_role.as_deref() != Some("proxy_admin")
        && facts.route != "/organization/new"
        && present
        && ((facts.org_admin_routes.contains(&facts.route) && facts.memberships.is_some())
            || facts.route == "/team/new" && !memberships.is_empty());
    if needs_lookup && (facts.identifier_pending || facts.hash_failure.is_some()) {
        return Decision {
            denial: None,
            organization_role: role,
            organization_ids: ids,
            read_identifier: facts.identifier_pending,
            hash_failure: facts.hash_failure,
            ..Decision::default()
        };
    }
    let denial = if !facts.user_present || facts.user_role.as_deref() == Some("proxy_admin") {
        None
    } else if facts.route == "/organization/new" {
        Some(Denial::CreateOrganization)
    } else if facts.org_admin_routes.contains(&facts.route) {
        if facts.memberships.is_none() {
            Some(Denial::NoMemberships)
        } else if !present {
            Some(Denial::MissingOrganization)
        } else if role.is_none() {
            Some(Denial::MissingRole)
        } else if role.as_deref() != Some("org_admin") {
            Some(Denial::WrongRole)
        } else {
            None
        }
    } else if facts.route == "/team/new" && !memberships.is_empty() {
        if !present {
            Some(Denial::TeamMissingOrganization)
        } else if role.as_deref() != Some("org_admin") {
            Some(Denial::TeamWrongRole)
        } else {
            None
        }
    } else {
        None
    };
    Decision {
        denial,
        organization_role: role,
        organization_ids: ids,
        read_identifier: false,
        hash_failure: None,
        ..Decision::default()
    }
}

pub fn user_is_admin(memberships: &[Membership], organizations: &[String]) -> bool {
    !organizations.is_empty()
        && organizations.iter().all(|id| {
            memberships.iter().any(|item| {
                item.organization_id.as_ref() == Some(id)
                    && item.role.as_deref() == Some("org_admin")
            })
        })
}

#[derive(Deserialize)]
pub struct Identifier {
    pub text: Option<String>,
    pub hash_failure: Option<String>,
}

#[derive(Deserialize)]
pub struct AdminFacts {
    pub memberships: Option<Vec<Membership>>,
    pub request_read: bool,
    pub singular_present: bool,
    pub plural_count: usize,
    pub candidate_index: usize,
    pub candidate: Option<Identifier>,
}

#[derive(Debug, PartialEq, Eq, Serialize, strum::VariantArray)]
pub enum AdminAction {
    ReadRequest,
    ReadSingular,
    ReadPlural,
    Allow,
    Deny,
    HashFailure,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct AdminDecision {
    pub action: AdminAction,
    pub candidate_index: usize,
    pub plural_index: usize,
    pub hash_failure: Option<String>,
}

pub fn admin(facts: &AdminFacts) -> AdminDecision {
    let result = |action| AdminDecision {
        action,
        candidate_index: 0,
        plural_index: 0,
        hash_failure: None,
    };
    let Some(memberships) = &facts.memberships else {
        return result(AdminAction::Deny);
    };
    if !facts.request_read {
        return result(AdminAction::ReadRequest);
    }
    let count = usize::from(facts.singular_present) + facts.plural_count;
    if count == 0 {
        return result(AdminAction::Deny);
    }
    let next = if let Some(candidate) = &facts.candidate {
        if let Some(message) = &candidate.hash_failure {
            return AdminDecision {
                hash_failure: Some(message.clone()),
                ..result(AdminAction::HashFailure)
            };
        }
        if !candidate.text.as_ref().is_some_and(|id| {
            memberships.iter().any(|item| {
                item.organization_id.as_ref() == Some(id)
                    && item.role.as_deref() == Some("org_admin")
            })
        }) {
            return result(AdminAction::Deny);
        }
        facts.candidate_index + 1
    } else {
        0
    };
    if next == count {
        return result(AdminAction::Allow);
    }
    AdminDecision {
        action: if facts.singular_present && next == 0 {
            AdminAction::ReadSingular
        } else {
            AdminAction::ReadPlural
        },
        candidate_index: next,
        plural_index: next.saturating_sub(usize::from(facts.singular_present)),
        hash_failure: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::no_memberships(false, false, None, None, AdminAction::Deny)]
    #[case::read_request(true, false, None, None, AdminAction::ReadRequest)]
    #[case::read_singular(true, true, None, None, AdminAction::ReadSingular)]
    #[case::allowed(true, true, Some("org"), None, AdminAction::Allow)]
    #[case::wrong_org(true, true, Some("other"), None, AdminAction::Deny)]
    #[case::unhashable(
        true,
        true,
        None,
        Some("unhashable type: 'list'"),
        AdminAction::HashFailure
    )]
    fn raw_admin_stages(
        #[case] present: bool,
        #[case] request_read: bool,
        #[case] id: Option<&str>,
        #[case] failure: Option<&str>,
        #[case] expected: AdminAction,
    ) {
        let memberships = present.then(|| {
            vec![Membership {
                organization_id: Some("org".into()),
                role: Some("org_admin".into()),
            }]
        });
        let candidate = (id.is_some() || failure.is_some()).then(|| Identifier {
            text: id.map(str::to_owned),
            hash_failure: failure.map(str::to_owned),
        });
        assert_eq!(
            admin(&AdminFacts {
                memberships,
                request_read,
                singular_present: true,
                plural_count: 0,
                candidate_index: 0,
                candidate
            })
            .action,
            expected
        );
    }

    #[rstest]
    #[case::none(None, Some(Denial::NoMemberships))]
    #[case::empty(Some(vec![]), Some(Denial::MissingOrganization))]
    fn membership_presence(
        #[case] memberships: Option<Vec<Membership>>,
        #[case] expected: Option<Denial>,
    ) {
        assert_eq!(
            decide(Facts {
                route: "/organization/update".into(),
                user_present: true,
                user_role: None,
                memberships,
                organization_id: None,
                org_admin_routes: vec!["/organization/update".into()],
                organization_id_present: false,
                identifier_pending: false,
                hash_failure: None,
                request_pending: false,
                memberships_pending: false,
            })
            .denial,
            expected
        );
    }
}
