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

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct Decision {
    pub denial: Option<Denial>,
    pub organization_role: Option<String>,
    pub organization_ids: Vec<String>,
}

pub fn decide(facts: Facts) -> Decision {
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
    let denial = if !facts.user_present || facts.user_role.as_deref() == Some("proxy_admin") {
        None
    } else if facts.route == "/organization/new" {
        Some(Denial::CreateOrganization)
    } else if facts.org_admin_routes.contains(&facts.route) {
        if facts.memberships.is_none() {
            Some(Denial::NoMemberships)
        } else if facts.organization_id.is_none() {
            Some(Denial::MissingOrganization)
        } else if role.is_none() {
            Some(Denial::MissingRole)
        } else if role.as_deref() != Some("org_admin") {
            Some(Denial::WrongRole)
        } else {
            None
        }
    } else if facts.route == "/team/new" && !memberships.is_empty() {
        if facts.organization_id.is_none() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

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
                org_admin_routes: vec!["/organization/update".into()]
            })
            .denial,
            expected
        );
    }
}
