//! The read-only hierarchy selected by a request, separate from its caller's affiliation.

use std::collections::BTreeMap;

use jiff::Timestamp;

use crate::record::BudgetWindow;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CustomerIdentity {
    pub organization_alias: Option<String>,
    pub member_id: Option<String>,
    pub team_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CustomerHierarchy {
    pub organization: CustomerScope,
    pub member: Option<CustomerScope>,
    pub selected_team: Option<usize>,
    pub teams: Vec<CustomerScope>,
}

impl CustomerHierarchy {
    pub fn team(&self) -> Option<&CustomerScope> {
        self.selected_team.and_then(|index| self.teams.get(index))
    }

    fn select_team(&mut self, id: &str) -> Result<(), CustomerReadError> {
        self.selected_team = Some(
            self.teams
                .iter()
                .position(|team| team.id == id)
                .ok_or(CustomerReadError::TeamNotInMember)?,
        );
        Ok(())
    }
}

impl CustomerIdentity {
    pub fn select(
        &self,
        org: Option<CustomerHierarchy>,
        member: Option<CustomerHierarchy>,
    ) -> Result<Option<CustomerHierarchy>, CustomerReadError> {
        let Some(mut hierarchy) = member else {
            return if self.team_id.is_some() {
                Err(CustomerReadError::TeamNotInMember)
            } else {
                Ok(org)
            };
        };
        if org.is_some_and(|org| org.organization.id != hierarchy.organization.id) {
            return Err(CustomerReadError::MemberOutsideOrganization);
        }
        if let Some(team) = self.team_id.as_deref() {
            hierarchy.select_team(team)?;
        }
        Ok(Some(hierarchy))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CustomerScope {
    pub id: String,
    pub alias: Option<String>,
    pub spend: f64,
    pub budget_reset_at: Option<Timestamp>,
    pub max_budget: Option<f64>,
    pub budget_limits: Vec<BudgetWindow>,
    pub tpm_limit: Option<i64>,
    pub rpm_limit: Option<i64>,
    pub allowed_models: Option<Vec<String>>,
    pub model_tpm_limit: Option<BTreeMap<String, i64>>,
    pub model_rpm_limit: Option<BTreeMap<String, i64>>,
    pub cap_policy: Option<CustomerCapPolicy>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::VariantArray)]
pub enum CustomerCapPolicy {
    PauseUntilReset,
    PauseAllowIncrease,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error, strum::VariantArray)]
pub enum CustomerReadError {
    #[error("the customer hierarchy could not be read")]
    Unread,
    #[error("the customer organization was not found")]
    OrganizationMissing,
    #[error("the customer organization alias is shared")]
    OrganizationAmbiguous,
    #[error("the customer member was not found")]
    MemberMissing,
    #[error("the member does not belong to the selected customer organization")]
    MemberOutsideOrganization,
    #[error("the team does not belong to the selected customer member")]
    TeamNotInMember,
}
