//! Reads customer identities and shapes the cached organization, team and member scopes.

use jiff::Timestamp;
use litellm_python_compat::datetime::IsoDateTime;
use litellm_redis_store::auth_cache::{
    BudgetLimitEntry, CustomerHierarchyEntry, CustomerScopeEntry, CustomerScopeKey, ExcludedIfNone,
    Ints,
};
use litellm_repositories::{
    Budget, CustomerHierarchyRepository, CustomerMemberRow, CustomerOrgRow,
};

use crate::{
    AliasMatch, Error, budget_limit,
    column::{nullable, time},
};

pub struct CustomerHierarchyDatabase {
    repository: CustomerHierarchyRepository,
}

pub struct LoadedCustomerMember {
    pub hierarchy: CustomerHierarchyEntry,
    pub next_override_expiry: Option<IsoDateTime>,
}

impl CustomerHierarchyDatabase {
    pub fn new(repository: CustomerHierarchyRepository) -> Self {
        Self { repository }
    }

    pub async fn member(
        &self,
        id: &str,
        now: Timestamp,
    ) -> Result<Option<LoadedCustomerMember>, Error> {
        self.repository
            .member(id)
            .await?
            .map(|row| member(row, now))
            .transpose()
    }

    pub async fn organization_alias(
        &self,
        alias: &str,
        now: Timestamp,
    ) -> Result<AliasMatch<CustomerHierarchyEntry>, Error> {
        let rows = self.repository.organizations(alias).await?;
        match rows.len() {
            0 => Ok(AliasMatch::Absent),
            1 => organization(rows.into_iter().next().expect("one organization"), now)
                .map(AliasMatch::Found),
            _ => Ok(AliasMatch::Ambiguous),
        }
    }
}

fn organization(row: CustomerOrgRow, now: Timestamp) -> Result<CustomerHierarchyEntry, Error> {
    let org = scope(
        CustomerScopeKey::Organization,
        row.customer_org_id,
        None,
        row.spend,
        row.budget,
        None,
        now,
    )?;
    Ok(CustomerHierarchyEntry {
        customer_org_id: Some(org.entity_id.clone()),
        customer_org_alias: row.alias,
        customer_team_id: None,
        customer_team_alias: None,
        customer_member_id: None,
        customer_member_alias: None,
        app_id: ExcludedIfNone(None),
        app_group_id: ExcludedIfNone(None),
        app_alias: ExcludedIfNone(None),
        app_customer_margin: ExcludedIfNone(None),
        scopes: vec![org],
        team_scopes: Vec::new(),
        unread: Vec::new(),
    })
}

fn member(row: CustomerMemberRow, now: Timestamp) -> Result<LoadedCustomerMember, Error> {
    let mut hierarchy = organization(row.parent_org, now)?;
    let expires = time("budget_override_expires_at", row.budget_override_expires_at)?;
    let mut next_override_expiry = future_expiry(expires, now);
    for team in row.teams {
        let expires = time(
            "budget_override_expires_at",
            team.budget_override_expires_at,
        )?;
        if let Some(expiry) = future_expiry(expires, now)
            && next_override_expiry
                .is_none_or(|old| expiry.utc_microseconds() < old.utc_microseconds())
        {
            next_override_expiry = Some(expiry);
        }
        hierarchy.team_scopes.push(scope(
            CustomerScopeKey::Team,
            team.customer_team_id,
            team.alias,
            team.spend,
            team.budget,
            team.budget_override
                .filter(|_| expires.is_none() || future_expiry(expires, now).is_some()),
            now,
        )?);
    }
    if let Some(team) = hierarchy.team_scopes.first() {
        hierarchy.customer_team_id = Some(team.entity_id.clone());
        hierarchy.customer_team_alias = team.alias.clone();
        hierarchy.scopes.push(team.clone());
    }
    hierarchy.scopes.push(scope(
        CustomerScopeKey::Member,
        row.customer_member_id.clone(),
        None,
        row.spend,
        row.budget,
        row.budget_override
            .filter(|_| expires.is_none() || next_override_active(expires, now)),
        now,
    )?);
    hierarchy.customer_member_id = Some(row.customer_member_id);
    hierarchy.customer_member_alias = row.alias;
    Ok(LoadedCustomerMember {
        hierarchy,
        next_override_expiry,
    })
}

fn next_override_active(expiry: Option<IsoDateTime>, now: Timestamp) -> bool {
    future_expiry(expiry, now).is_some()
}

fn future_expiry(expiry: Option<IsoDateTime>, now: Timestamp) -> Option<IsoDateTime> {
    expiry.filter(|expiry| expiry.utc_microseconds() > now.as_microsecond())
}

fn max_budget(
    budget: &Budget,
    inherited: Option<f64>,
    now: Timestamp,
) -> Result<Option<f64>, Error> {
    let expiry = time("temp_budget_expiry", budget.temp_budget_expiry)?;
    let increase = budget
        .temp_budget_increase
        .filter(|_| future_expiry(expiry, now).is_some());
    let base = budget.max_budget.or(inherited);
    if budget
        .max_budget
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        return Err(Error::row("max_budget"));
    }
    let effective = base.map(|base| base + increase.unwrap_or_default());
    if effective.is_some_and(|value| !value.is_finite() || value < 0.0) {
        return Err(Error::row("max_budget"));
    }
    Ok(effective)
}

fn scope(
    key: CustomerScopeKey,
    entity_id: String,
    alias: Option<String>,
    spend: f64,
    template: Option<Budget>,
    override_budget: Option<Budget>,
    now: Timestamp,
) -> Result<CustomerScopeEntry, Error> {
    let template = template.as_ref();
    let override_budget = override_budget.as_ref();
    let inherited_max = template
        .map(|budget| max_budget(budget, None, now))
        .transpose()?
        .flatten();
    let max_override = override_budget
        .map(|budget| max_budget(budget, inherited_max, now))
        .transpose()?
        .flatten();
    let reset_budget = override_budget
        .filter(|budget| budget.budget_duration.is_some())
        .or(template);
    let windows = |budget: Option<&Budget>| -> Result<Vec<BudgetLimitEntry>, Error> {
        Ok(budget_limit::windows(nullable(
            "budget_limits",
            budget.and_then(|budget| budget.budget_limits.as_ref()),
        )?)
        .unwrap_or_default())
    };
    let template_windows = windows(template)?;
    let override_windows = windows(override_budget)?;
    let mut budget_limits = if max_override.is_some() {
        Vec::new()
    } else {
        template_windows
    };
    for window in override_windows {
        match budget_limits
            .iter_mut()
            .find(|old| old.budget_duration == window.budget_duration)
        {
            Some(old) => *old = window,
            None => budget_limits.push(window),
        }
    }
    let tpm_limit = override_budget
        .and_then(|budget| budget.tpm_limit)
        .or(template.and_then(|budget| budget.tpm_limit));
    let rpm_limit = override_budget
        .and_then(|budget| budget.rpm_limit)
        .or(template.and_then(|budget| budget.rpm_limit));
    let allowed_models = |budget: Option<&Budget>| {
        budget
            .and_then(|budget| budget.allowed_models.clone())
            .filter(|models| !models.is_empty())
    };
    let map = |column, template: Option<Ints>, over: Option<Ints>| -> Result<Option<Ints>, Error> {
        if template
            .iter()
            .chain(over.iter())
            .any(|map| map.0.iter().any(|(_, value)| *value < 0))
        {
            return Err(Error::row(column));
        }
        Ok(over.or(template))
    };
    let caps = |budget: Option<&Budget>| {
        budget
            .and_then(|budget| budget.cap_policy.as_deref())
            .map(|cap| cap.parse().map_err(|_| Error::row("cap_policy")))
            .transpose()
    };
    let cap_policy = caps(override_budget)?.or(caps(template)?);
    if tpm_limit.is_some_and(|value| value < 0) || rpm_limit.is_some_and(|value| value < 0) {
        return Err(Error::row("rate_limit"));
    }
    Ok(CustomerScopeEntry {
        key,
        entity_id,
        alias,
        spend,
        budget_reset_at: time(
            "budget_reset_at",
            reset_budget.and_then(|budget| budget.budget_reset_at),
        )?,
        max_budget: max_override.or(inherited_max),
        budget_limits,
        soft_budget_duration: None,
        tpm_limit,
        rpm_limit,
        allowed_models: allowed_models(override_budget).or_else(|| allowed_models(template)),
        model_tpm_limit: map(
            "model_tpm_limit",
            nullable(
                "model_tpm_limit",
                template.and_then(|budget| budget.model_tpm_limit.as_ref()),
            )?,
            nullable(
                "model_tpm_limit",
                override_budget.and_then(|budget| budget.model_tpm_limit.as_ref()),
            )?,
        )?,
        model_rpm_limit: map(
            "model_rpm_limit",
            nullable(
                "model_rpm_limit",
                template.and_then(|budget| budget.model_rpm_limit.as_ref()),
            )?,
            nullable(
                "model_rpm_limit",
                override_budget.and_then(|budget| budget.model_rpm_limit.as_ref()),
            )?,
        )?,
        cap_policy,
        unread: Vec::new(),
    })
}
