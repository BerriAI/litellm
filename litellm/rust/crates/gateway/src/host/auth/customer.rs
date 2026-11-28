//! Projects the shared customer cache into the request's read-only context.

use jiff::Timestamp;
use litellm_gateway_auth_core::customer::{
    CustomerCapPolicy, CustomerHierarchy, CustomerReadError, CustomerScope,
};
use litellm_redis_store::auth_cache::{
    CustomerCapPolicy as CachedCap, CustomerHierarchyEntry, CustomerScopeEntry, CustomerScopeKey,
};

use super::affiliation::windows;

pub(super) fn project(
    entry: CustomerHierarchyEntry,
) -> Result<CustomerHierarchy, CustomerReadError> {
    let unread = CustomerReadError::Unread;
    if entry.app_id.0.is_some()
        || entry.app_group_id.0.is_some()
        || entry.app_alias.0.is_some()
        || entry.app_customer_margin.0.is_some()
        || !entry.unread.is_empty()
    {
        return Err(unread);
    }
    let mut scopes = entry.scopes.into_iter();
    let mut org = scopes.next().ok_or(unread)?;
    if org.key != CustomerScopeKey::Organization
        || entry.customer_org_id.as_ref() != Some(&org.entity_id)
    {
        return Err(unread);
    }
    org.alias = entry.customer_org_alias;

    let teams = entry
        .team_scopes
        .into_iter()
        .map(|team| match team.key {
            CustomerScopeKey::Team => scope(team),
            _ => Err(unread),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let selected_team = if let Some(id) = entry.customer_team_id.as_ref() {
        let index = teams.iter().position(|team| &team.id == id).ok_or(unread)?;
        let selected = scopes.next().ok_or(unread)?;
        if selected.key != CustomerScopeKey::Team || teams[index] != scope(selected)? {
            return Err(unread);
        }
        Some(index)
    } else if teams.is_empty() {
        None
    } else {
        return Err(unread);
    };

    let member = entry
        .customer_member_id
        .map(|id| {
            let mut member = scopes.next().ok_or(unread)?;
            if member.key != CustomerScopeKey::Member || member.entity_id != id {
                return Err(unread);
            }
            member.alias = entry.customer_member_alias;
            scope(member)
        })
        .transpose()?;
    if scopes.next().is_some() || (member.is_none() && selected_team.is_some()) {
        return Err(unread);
    }
    Ok(CustomerHierarchy {
        organization: scope(org)?,
        member,
        selected_team,
        teams,
    })
}

fn scope(entry: CustomerScopeEntry) -> Result<CustomerScope, CustomerReadError> {
    if !entry.unread.is_empty() || entry.soft_budget_duration.is_some() {
        return Err(CustomerReadError::Unread);
    }
    let cap_policy = entry.cap_policy.map(|cap| match cap {
        CachedCap::PauseUntilReset => CustomerCapPolicy::PauseUntilReset,
        CachedCap::PauseAllowIncrease => CustomerCapPolicy::PauseAllowIncrease,
    });
    Ok(CustomerScope {
        id: entry.entity_id,
        alias: entry.alias,
        spend: entry.spend,
        budget_reset_at: entry
            .budget_reset_at
            .map(|reset| Timestamp::from_microsecond(reset.utc_microseconds()))
            .transpose()
            .map_err(|_| CustomerReadError::Unread)?,
        max_budget: entry.max_budget,
        budget_limits: windows(Some(&entry.budget_limits))
            .map_err(|_| CustomerReadError::Unread)?,
        tpm_limit: entry.tpm_limit,
        rpm_limit: entry.rpm_limit,
        allowed_models: entry.allowed_models,
        model_tpm_limit: entry
            .model_tpm_limit
            .map(|limits| limits.0.into_iter().collect()),
        model_rpm_limit: entry
            .model_rpm_limit
            .map(|limits| limits.0.into_iter().collect()),
        cap_policy,
    })
}
