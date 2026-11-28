//! Model names and precedence for a per-model budget.

use std::{cmp::Reverse, collections::BTreeMap};

use litellm_gateway_auth_core::record::ModelBudget;
use litellm_model_catalog::{Catalog, bedrock_base_model};
use litellm_redis_store::model_budgets::Period;
use litellm_router_core::registry::{DeploymentFilter, Registry};

pub(super) type Entries = BTreeMap<String, ModelBudget>;

/// One resolved budget. The configured name owns the counter even when a different name matched.
pub(super) struct Resolved<'a> {
    pub model: &'a str,
    pub duration: &'a str,
    pub period: Period,
    pub max: Option<f64>,
}

fn usable<'a>(model: &'a str, budget: &'a ModelBudget) -> Option<Resolved<'a>> {
    let ModelBudget::Config(config) = budget else {
        return None;
    };
    let duration = config.budget_duration.as_deref()?;
    Some(Resolved {
        model,
        duration,
        period: Period::parse(duration)?,
        max: config.max_budget.filter(|max| *max >= 0.0),
    })
}

fn names<'a>(catalog: &Catalog, model: &'a str) -> impl Iterator<Item = &'a str> {
    let base = bedrock_base_model(model);
    let bedrock = catalog
        .get(base)
        .and_then(|row| row.get("litellm_provider"))
        .and_then(serde_json::Value::as_str)
        .is_some_and(|provider| provider.starts_with("bedrock"));
    [
        Some(model),
        model.rsplit('/').next(),
        bedrock.then_some(base),
        bedrock
            .then(|| base.split_once('.').map(|(_, name)| name))
            .flatten(),
    ]
    .into_iter()
    .flatten()
}

fn candidates<'a>(catalog: &Catalog, registry: &'a Registry, model: &'a str) -> Vec<&'a str> {
    [Some(model), registry.alias_of(model)]
        .into_iter()
        .flatten()
        .flat_map(|name| names(catalog, name))
        .collect()
}

pub(super) fn latest(catalog: &Catalog, model: &str) -> Option<String> {
    let aliases = catalog.aliases();
    let Some((provider, name)) = model.split_once('/') else {
        return aliases.get(model).cloned();
    };
    let prefix = format!("{provider}/");
    let owner = aliases.get(model).or_else(|| {
        let owner = aliases.get(name)?;
        let row = catalog.get(owner)?;
        (row.get("litellm_provider")?.as_str()? == provider).then_some(owner)
    })?;
    let target = owner.strip_prefix(&prefix).unwrap_or(owner);
    (target != name).then(|| format!("{prefix}{target}"))
}

fn matches(pattern: &str, name: &str) -> bool {
    let Some((head, rest)) = pattern.split_once('*') else {
        return pattern == name;
    };
    let (middle, tail) = rest.rsplit_once('*').map_or(("", rest), |parts| parts);
    let Some(mut remaining) = name
        .strip_prefix(head)
        .and_then(|name| name.strip_suffix(tail))
    else {
        return false;
    };
    for piece in middle.split('*').filter(|piece| !piece.is_empty()) {
        let Some((_, tail)) = remaining.split_once(piece) else {
            return false;
        };
        remaining = tail;
    }
    true
}

fn groups(registry: &Registry, model: &str, team: Option<&str>) -> Vec<String> {
    let own = team
        .map(|team| {
            registry
                .deployments(&DeploymentFilter {
                    team_id: Some(team),
                    ..Default::default()
                })
                .into_iter()
                .filter(|deployment| {
                    deployment.model_info.team_public_model_name.as_deref() == Some(model)
                })
                .collect::<Vec<_>>()
        })
        .filter(|own| !own.is_empty());
    own.or_else(|| registry.group(model))
        .into_iter()
        .flatten()
        .flat_map(|deployment| deployment.model_info.access_groups.clone())
        .filter(|group| registry.group(group).is_none())
        .collect()
}

pub(super) fn resolve<'a>(
    entries: &'a Entries,
    key: Option<&Entries>,
    catalog: &Catalog,
    registry: &Registry,
    model: &str,
    team: Option<&str>,
) -> Option<Resolved<'a>> {
    let names = candidates(catalog, registry, model);
    let exact = |name: &str| {
        entries
            .get_key_value(name)
            .and_then(|(name, budget)| usable(name, budget))
    };
    let replaces = key.is_some_and(|key| {
        names
            .iter()
            .find_map(|name| usable(name, key.get(*name)?))
            .is_some_and(|budget| budget.max.is_some())
    });
    if !replaces && let Some(budget) = names.iter().find_map(|name| exact(name)) {
        return Some(budget);
    }
    let groups = groups(registry, model, team);
    if let Some(budget) = entries
        .keys()
        .filter(|name| !names.contains(&name.as_str()) && groups.contains(name))
        .find_map(|name| exact(name))
    {
        return Some(budget);
    }
    entries
        .iter()
        .filter(|(pattern, _)| {
            pattern.contains('*') && names.iter().any(|name| matches(pattern, name))
        })
        .filter_map(|(name, budget)| {
            let wildcards = name.bytes().filter(|byte| *byte == b'*').count();
            Some((
                (name.chars().count() - wildcards, Reverse(wildcards), name),
                usable(name, budget)?,
            ))
        })
        .max_by_key(|(precedence, _)| *precedence)
        .map(|(_, budget)| budget)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::a_glob_matches_multiple_pieces("a*b*c", "aa-bb-c", true)]
    #[case::a_glob_requires_the_end("a*b*c", "aa-bb-cd", false)]
    #[case::pieces_must_not_overlap("aa*aa", "aaa", false)]
    #[case::consecutive_wildcards("a**b", "ab", true)]
    fn glob_requires_literals_in_order(
        #[case] pattern: &str,
        #[case] model: &str,
        #[case] wanted: bool,
    ) {
        assert_eq!(matches(pattern, model), wanted);
    }
}
