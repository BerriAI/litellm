use crate::{Error, model_access::Provider, patterns};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Deserialize, PartialEq, Eq, strum::VariantArray)]
pub enum Kind {
    Group,
    Helper,
}

#[derive(Clone, Deserialize)]
pub struct Facts {
    pub kind: Kind,
    pub model: String,
    pub grants: Option<Vec<String>>,
    pub team_id: Option<String>,
    pub team_aliases: Vec<String>,
    pub router_present: bool,
    pub groups: Option<Vec<String>>,
    pub names: Option<Vec<String>>,
    pub provider_loaded: bool,
    pub provider: Option<Provider>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Complete { allowed: bool },
    ReadGroups,
    ReadNames,
    ReadProvider,
}

pub fn decide(facts: &Facts) -> Result<Decision, Error> {
    let done = |allowed| Ok(Decision::Complete { allowed });
    if facts.kind == Kind::Group
        && facts
            .grants
            .as_ref()
            .is_none_or(|grants| grants.contains(&facts.model))
    {
        return done(true);
    }
    let grants = facts.grants.as_ref().ok_or(Error::InvalidAnswer)?;
    if facts.router_present && facts.groups.is_none() {
        return Ok(Decision::ReadGroups);
    }
    let groups = facts.groups.as_deref().unwrap_or_default();
    if facts.kind == Kind::Group {
        return done(grants.iter().any(|grant| groups.contains(grant)));
    }
    let expanded = if facts.router_present
        && facts.team_id.is_some()
        && grants.iter().any(|grant| grant == "all-team-models")
    {
        let Some(names) = &facts.names else {
            return Ok(Decision::ReadNames);
        };
        let mut seen = BTreeSet::new();
        let values: Vec<_> = grants
            .iter()
            .filter(|grant| *grant != "all-team-models")
            .chain(names)
            .filter(|grant| seen.insert(*grant))
            .cloned()
            .collect();
        if values.is_empty() {
            grants.clone()
        } else {
            values
        }
    } else {
        grants.clone()
    };
    if expanded.iter().any(|grant| groups.contains(grant))
        || facts.team_aliases.contains(&facts.model)
    {
        return done(true);
    }
    let filtered: Vec<_> = expanded
        .iter()
        .filter(|grant| !groups.contains(grant))
        .collect();
    let wildcards: Vec<_> = filtered
        .iter()
        .filter(|grant| grant.contains('*'))
        .collect();
    for pattern in &wildcards {
        if patterns::model_pattern(&facts.model, pattern)? {
            return done(true);
        }
    }
    if !wildcards.is_empty() {
        if !facts.provider_loaded {
            return Ok(Decision::ReadProvider);
        }
        if let Some(provider) = &facts.provider
            && (provider.stripped_model != facts.model || !facts.model.contains('/'))
        {
            for pattern in wildcards {
                if patterns::model_pattern(
                    &format!("{}/{}", provider.provider, provider.stripped_model),
                    pattern,
                )? {
                    return done(true);
                }
            }
        }
    }
    done(
        (filtered.is_empty() && expanded.is_empty())
            || filtered.iter().any(|grant| {
                matches!(grant.as_str(), "*" | "all-proxy-models") || **grant == facts.model
            }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::group(Kind::Group, Decision::Complete { allowed: true })]
    #[case::helper(Kind::Helper, Decision::ReadGroups)]
    fn direct_match_read_order(#[case] kind: Kind, #[case] expected: Decision) {
        let facts = Facts {
            kind,
            model: "model".into(),
            grants: Some(vec!["model".into()]),
            team_id: None,
            team_aliases: vec![],
            router_present: true,
            groups: None,
            names: None,
            provider_loaded: false,
            provider: None,
        };
        assert_eq!(decide(&facts).unwrap(), expected);
    }
}
