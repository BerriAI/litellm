use std::cmp::Ordering::{Equal, Greater, Less};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Entry {
    pub name: String,
    pub max_budget: Option<f64>,
    pub duration: Option<String>,
    pub duration_seconds: Option<f64>,
}

impl Entry {
    fn usable(&self) -> bool {
        self.duration.is_some() && self.duration_seconds.is_some()
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct ResolveFacts {
    pub model: String,
    pub bedrock_base: String,
    pub bedrock_provider: String,
    pub entries: Vec<Entry>,
    pub key_entries: Vec<Entry>,
    pub team: bool,
    pub access_groups: Option<Vec<String>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Resolution {
    pub read_groups: bool,
    pub entry: Option<Entry>,
    pub named_model: bool,
}

fn candidates(facts: &ResolveFacts) -> Vec<&str> {
    let base_names = if facts.bedrock_provider.starts_with("bedrock") {
        vec![
            facts.bedrock_base.as_str(),
            facts.bedrock_base.split_once('.').map_or("", |(_, s)| s),
        ]
    } else {
        vec![]
    };
    std::iter::once(facts.model.as_str())
        .chain(std::iter::once(
            facts.model.rsplit('/').next().unwrap_or(&facts.model),
        ))
        .chain(base_names.into_iter().filter(|s| !s.is_empty()))
        .collect()
}

fn exact<'a>(entries: &'a [Entry], names: &[&str]) -> Option<&'a Entry> {
    names.iter().find_map(|name| {
        entries
            .iter()
            .find(|entry| entry.name == *name && entry.usable())
    })
}

pub fn resolve(facts: &ResolveFacts) -> Resolution {
    let names = candidates(facts);
    let key_replaces = facts.team
        && exact(&facts.key_entries, &names)
            .is_some_and(|entry| entry.max_budget.is_some_and(|limit| limit >= 0.0));
    if !key_replaces && let Some(entry) = exact(&facts.entries, &names) {
        return Resolution {
            read_groups: false,
            entry: Some(entry.clone()),
            named_model: true,
        };
    }
    let group_entries: Vec<&Entry> = facts
        .entries
        .iter()
        .filter(|entry| !entry.name.contains('*') && !names.contains(&entry.name.as_str()))
        .collect();
    if !group_entries.is_empty() && facts.access_groups.is_none() {
        return Resolution {
            read_groups: true,
            entry: None,
            named_model: false,
        };
    }
    let group = group_entries
        .into_iter()
        .filter(|entry| {
            entry.usable()
                && facts
                    .access_groups
                    .as_ref()
                    .is_some_and(|groups| groups.contains(&entry.name))
        })
        .min_by(|a, b| a.name.cmp(&b.name));
    if let Some(entry) = group {
        return Resolution {
            read_groups: false,
            entry: Some(entry.clone()),
            named_model: false,
        };
    }
    let entry = facts
        .entries
        .iter()
        .filter(|entry| {
            entry.usable()
                && !(key_replaces && names.contains(&entry.name.as_str()))
                && entry.name.contains('*')
                && names.iter().any(|name| glob(&entry.name, name))
        })
        .max_by_key(|entry| {
            let wildcards = entry.name.chars().filter(|c| *c == '*').count();
            (
                entry.name.chars().count() - wildcards,
                std::cmp::Reverse(wildcards),
                &entry.name,
            )
        })
        .cloned();
    Resolution {
        read_groups: false,
        entry,
        named_model: false,
    }
}

fn glob(pattern: &str, name: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == name;
    }
    let head = parts[0];
    let tail = parts[parts.len() - 1];
    if !name.starts_with(head) || !name.ends_with(tail) || head.len() + tail.len() > name.len() {
        return false;
    }
    parts[1..parts.len() - 1]
        .iter()
        .try_fold(head.len(), |position, piece| {
            name[position..name.len() - tail.len()]
                .find(piece)
                .map(|index| position + index + piece.len())
        })
        .is_some()
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, strum::VariantArray)]
pub enum Entity {
    Key,
    User,
    Team,
    TeamMember,
    EndUser,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Snapshot {
    pub max_budget: Option<f64>,
    pub local_spend: Option<f64>,
    pub legacy_spend: Option<f64>,
    pub shared_spend: Option<f64>,
    pub shared_started_at: Option<f64>,
    pub shared_window_seconds: Option<f64>,
    pub now: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct QueryFacts {
    pub entity: Entity,
    pub snapshot: Snapshot,
    pub duration_seconds: Option<f64>,
    pub process_uptime: f64,
    pub same_counter: bool,
    pub redis_present: bool,
    pub shared_read: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, strum::VariantArray)]
pub enum Read {
    Local,
    Legacy,
    Shared,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Decision {
    pub read: Option<Read>,
    pub denied: bool,
    pub current_spend: f64,
}

pub fn check(snapshot: &Snapshot) -> Decision {
    let local = snapshot.local_spend.unwrap_or(0.0) + snapshot.legacy_spend.unwrap_or(0.0);
    let current_spend = match (
        snapshot.shared_spend,
        snapshot.shared_started_at,
        snapshot.shared_window_seconds,
    ) {
        (Some(spend), Some(start), Some(seconds))
            if snapshot.now < start + seconds && spend > local =>
        {
            spend
        }
        _ => local,
    };
    Decision {
        read: None,
        denied: snapshot
            .max_budget
            .is_some_and(|limit| limit.partial_cmp(&0.0) != Some(Less) && current_spend >= limit),
        current_spend,
    }
}

pub fn query(facts: &QueryFacts) -> Decision {
    let done = check(&facts.snapshot);
    let Some(limit) = facts.snapshot.max_budget else {
        return done;
    };
    if limit < 0.0 {
        return done;
    }
    let read = if facts.snapshot.local_spend.is_none() {
        Some(Read::Local)
    } else if matches!(facts.entity, Entity::Key | Entity::EndUser)
        && facts
            .duration_seconds
            .is_some_and(|seconds| facts.process_uptime < seconds)
        && !facts.same_counter
        && facts.snapshot.legacy_spend.is_none()
    {
        Some(Read::Legacy)
    } else if !matches!(
        (facts.snapshot.local_spend.unwrap_or(0.0) + facts.snapshot.legacy_spend.unwrap_or(0.0))
            .partial_cmp(&limit),
        Some(Greater | Equal)
    ) && !facts.redis_present
        && facts.entity != Entity::EndUser
        && !facts.shared_read
    {
        Some(Read::Shared)
    } else {
        None
    };
    Decision { read, ..done }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::empty("*", "", true)]
    #[case::unicode("é*界", "é世世界", true)]
    #[case::overlap("ab*bc", "abc", false)]
    #[case::middle("a*b*c", "axybzc", true)]
    #[case::not_regex("a?*", "ab", false)]
    fn wildcard(#[case] pattern: &str, #[case] name: &str, #[case] matches: bool) {
        assert_eq!(glob(pattern, name), matches);
    }
}
