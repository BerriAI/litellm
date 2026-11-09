use crate::{
    Error,
    model_access::{self, Decision as ModelDecision, Facts, ModelRecord, Provider, Registry},
    model_helpers,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, strum::VariantArray)]
pub enum ReadKind {
    Compaction,
    Aliases,
    AliasTarget,
    Deployment,
    Groups,
    Names,
    Provider,
}

#[derive(Clone, Deserialize)]
pub struct Reading {
    pub kind: ReadKind,
    pub model: String,
    pub values: Vec<String>,
    pub value: Option<String>,
    pub team: Option<String>,
    pub provider: Option<Provider>,
}

#[derive(Clone, Deserialize)]
pub struct Evaluation {
    pub grants: Vec<String>,
    pub reads: Vec<Reading>,
    pub aliases: Option<model_access::Aliases>,
}

#[derive(Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Complete { decision: ModelDecision },
    Read { read: ReadKind, model: String },
}

enum Pending {
    Read(ReadKind, String),
    Invalid(Error),
}
impl From<Error> for Pending {
    fn from(error: Error) -> Self {
        Self::Invalid(error)
    }
}

struct Reader<'a> {
    reads: &'a [Reading],
    index: usize,
}
impl<'a> Reader<'a> {
    fn read(&mut self, kind: ReadKind, model: &str) -> Result<&'a Reading, Pending> {
        let Some(value) = self.reads.get(self.index) else {
            return Err(Pending::Read(kind, model.to_owned()));
        };
        if value.kind != kind || value.model != model {
            return Err(Pending::Invalid(Error::InvalidAnswer));
        }
        self.index += 1;
        Ok(value)
    }
}

fn helper(
    model: &str,
    facts: &Facts,
    aliases: bool,
    reader: &mut Reader<'_>,
) -> Result<bool, Pending> {
    let mut helper = model_helpers::Facts {
        kind: model_helpers::Kind::Helper,
        model: model.to_owned(),
        grants: Some(facts.grants.clone()),
        team_id: facts.team_id.clone(),
        team_aliases: if aliases {
            facts.aliases.team.keys().cloned().collect()
        } else {
            vec![]
        },
        router_present: facts
            .registry
            .helper_present
            .unwrap_or(facts.registry.present),
        groups: None,
        names: None,
        provider_loaded: false,
        provider: None,
        provider_reads: Some(vec![]),
    };
    loop {
        match model_helpers::decide(&helper)? {
            model_helpers::Decision::Complete { allowed } => return Ok(allowed),
            model_helpers::Decision::ReadGroups => {
                helper.groups = Some(reader.read(ReadKind::Groups, model)?.values.clone())
            }
            model_helpers::Decision::ReadNames => {
                helper.names = Some(reader.read(ReadKind::Names, model)?.values.clone())
            }
            model_helpers::Decision::ReadProvider => {
                helper
                    .provider_reads
                    .as_mut()
                    .ok_or(Error::InvalidAnswer)?
                    .push(reader.read(ReadKind::Provider, model)?.provider.clone());
            }
        }
    }
}

fn check_one(model: &str, facts: &Facts, reader: &mut Reader<'_>) -> Result<bool, Pending> {
    let compaction = reader.read(ReadKind::Compaction, model)?.value.clone();
    let mut registry = Registry {
        present: facts.registry.present,
        ..Registry::default()
    };
    if let Some(parent) = compaction {
        registry.compaction_parents.insert(model.to_owned(), parent);
    }
    if !facts.aliases.global.contains_key(model) && registry.present {
        let aliases = reader.read(ReadKind::Aliases, model)?;
        if aliases.values.iter().any(|alias| alias == model)
            && let Some(target) = &reader.read(ReadKind::AliasTarget, model)?.value
        {
            registry.aliases.insert(model.to_owned(), target.clone());
        }
    }
    let after_team = facts.aliases.team.get(model).map_or(model, String::as_str);
    let after_key = facts
        .aliases
        .key
        .get(after_team)
        .map_or(after_team, String::as_str);
    let after_global = facts
        .aliases
        .global
        .get(after_key)
        .map_or(after_key, String::as_str);
    let dispatched = facts
        .aliases
        .key
        .get(after_global)
        .map_or(after_global, String::as_str);
    let key_alias = after_key != after_team || dispatched != after_global;
    if !key_alias
        && registry.present
        && facts
            .team_id
            .as_ref()
            .is_some_and(|team| model.starts_with(&format!("model_name_{team}_")))
    {
        let deployment = reader.read(ReadKind::Deployment, model)?;
        registry.records.insert(
            model.to_owned(),
            ModelRecord {
                deployment_team: deployment.team.clone(),
                deployment_public_name: deployment.value.clone(),
                ..ModelRecord::default()
            },
        );
    }
    let (candidates, owning) =
        model_access::candidate_names(model, &facts.aliases, &registry, facts.team_id.as_deref());
    for candidate in candidates {
        if helper(&candidate, facts, true, reader)? {
            return Ok(true);
        }
    }
    for candidate in owning {
        if helper(&candidate, facts, false, reader)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn check(facts: &Facts, reader: &mut Reader<'_>) -> Result<ModelDecision, Pending> {
    if facts.fallback_depth >= facts.max_depth {
        return Ok(ModelDecision::Depth { model: None });
    }
    for model in &facts.requested {
        if facts.is_list && facts.fallback_depth + 1 >= facts.max_depth {
            return Ok(ModelDecision::Depth {
                model: Some(model.clone()),
            });
        }
        if !check_one(model, facts, reader)? {
            return Ok(ModelDecision::Deny {
                model: model.clone(),
                grants: facts.grants.clone(),
                object_type: facts.object_type,
            });
        }
    }
    Ok(ModelDecision::Allow)
}

pub fn decide(facts: &Facts, reads: &[Reading]) -> Result<Decision, Error> {
    let mut reader = Reader { reads, index: 0 };
    match check(facts, &mut reader) {
        Ok(decision) if reader.index == reads.len() => Ok(Decision::Complete { decision }),
        Ok(_) => Err(Error::InvalidAnswer),
        Err(Pending::Read(read, model)) => Ok(Decision::Read { read, model }),
        Err(Pending::Invalid(error)) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    #[rstest]
    #[case::empty(vec![], true)]
    #[case::model(vec!["model".to_owned()], false)]
    fn empty_list_never_reads_registry(#[case] requested: Vec<String>, #[case] complete: bool) {
        let facts = Facts {
            requested,
            grants: vec![],
            team_id: None,
            aliases: Default::default(),
            registry: Default::default(),
            object_type: model_access::ObjectType::Key,
            fallback_depth: 0,
            max_depth: 5,
            is_list: true,
        };
        assert_eq!(
            matches!(decide(&facts, &[]).unwrap(), Decision::Complete { .. }),
            complete
        );
    }
}
