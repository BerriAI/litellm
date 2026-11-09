use crate::Error;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Default, Deserialize)]
pub struct Facts {
    pub requested: Vec<String>,
    pub aliases: BTreeMap<String, String>,
    pub team_id: Option<String>,
    pub router_present: bool,
    pub names: Option<Vec<Vec<String>>>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    ReadNames { count: usize },
    Complete { models: Vec<String> },
}

pub fn decide(facts: &Facts) -> Result<Decision, Error> {
    let prefix = format!("model_name_{}_", facts.team_id.as_deref().unwrap_or("None"));
    let needs_names = |target: &str| facts.router_present && target.starts_with(&prefix);
    let count = facts
        .requested
        .iter()
        .filter_map(|model| facts.aliases.get(model))
        .filter(|target| needs_names(target))
        .count();
    let names = match &facts.names {
        None if count != 0 => return Ok(Decision::ReadNames { count }),
        Some(names) if names.len() != count => return Err(Error::InvalidAnswer),
        names => names.as_deref().unwrap_or_default(),
    };
    let mut reads = names.iter();
    let models = facts
        .requested
        .iter()
        .map(|model| {
            let Some(target) = facts.aliases.get(model) else {
                return model.clone();
            };
            if needs_names(target) && !reads.next().is_some_and(|names| names.contains(target)) {
                model.clone()
            } else {
                target.clone()
            }
        })
        .collect();
    Ok(Decision::Complete { models })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::missing(false, "alias")]
    #[case::live(true, "model_name_team_target")]
    fn deleted_target(#[case] live: bool, #[case] expected: &str) {
        let target = "model_name_team_target".to_owned();
        let facts = Facts {
            requested: vec!["alias".into()],
            aliases: BTreeMap::from([("alias".into(), target.clone())]),
            team_id: Some("team".into()),
            router_present: true,
            names: Some(vec![if live { vec![target] } else { vec![] }]),
        };
        assert_eq!(
            decide(&facts).unwrap(),
            Decision::Complete {
                models: vec![expected.into()]
            }
        );
    }
}
