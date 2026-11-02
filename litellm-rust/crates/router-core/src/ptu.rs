//! PTU shares (`litellm_core_utils/ptu_shares.py`): a deployment whose `model_info.ptu_shares`
//! splits it across teams serves only those teams, and the teams holding a share are kept to
//! their share.

use std::{collections::BTreeSet, sync::Arc};

use serde_json::{Map, Value};

use crate::registry::Deployment;

/// `parsed_ptu_shares`: the teams of a non-empty `ptu_shares` map whose every entry is a team id
/// and a positive whole number of PTUs (`2.0` reads as 2, a bool is no number), else None.
pub fn share_teams(model_info: &Map<String, Value>) -> Option<BTreeSet<String>> {
    let shares = model_info.get("ptu_shares")?.as_object()?;
    if shares.is_empty() {
        return None;
    }
    shares
        .iter()
        .map(|(team, share)| (!team.is_empty() && whole_ptus(share)).then(|| team.clone()))
        .collect()
}

/// `_whole_ptus`: a positive whole number.
fn whole_ptus(share: &Value) -> bool {
    match share {
        Value::Number(number) => number.as_i64().map_or_else(
            || {
                number
                    .as_u64()
                    .map(|whole| whole > 0)
                    .or_else(|| number.as_f64().map(|f| f.fract() == 0.0 && f > 0.0))
                    .unwrap_or(false)
            },
            |whole| whole > 0,
        ),
        _ => false,
    }
}

/// `filter_ptu_shared_deployments`: the deployments `team_id` holds a share on when it holds one,
/// else the ones split into no shares; a caller without a team holds none. Also whether any
/// deployment was withheld.
pub fn filter_shared(
    deployments: Vec<Arc<Deployment>>,
    team_id: Option<&str>,
) -> (Vec<Arc<Deployment>>, bool) {
    let team = team_id.unwrap_or("");
    let checks: Vec<(Arc<Deployment>, Option<BTreeSet<String>>)> = deployments
        .into_iter()
        .map(|deployment| {
            let shares = share_teams(&deployment.model_info.raw);
            (deployment, shares)
        })
        .collect();
    let holds_share = checks
        .iter()
        .any(|(_, shares)| shares.as_ref().is_some_and(|teams| teams.contains(team)));
    let total = checks.len();
    let kept: Vec<Arc<Deployment>> = checks
        .into_iter()
        .filter(|(_, shares)| match shares {
            Some(teams) => teams.contains(team),
            None => !holds_share,
        })
        .map(|(deployment, _)| deployment)
        .collect();
    let withheld = kept.len() < total;
    (kept, withheld)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case(json!({"TA": 50}), Some(vec!["TA"]))]
    #[case(json!({"TA": 2.0, "TB": 1}), Some(vec!["TA", "TB"]))]
    #[case(json!({"TA": 2.5}), None)]
    #[case(json!({"TA": true}), None)]
    #[case(json!({"TA": 0}), None)]
    #[case(json!({}), None)]
    #[case(json!({"": 1}), None)]
    #[case(json!("TA"), None)]
    fn reads_the_share_teams(#[case] shares: Value, #[case] expected: Option<Vec<&str>>) {
        let mut model_info = Map::new();
        model_info.insert("ptu_shares".to_owned(), shares);
        assert_eq!(
            share_teams(&model_info),
            expected.map(|teams| teams.into_iter().map(str::to_owned).collect())
        );
    }
}
