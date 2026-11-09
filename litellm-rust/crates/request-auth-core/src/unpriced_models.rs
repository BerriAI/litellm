use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize)]
pub struct Metric {
    pub name: String,
    pub scalar: Option<f64>,
    pub nested: Vec<Option<f64>>,
}

#[derive(Clone, Deserialize)]
pub struct Entry {
    pub tiered_present: bool,
    pub metrics: Vec<Metric>,
}

#[derive(Clone, Deserialize)]
pub struct Deployment {
    pub parameter_keys: Vec<String>,
    pub has_id: bool,
    pub info_loaded: bool,
    pub info: Option<Entry>,
    pub cost_keys: Option<Vec<String>>,
}

#[derive(Clone, Deserialize)]
pub struct Facts {
    pub model: Option<String>,
    pub router_present: bool,
    pub group_found: Option<bool>,
    pub deployments: Option<Vec<Deployment>>,
    pub cost_deployments: Option<Vec<Deployment>>,
}

#[derive(Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Complete { unpriced: bool },
    ReadGroup,
    ReadDeployments,
    ReadCostDeployments,
    ReadInfo { index: usize },
    ReadCost { index: usize },
}

pub fn entry_has_priced_metric(entry: &Entry) -> bool {
    entry.tiered_present
        || entry.metrics.iter().any(|metric| {
            metric.name.contains("cost_per")
                && (metric.scalar.is_some_and(|price| price > 0.0)
                    || metric
                        .nested
                        .iter()
                        .any(|price| price.is_some_and(|value| value > 0.0)))
        })
}

pub fn decide(facts: &Facts) -> Decision {
    if !facts.router_present || facts.model.as_deref().is_none_or(str::is_empty) {
        return Decision::Complete { unpriced: false };
    }
    match facts.group_found {
        None => return Decision::ReadGroup,
        Some(false) => return Decision::Complete { unpriced: false },
        Some(true) => {}
    }
    let Some(deployments) = &facts.deployments else {
        return Decision::ReadDeployments;
    };
    for (index, deployment) in deployments.iter().enumerate() {
        if deployment
            .parameter_keys
            .iter()
            .any(|key| key.contains("cost_per") || key == "tiered_pricing")
        {
            return Decision::Complete { unpriced: false };
        }
        if !deployment.has_id {
            continue;
        }
        if !deployment.info_loaded {
            return Decision::ReadInfo { index };
        }
        if deployment
            .info
            .as_ref()
            .is_some_and(entry_has_priced_metric)
        {
            return Decision::Complete { unpriced: false };
        }
    }
    let Some(cost_deployments) = &facts.cost_deployments else {
        return Decision::ReadCostDeployments;
    };
    for (index, deployment) in cost_deployments.iter().enumerate() {
        if !deployment.has_id {
            continue;
        }
        let Some(keys) = &deployment.cost_keys else {
            return Decision::ReadCost { index };
        };
        if keys.iter().any(|key| {
            matches!(
                key.as_str(),
                "input_cost_per_token" | "output_cost_per_token"
            )
        }) {
            return Decision::Complete { unpriced: false };
        }
    }
    Decision::Complete { unpriced: true }
}

#[cfg(test)]
mod tests {
    use super::{Entry, Metric, entry_has_priced_metric};
    use rstest::rstest;

    #[rstest]
    #[case::positive("input_cost_per_character", Some(0.1), vec![], false, true)]
    #[case::zero("input_cost_per_token", Some(0.0), vec![], false, false)]
    #[case::negative("cost_per_query", Some(-1.0), vec![], false, false)]
    #[case::nested("cost_per_image", None, vec![None, Some(0.2)], false, true)]
    #[case::non_price("context_window", Some(1000.0), vec![], false, false)]
    #[case::tiered("context_window", None, vec![], true, true)]
    fn metrics(
        #[case] name: &str,
        #[case] scalar: Option<f64>,
        #[case] nested: Vec<Option<f64>>,
        #[case] tiered_present: bool,
        #[case] expected: bool,
    ) {
        assert_eq!(
            entry_has_priced_metric(&Entry {
                tiered_present,
                metrics: vec![Metric {
                    name: name.to_owned(),
                    scalar,
                    nested
                }]
            }),
            expected
        );
    }
}
