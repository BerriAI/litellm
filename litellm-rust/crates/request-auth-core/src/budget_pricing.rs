use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize)]
pub struct DeploymentPrice {
    pub has_input_price: bool,
    pub has_output_price: bool,
    pub input_price: Option<f64>,
    pub output_price: Option<f64>,
    pub query_price: Option<f64>,
    pub ptu_count: Option<f64>,
    pub cost_per_ptu_per_hour: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelPriceFacts {
    pub cached: Option<bool>,
    pub lookup_failed: bool,
    pub group_found: bool,
    pub input_price: Option<f64>,
    pub output_price: Option<f64>,
    pub deployments: Option<Vec<DeploymentPrice>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PriceDecision {
    pub zero_cost: bool,
    pub cache_value: Option<bool>,
    pub read_deployments: bool,
}

pub fn price(facts: &ModelPriceFacts) -> PriceDecision {
    if let Some(cached) = facts.cached {
        return PriceDecision {
            zero_cost: cached,
            cache_value: None,
            read_deployments: false,
        };
    }
    if facts.lookup_failed {
        return PriceDecision {
            zero_cost: false,
            cache_value: None,
            read_deployments: false,
        };
    }
    let group_zero = facts.group_found
        && matches!((facts.input_price, facts.output_price), (Some(input), Some(output)) if !(input > 0.0 || output > 0.0));
    if group_zero && facts.deployments.is_none() {
        return PriceDecision {
            zero_cost: false,
            cache_value: None,
            read_deployments: true,
        };
    }
    let deployments = facts.deployments.as_deref().unwrap_or_default();
    let zero_cost = group_zero
        && deployments
            .iter()
            .any(|d| d.has_input_price || d.has_output_price)
        && !deployments.iter().any(|d| {
            d.input_price.is_some_and(|p| p > 0.0)
                || d.output_price.is_some_and(|p| p > 0.0)
                || d.query_price.is_some_and(|p| p > 0.0)
                || d.ptu_count.is_some() && d.cost_per_ptu_per_hour.is_some()
        });
    PriceDecision {
        zero_cost,
        cache_value: Some(zero_cost),
        read_deployments: false,
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct RouteBudgetFacts {
    pub side_effect_route: bool,
    pub discovery_route: bool,
    pub llm_route: bool,
    pub registered_pass_through: Option<bool>,
    pub route: String,
    pub method: Option<String>,
    pub body_present: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, strum::VariantArray)]
pub enum RouteBudgetDecision {
    ReadPassThrough,
    ReadPricing,
    Skip,
}

pub fn route(facts: &RouteBudgetFacts) -> RouteBudgetDecision {
    use RouteBudgetDecision::*;
    if !facts.side_effect_route {
        if facts.discovery_route {
            return Skip;
        }
        if !facts.llm_route {
            match facts.registered_pass_through {
                None => return ReadPassThrough,
                Some(false) => return Skip,
                Some(true) => (),
            }
        }
    }
    let mcp_discovery = facts.body_present
        && (matches!(
            facts.route.as_str(),
            "/mcp-rest/tools/list" | "/v1/mcp/tools"
        ) || !matches!(
            facts.route.as_str(),
            "/mcp/tools/call" | "/mcp-rest/tools/call"
        ) && (facts.route == "/mcp" || facts.route.starts_with("/mcp/"))
            && matches!(
                facts.method.as_deref(),
                Some("initialize" | "notifications/initialized" | "ping" | "tools/list")
            ));
    if mcp_discovery { Skip } else { ReadPricing }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::zero(Some(0.0), true)]
    #[case::negative(Some(-1.0), true)]
    #[case::nan(Some(f64::NAN), true)]
    #[case::missing(None, false)]
    #[case::positive(Some(1.0), false)]
    fn preserves_group_comparison(#[case] input: Option<f64>, #[case] zero_cost: bool) {
        let facts = ModelPriceFacts {
            cached: None,
            lookup_failed: false,
            group_found: true,
            input_price: input,
            output_price: Some(0.0),
            deployments: Some(vec![DeploymentPrice {
                has_input_price: true,
                has_output_price: false,
                input_price: None,
                output_price: None,
                query_price: None,
                ptu_count: None,
                cost_per_ptu_per_hour: None,
            }]),
        };
        assert_eq!(
            price(&facts),
            PriceDecision {
                zero_cost,
                cache_value: Some(zero_cost),
                read_deployments: false,
            }
        );
    }
}
