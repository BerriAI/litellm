//! `Router(plugins=[...])`: the routing-plugin pipeline. The host runs the plugins (live Python
//! objects) in the async pick's pre-routing hook and reports the candidate models they left on
//! the request (`_routing_plugin_candidate_models`); router-core decides which deployments that
//! keeps (`_filter_by_routing_plugin_candidates`) and refuses the paths that never run the
//! pipeline (`_get_available_deployment_unresolved`).

use std::sync::Arc;

use crate::{config::RoutingStrategy, error::Error, host::RouterRequest, registry::Deployment};

/// Whether the router's own strategy has an async selector: the others pick on the sync path
/// even for async callers (`_async_get_available_deployment`).
fn picks_async(strategy: &RoutingStrategy) -> bool {
    matches!(
        strategy,
        RoutingStrategy::UsageBasedV2
            | RoutingStrategy::SimpleShuffle
            | RoutingStrategy::CostBased
            | RoutingStrategy::LatencyBased
            | RoutingStrategy::LeastBusy
    )
}

/// The sync pick never runs the pipeline, so a router with plugins refuses it rather than
/// letting a policy plugin be bypassed. `request` is a pick or a call, never a listing.
pub(crate) fn refuse_sync_pick(
    has_plugins: bool,
    strategy: &RoutingStrategy,
    request: &RouterRequest,
) -> Result<(), Error> {
    if has_plugins
        && request.pass_through.is_none()
        && (!request.pre_routes || !picks_async(strategy))
    {
        return Err(Error::RoutingPluginsNeedAsync);
    }
    Ok(())
}

/// `_filter_by_routing_plugin_candidates`: keep the deployments whose litellm model the plugins
/// left; nothing left is an error, never a fallback to the unfiltered pool.
pub(crate) fn narrow(
    healthy: Vec<Arc<Deployment>>,
    models: Option<&[String]>,
) -> Result<Vec<Arc<Deployment>>, Error> {
    let Some(models) = models else {
        return Ok(healthy);
    };
    let kept: Vec<Arc<Deployment>> = healthy
        .into_iter()
        .filter(|d| models.contains(&d.params.model))
        .collect();
    if kept.is_empty() {
        return Err(Error::RoutingPluginsLeftNothing {
            models: models.to_vec(),
        });
    }
    Ok(kept)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::registry::DeploymentSpec;

    fn deployment(id: &str, model: &str) -> Arc<Deployment> {
        let spec: DeploymentSpec = serde_json::from_value(json!({
            "model_name": "g", "litellm_params": {"model": model}, "model_info": {"id": id},
        }))
        .unwrap();
        Arc::new(Deployment::from_spec(spec).unwrap())
    }

    #[rstest]
    fn the_plugins_models_narrow_the_pool() {
        let pool = vec![deployment("a", "openai/a"), deployment("b", "openai/b")];
        let kept = narrow(pool, Some(&["openai/b".to_owned()])).unwrap();
        assert_eq!(
            kept.iter().map(|d| d.id.as_str()).collect::<Vec<_>>(),
            ["b"]
        );
    }

    #[rstest]
    fn narrowing_to_nothing_is_an_error() {
        let pool = vec![deployment("a", "openai/a")];
        assert!(matches!(
            narrow(pool, Some(&[])),
            Err(Error::RoutingPluginsLeftNothing { models }) if models.is_empty()
        ));
    }

    #[rstest]
    #[case::async_pick(true, RoutingStrategy::SimpleShuffle, false)]
    #[case::sync_pick(false, RoutingStrategy::SimpleShuffle, true)]
    #[case::v1_strategy(true, RoutingStrategy::UsageBased, true)]
    fn only_the_async_pick_runs_plugins(
        #[case] pre_routes: bool,
        #[case] strategy: RoutingStrategy,
        #[case] refused: bool,
    ) {
        let request = RouterRequest {
            pre_routes,
            ..RouterRequest::default()
        };
        assert_eq!(
            refuse_sync_pick(true, &strategy, &request).is_err(),
            refused
        );
    }
}
