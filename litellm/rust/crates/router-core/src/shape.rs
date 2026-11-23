use crate::host::Attempt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttemptShape {
    pub fallback_model_group: Option<String>,
    pub is_fallback: bool,
    pub drops_mock_timeout: bool,
}

impl AttemptShape {
    pub fn of(attempt: &Attempt) -> Self {
        let is_fallback = attempt.fallback_depth > 0;
        Self {
            fallback_model_group: (attempt.fallback_level > 0).then(|| attempt.name.clone()),
            is_fallback,
            drops_mock_timeout: is_fallback
                && !attempt.deployment.raw_params.contains_key("mock_timeout"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::{fixture, rstest};
    use serde_json::{Value, json};

    use super::*;
    use crate::{
        config::Op,
        registry::{Deployment, DeploymentSpec},
    };

    fn deployment(params: Value) -> Arc<Deployment> {
        let spec: DeploymentSpec = serde_json::from_value(json!({
            "model_name": "primary", "litellm_params": params, "model_info": {"id": "d1"},
        }))
        .unwrap();
        Arc::new(Deployment::from_spec(spec).unwrap())
    }

    #[fixture]
    fn attempt() -> Attempt {
        Attempt {
            op: Op::new("acompletion"),
            request: Arc::default(),
            deployment: deployment(json!({"model": "openai/gpt-4o"})),
            name: "backup".to_owned(),
            model_group: "backup".to_owned(),
            routed: None,
            target_order: None,
            candidate_ids: vec!["d1".to_owned()],
            retry: 0,
            fallback_depth: 0,
            fallback_level: 0,
            timeout: None,
            params: None,
            scope: 0,
            max_retries: 0,
            model_group_size: 1,
        }
    }

    #[rstest]
    #[case::requested_group(0, None)]
    #[case::first_fallback(1, Some("backup"))]
    #[case::nested_fallback(2, Some("backup"))]
    fn a_fallback_level_routes_its_target_in_the_metadata(
        attempt: Attempt,
        #[case] level: u32,
        #[case] expected: Option<&str>,
    ) {
        let shape = AttemptShape::of(&Attempt {
            fallback_level: level,
            ..attempt
        });
        assert_eq!(shape.fallback_model_group.as_deref(), expected);
    }

    #[rstest]
    #[case::requested_group_keeps_the_request_timeout(0, json!({"model": "openai/gpt-4o"}), false, false)]
    #[case::hop_drops_the_request_timeout(1, json!({"model": "openai/gpt-4o"}), true, true)]
    #[case::hop_keeps_a_deployment_timeout(
        2,
        json!({"model": "openai/gpt-4o", "mock_timeout": true}),
        true,
        false
    )]
    #[case::hop_keeps_a_null_deployment_timeout(
        1,
        json!({"model": "openai/gpt-4o", "mock_timeout": null}),
        true,
        false
    )]
    fn a_fallback_hop_copies_its_metadata_and_drops_the_request_mock_timeout(
        attempt: Attempt,
        #[case] depth: u32,
        #[case] params: Value,
        #[case] hop: bool,
        #[case] drops: bool,
    ) {
        let shape = AttemptShape::of(&Attempt {
            fallback_depth: depth,
            deployment: deployment(params),
            ..attempt
        });
        assert_eq!((shape.is_fallback, shape.drops_mock_timeout), (hop, drops));
    }
}
