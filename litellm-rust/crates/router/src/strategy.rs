//! Routing policy: how the router picks one deployment from a model group.

use rand::seq::SliceRandom;

use crate::deployment::Deployment;

/// How the router chooses among the deployments sharing a `model_name`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// Uniform random pick among the matching deployments.
    #[default]
    SimpleShuffle,
}

impl RoutingStrategy {
    /// Choose one deployment from `candidates` (all sharing the requested
    /// `model_name`). Returns `None` when there are no candidates.
    pub fn select<'a>(&self, candidates: &[&'a Deployment]) -> Option<&'a Deployment> {
        match self {
            RoutingStrategy::SimpleShuffle => candidates.choose(&mut rand::thread_rng()).copied(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deployment::LiteLLMParams;

    fn deployment(model: &str) -> Deployment {
        Deployment {
            model_name: "gpt-realtime".to_string(),
            litellm_params: LiteLLMParams {
                model: model.to_string(),
                api_key: None,
                api_base: None,
            },
        }
    }

    #[test]
    fn simple_shuffle_picks_from_candidates() {
        let a = deployment("key-a");
        let b = deployment("key-b");
        let candidates = vec![&a, &b];
        for _ in 0..20 {
            let chosen = RoutingStrategy::SimpleShuffle
                .select(&candidates)
                .expect("non-empty");
            assert!(matches!(
                chosen.litellm_params.model.as_str(),
                "key-a" | "key-b"
            ));
        }
    }

    #[test]
    fn empty_candidates_select_none() {
        assert!(RoutingStrategy::SimpleShuffle.select(&[]).is_none());
    }
}
