use std::{collections::HashMap, sync::Arc};

use arc_swap::ArcSwap;
use serde::Deserialize;

use crate::{pyrepr::PyNumber, settings::Settings};

/// A deployment as Python normalized it (`model_info.id` already generated), reduced to what
/// routing reads.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct RoutedDeployment {
    pub id: String,
    pub model_name: String,
    /// `litellm_params.model`.
    pub model: String,
    /// `litellm_params.weight`, the only shuffle weight the first increment serves.
    #[serde(default)]
    pub weight: Option<f64>,
    /// `model_info.cooldown_time`, else `litellm_params.cooldown_time`.
    #[serde(default)]
    pub cooldown_time: Option<PyNumber>,
    /// `litellm_params.num_retries`, stamped onto the exceptions its attempts raise.
    #[serde(default)]
    pub num_retries: Option<u32>,
    /// `model_info.supports_web_search`.
    #[serde(default)]
    pub supports_web_search: Option<bool>,
}

/// Everything one routing decision reads, swapped as a whole so a call never sees a
/// half-applied update.
#[derive(Debug, Default)]
pub struct Snapshot {
    deployments: Vec<RoutedDeployment>,
    by_group: HashMap<String, Vec<usize>>,
    by_id: HashMap<String, usize>,
    pub settings: Settings,
    /// `litellm.provider_list`, read by the provider-stripped fallback key match.
    pub providers: Vec<String>,
}

impl Snapshot {
    pub fn new(
        deployments: Vec<RoutedDeployment>,
        settings: Settings,
        providers: Vec<String>,
    ) -> Self {
        let by_id = deployments
            .iter()
            .enumerate()
            .map(|(index, deployment)| (deployment.id.clone(), index))
            .collect();
        let by_group = deployments.iter().enumerate().fold(
            HashMap::<String, Vec<usize>>::new(),
            |mut groups, (index, deployment)| {
                groups
                    .entry(deployment.model_name.clone())
                    .or_default()
                    .push(index);
                groups
            },
        );
        Self {
            deployments,
            by_group,
            by_id,
            settings,
            providers,
        }
    }

    /// The group's deployments in config order.
    pub fn group(&self, model_group: &str) -> Vec<&RoutedDeployment> {
        self.by_group
            .get(model_group)
            .map(|indices| {
                indices
                    .iter()
                    .map(|index| &self.deployments[*index])
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn has_group(&self, model_group: &str) -> bool {
        self.by_group.contains_key(model_group)
    }

    pub fn by_id(&self, id: &str) -> Option<&RoutedDeployment> {
        self.by_id.get(id).map(|index| &self.deployments[*index])
    }

    /// Deployments whose `litellm_params.model` is `model`, for a request that names a
    /// provider model instead of a group.
    pub fn by_litellm_model(&self, model: &str) -> Vec<&RoutedDeployment> {
        self.deployments
            .iter()
            .filter(|deployment| deployment.model == model)
            .collect()
    }

    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.deployments
            .iter()
            .map(|deployment| deployment.id.as_str())
    }

    /// The size of the group `id` belongs to (`len(get_model_group(id))`), or `None` for an
    /// unknown id.
    pub fn group_size_of(&self, id: &str) -> Option<usize> {
        self.by_id(id)
            .map(|deployment| self.group(&deployment.model_name).len())
    }
}

/// The router's registry: readers load the current snapshot without locking, writers swap
/// in a new one.
#[derive(Debug, Default)]
pub struct Registry(ArcSwap<Snapshot>);

impl Registry {
    pub fn new(snapshot: Snapshot) -> Self {
        Self(ArcSwap::from_pointee(snapshot))
    }

    pub fn load(&self) -> Arc<Snapshot> {
        self.0.load_full()
    }

    pub fn replace(&self, snapshot: Snapshot) {
        self.0.store(Arc::new(snapshot));
    }
}
