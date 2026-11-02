//! The configured deployments, indexed by id, model group, alias, wildcard pattern, routing
//! group and team. A [`Registry`] is immutable; [`Registry::apply`] builds the next version.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::Write as _,
    sync::Arc,
};

use litellm_router_types::{LitellmParams, Spelled};
use serde::Deserialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::{
    error::ConfigError,
    host_index::{HostIndexes, IndexWrite},
};

/// A registry's version; every change produces a higher one.
pub type RegistryVersion = u64;

/// A deployment as a config or an `apply` spells it.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct DeploymentSpec {
    pub model_name: String,
    pub litellm_params: Map<String, Value>,
    #[serde(default)]
    pub model_info: Map<String, Value>,
    /// The entry's other top-level keys (a config's `tpm`, ...), kept so the host's view of the
    /// deployment list reads back the entries it stored.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
    /// The provider the host resolved when it built the deployment ([`crate::build`]); never
    /// read from a config or an `apply`.
    #[serde(skip)]
    pub provider: Option<String>,
}

/// `routing_groups`: a named subset of model groups sharing a routing strategy.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct RoutingGroupSpec {
    pub group_name: String,
    pub models: Vec<String>,
    pub routing_strategy: crate::config::RoutingStrategy,
    /// A priority group's member priorities, served as each member's `order`.
    #[serde(default)]
    pub model_priorities: Option<BTreeMap<String, i64>>,
}

/// The `model_info` fields the router reads; every other key stays in [`Self::raw`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelInfo {
    pub team_id: Option<String>,
    pub team_public_model_name: Option<String>,
    pub access_groups: Vec<String>,
    pub blocked: bool,
    pub cooldown_time: Option<f64>,
    pub allowed_fails: Option<u32>,
    /// `model_info.allowed_fails_policy` when it is a dict, read as Python reads it: field name
    /// to allowed fails, values as given.
    pub allowed_fails_policy: Option<Map<String, Value>>,
    pub raw: Map<String, Value>,
}

impl ModelInfo {
    fn from_raw(raw: Map<String, Value>) -> Self {
        let text = |key: &str| raw.get(key).and_then(Value::as_str).map(str::to_owned);
        Self {
            team_id: text("team_id"),
            team_public_model_name: text("team_public_model_name"),
            access_groups: raw
                .get("access_groups")
                .and_then(Value::as_array)
                .map(|groups| {
                    groups
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default(),
            blocked: raw.get("blocked").and_then(Value::as_bool) == Some(true),
            cooldown_time: raw.get("cooldown_time").and_then(Value::as_f64),
            allowed_fails: raw
                .get("allowed_fails")
                .and_then(Value::as_u64)
                .and_then(|fails| u32::try_from(fails).ok()),
            allowed_fails_policy: raw
                .get("allowed_fails_policy")
                .and_then(Value::as_object)
                .cloned(),
            raw,
        }
    }
}

/// One configured deployment.
#[derive(Clone, Debug, PartialEq)]
pub struct Deployment {
    pub id: String,
    pub model_name: String,
    pub params: LitellmParams,
    /// `litellm_params` as written, in order: what the id hashes and the host receives.
    pub raw_params: Map<String, Value>,
    pub model_info: ModelInfo,
    /// The entry's other top-level keys, as written.
    pub extra: Map<String, Value>,
    /// The provider the host resolved for it, when it was built through the host.
    pub resolved_provider: Option<String>,
}

impl Deployment {
    pub fn from_spec(spec: DeploymentSpec) -> Result<Self, ConfigError> {
        let params: LitellmParams = serde_json::from_value(Value::Object(
            spec.litellm_params.clone(),
        ))
        .map_err(|source| ConfigError::InvalidDeployment {
            model_name: spec.model_name.clone(),
            source,
        })?;
        let id = match spec.model_info.get("id") {
            Some(Value::String(id)) => id.clone(),
            Some(Value::Number(id)) => id.to_string(),
            _ => deployment_id(&spec.model_name, &spec.litellm_params),
        };
        Ok(Self {
            id,
            model_name: spec.model_name,
            params,
            raw_params: spec.litellm_params,
            model_info: ModelInfo::from_raw(spec.model_info),
            extra: spec.extra,
            resolved_provider: spec.provider,
        })
    }

    /// The deployment as a `model_list` entry: its other top-level keys, then `model_name`,
    /// `litellm_params` and `model_info` (with its id).
    pub fn record(&self) -> Value {
        let mut model_info = self.model_info.raw.clone();
        model_info.insert("id".to_owned(), Value::String(self.id.clone()));
        let mut record = self.extra.clone();
        record.insert(
            "model_name".to_owned(),
            Value::String(self.model_name.clone()),
        );
        record.insert(
            "litellm_params".to_owned(),
            Value::Object(self.raw_params.clone()),
        );
        record.insert("model_info".to_owned(), Value::Object(model_info));
        Value::Object(record)
    }

    /// The stand-in an unrouted attempt carries: the requested name as its group and model, no id.
    pub fn unrouted(name: &str) -> Result<Self, ConfigError> {
        let mut params = Map::new();
        params.insert("model".to_owned(), Value::String(name.to_owned()));
        let mut info = Map::new();
        info.insert("id".to_owned(), Value::String(String::new()));
        Self::from_spec(DeploymentSpec {
            model_name: name.to_owned(),
            litellm_params: params,
            model_info: info,
            extra: Map::new(),
            provider: None,
        })
    }

    /// The provider: the one the host resolved when it built the deployment, else
    /// `custom_llm_provider`, else the `model` prefix before `/`.
    pub fn provider(&self) -> &str {
        self.resolved_provider
            .as_deref()
            .or(self.params.custom_llm_provider.as_deref())
            .or_else(|| {
                self.params
                    .model
                    .split_once('/')
                    .map(|(provider, _)| provider)
            })
            .unwrap_or("openai")
    }

    /// A numeric `litellm_params` value, as `simple_shuffle._metric_weight` reads it.
    pub fn metric(&self, key: &str) -> Option<f64> {
        let spelled = |value: &Option<Spelled<f64>>| match value {
            Some(Spelled::Value(value)) => Some(*value),
            Some(Spelled::Text(_)) | None => None,
        };
        match key {
            "rpm" => spelled(&self.params.rpm),
            "tpm" => spelled(&self.params.tpm),
            _ => self.raw_params.get(key).and_then(Value::as_f64),
        }
    }

    /// `simple_shuffle._metric_weight`: 0 when unset, a number (a bool counts as 0 or 1), or
    /// `None` when the value is not numeric. A `rpm`/`tpm` spelled as text reads as unset.
    pub fn metric_weight(&self, key: &str) -> Option<f64> {
        if key != "weight" {
            return Some(self.metric(key).unwrap_or(0.0));
        }
        match self.raw_params.get(key) {
            None | Some(Value::Null) => Some(0.0),
            Some(Value::Bool(flag)) => Some(f64::from(u8::from(*flag))),
            Some(Value::Number(number)) => number.as_f64(),
            Some(_) => None,
        }
    }

    /// `litellm_params.order` (`get_deployment_order`).
    pub fn order(&self) -> Option<i64> {
        self.raw_params.get("order").and_then(Value::as_i64)
    }

    /// Whether `team_id` may use this deployment (`deployment_usable_by_team`).
    pub fn usable_by_team(&self, team_id: Option<&str>) -> bool {
        self.model_info
            .team_id
            .as_deref()
            .is_none_or(|owner| Some(owner) == team_id)
    }

    /// Whether the router counts per-minute usage for this deployment
    /// (`_deployment_tracks_usage`, without the model-catalog limits).
    pub fn tracks_usage(&self) -> bool {
        self.params.tpm.is_some()
            || self.params.rpm.is_some()
            || self.params.itpm.is_some()
            || self.params.otpm.is_some()
    }
}

/// The connection params of one deployment (`credentials(id)`).
#[derive(Clone, Debug, PartialEq)]
pub struct Credentials {
    pub provider: String,
    pub api_base: Option<String>,
    pub api_version: Option<String>,
    pub credential_name: Option<String>,
    pub params: LitellmParams,
}

/// What a name resolved through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolveKind {
    SpecificDeployment,
    Id,
    /// A name nothing else serves, routed to the router's `default_deployment` (a copy whose
    /// `litellm_params.model` is the name). Served like a deployment named by its id.
    Default,
    Alias,
    RoutingGroup,
    Group,
    Wildcard,
}

impl ResolveKind {
    /// The name resolved to one deployment that is served as is: a deployment id, or the
    /// default deployment (`_common_checks_available_deployment` returns a dict for both).
    pub fn is_single(&self) -> bool {
        matches!(self, Self::Id | Self::Default)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Resolution {
    pub kind: ResolveKind,
    /// The model group the deployments are routed as.
    pub group: String,
    pub deployments: Vec<Arc<Deployment>>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ResolveScope {
    /// The caller named a deployment's `litellm_params.model` (`specific_deployment=True`).
    pub specific_deployment: bool,
}

#[derive(Clone, Debug, Default)]
pub struct DeploymentFilter<'a> {
    pub model_name: Option<&'a str>,
    pub team_id: Option<&'a str>,
    pub provider: Option<&'a str>,
}

#[derive(Clone, Debug, Default)]
pub struct NameFilter {
    pub include_aliases: bool,
}

/// The registry indexes as the host wrote them over the ones the deployments build (the
/// deprecated `model_names`, `model_name_to_deployment_indices` and
/// `team_model_to_deployment_indices` writes), which resolution reads instead from then on.
/// Indices point into the deployment list; one past its end is skipped.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IndexOverrides {
    /// `_core_model_names`: the names `name in model_names` finds.
    pub model_names: Option<BTreeSet<String>>,
    /// `_core_model_name_to_deployment_indices`: a model name's deployments.
    pub model_name: Option<BTreeMap<String, Vec<usize>>>,
    /// `_core_team_model_to_deployment_indices`, as `(team_id, public name, indices)` entries.
    pub team_model: Option<Vec<(String, String, Vec<usize>)>>,
}

impl IndexOverrides {
    fn is_empty(&self) -> bool {
        self.model_names.is_none() && self.model_name.is_none() && self.team_model.is_none()
    }
}

/// One atomic change: optionally a whole new deployment list, then upserts, deletes and,
/// optionally, a new dynamic config.
#[derive(Clone, Debug, Default)]
pub struct ChangeSet {
    /// Every deployment, in order, replacing the current list (the host registry's `model_list`
    /// after a write; it may hold one id more than once, as the Python router's list can).
    pub replace: Option<Vec<DeploymentSpec>>,
    /// Deployments added at the end, beside any that share their id (a deployment the host
    /// retries after its provider resolved).
    pub appends: Vec<DeploymentSpec>,
    pub upserts: Vec<DeploymentSpec>,
    pub deletes: Vec<String>,
    pub settings: Option<crate::config::RouterDynamicConfig>,
    /// The indexes the host wrote ([`IndexOverrides`]), replacing the ones held; an empty one
    /// goes back to the indexes the deployments build.
    pub index: Option<IndexOverrides>,
    /// The deployment at the position the id map holds for this id, removed with every index
    /// shifted (`_delete_deployment`, and an upsert's removal of the deployment it edits).
    pub remove: Option<String>,
    /// A host write to the indexes it hands out ([`HostIndexes`]).
    pub host_index: Option<IndexWrite>,
}

#[derive(Clone, Debug, Default)]
pub struct Registry {
    version: RegistryVersion,
    /// The version the deployments before the newest appends last changed at: a host that read
    /// the list at or after it reads only what was appended since ([`Self::deployments_from`]).
    rewritten: RegistryVersion,
    deployments: Vec<Arc<Deployment>>,
    by_id: HashMap<String, usize>,
    by_group: BTreeMap<String, Vec<usize>>,
    by_team: HashMap<String, Vec<usize>>,
    wildcards: Vec<(String, Vec<usize>)>,
    aliases: BTreeMap<String, String>,
    routing_groups: Vec<RoutingGroupSpec>,
    overrides: Option<Arc<IndexOverrides>>,
    /// The indexes the host hands out, kept by the Python router's history rules.
    host: Arc<HostIndexes>,
    /// `_model_to_group` as Python last built it, when a raw `routing_groups` write replaced the
    /// groups since: member calls keep routing by it (see [`Self::member_routing_group`]).
    member_groups: Option<Arc<BTreeMap<String, String>>>,
}

impl Registry {
    pub fn build(
        specs: Vec<DeploymentSpec>,
        aliases: BTreeMap<String, String>,
        routing_groups: Vec<RoutingGroupSpec>,
    ) -> Result<Self, ConfigError> {
        // One id may name several deployments (a per-organization fan-out, or two entries
        // declaring one id): each is kept, and the id finds the last, as the Python router keeps them.
        let deployments = specs
            .into_iter()
            .map(|spec| Deployment::from_spec(spec).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::index(0, deployments, aliases, routing_groups))
    }

    fn index(
        version: RegistryVersion,
        deployments: Vec<Arc<Deployment>>,
        aliases: BTreeMap<String, String>,
        routing_groups: Vec<RoutingGroupSpec>,
    ) -> Self {
        let mut by_id = HashMap::new();
        let mut by_group: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let mut by_team: HashMap<String, Vec<usize>> = HashMap::new();
        for (index, deployment) in deployments.iter().enumerate() {
            by_id.insert(deployment.id.clone(), index);
            by_group
                .entry(deployment.model_name.clone())
                .or_default()
                .push(index);
            if let Some(team) = &deployment.model_info.team_id {
                by_team.entry(team.clone()).or_default().push(index);
            }
        }
        // In the order each pattern first appears, then most specific first, as
        // `PatternUtils.sorted_patterns` keeps them: a stable sort, longest regex first,
        // then the most regex syntax.
        let mut wildcards: Vec<(String, Vec<usize>)> = Vec::new();
        for deployment in &deployments {
            let name = &deployment.model_name;
            if name.contains('*') && !wildcards.iter().any(|(pattern, _)| pattern == name) {
                wildcards.push((
                    name.clone(),
                    by_group.get(name).cloned().unwrap_or_default(),
                ));
            }
        }
        wildcards.sort_by_key(|(pattern, _)| std::cmp::Reverse(pattern_specificity(pattern)));
        Self {
            version,
            rewritten: version,
            deployments,
            by_id,
            by_group,
            by_team,
            wildcards,
            aliases,
            routing_groups,
            overrides: None,
            host: Arc::default(),
            member_groups: None,
        }
    }

    /// This registry with the stale routing-group member map (`None` derives it from the groups).
    #[must_use]
    pub fn with_member_groups(self, member_groups: Option<BTreeMap<String, String>>) -> Self {
        Self {
            member_groups: member_groups.map(Arc::new),
            ..self
        }
    }

    pub fn version(&self) -> RegistryVersion {
        self.version
    }

    /// The version the deployments before the newest appends last changed at (see `rewritten`).
    pub fn rewritten(&self) -> RegistryVersion {
        self.rewritten
    }

    /// The deployments from position `start` on, in list order.
    pub fn deployments_from(&self, start: usize) -> &[Arc<Deployment>] {
        self.deployments.get(start..).unwrap_or_default()
    }

    /// The next registry: upserts, then deletes (`apply_changes`). An upserted deployment whose
    /// id exists is removed and appended at the end, as the Python router does on an edit.
    pub fn apply(
        &self,
        changes: &ChangeSet,
        aliases: Option<BTreeMap<String, String>>,
    ) -> Result<Self, ConfigError> {
        let mut deployments = match &changes.replace {
            Some(specs) => specs
                .iter()
                .map(|spec| self.carried(spec.clone()).map(Arc::new))
                .collect::<Result<Vec<_>, _>>()?,
            None => self.deployments.clone(),
        };
        for spec in &changes.appends {
            deployments.push(Arc::new(self.carried(spec.clone())?));
        }
        for spec in &changes.upserts {
            let deployment = self.carried(spec.clone())?;
            deployments.retain(|existing| existing.id != deployment.id);
            deployments.push(Arc::new(deployment));
        }
        deployments.retain(|deployment| !changes.deletes.contains(&deployment.id));
        let mut host = HostIndexes::clone(&self.host);
        if let Some(id) = &changes.remove
            && let Some(position) = host.position_of(id)
            && position < deployments.len()
        {
            deployments.remove(position);
            host.removed(id, position)
                .map_err(|error| ConfigError::Removal(error.to_string()))?;
        }
        if let Some(write) = &changes.host_index {
            host.write(write.clone());
        }
        let overrides = match &changes.index {
            Some(index) => (!index.is_empty()).then(|| Arc::new(index.clone())),
            None => self.overrides.clone(),
        };
        // Appends alone leave every deployment before them as it was.
        let appended_only = changes.replace.is_none()
            && changes.upserts.is_empty()
            && changes.deletes.is_empty()
            && changes.remove.is_none();
        let next = Self::index(
            self.version + 1,
            deployments,
            aliases.unwrap_or_else(|| self.aliases.clone()),
            self.routing_groups.clone(),
        );
        Ok(Self {
            overrides,
            host: Arc::new(host),
            member_groups: self.member_groups.clone(),
            rewritten: if appended_only {
                self.rewritten
            } else {
                next.rewritten
            },
            ..next
        })
    }

    /// The next registry for a build starting over (`set_model_list`): no deployments, and the
    /// indexes [`HostIndexes::reset`] clears cleared.
    #[must_use]
    pub fn emptied(&self) -> Self {
        let mut host = HostIndexes::clone(&self.host);
        host.reset();
        Self {
            overrides: self.overrides.clone(),
            host: Arc::new(host),
            member_groups: self.member_groups.clone(),
            ..Self::index(
                self.version + 1,
                Vec::new(),
                self.aliases.clone(),
                self.routing_groups.clone(),
            )
        }
    }

    /// Add one deployment a build accepted at the end and index it in place (`add` is
    /// `add_deployment`, which also names its group). A build installs into a registry of its
    /// own this way, so a list of n deployments builds in O(n) rather than reindexing per
    /// deployment.
    pub fn push_built(&mut self, spec: DeploymentSpec, add: bool) -> Result<(), ConfigError> {
        let deployment = Arc::new(self.carried(spec)?);
        let index = self.deployments.len();
        let record = deployment.record();
        Arc::make_mut(&mut self.host).added(
            record.as_object().expect("a record is an object"),
            index,
            add,
        );
        self.by_id.insert(deployment.id.clone(), index);
        let name = &deployment.model_name;
        self.by_group.entry(name.clone()).or_default().push(index);
        if let Some(team) = &deployment.model_info.team_id {
            self.by_team.entry(team.clone()).or_default().push(index);
        }
        if name.contains('*') {
            match self
                .wildcards
                .iter_mut()
                .find(|(pattern, _)| pattern == name)
            {
                Some((_, indices)) => indices.push(index),
                None => {
                    // A new pattern joins after every one it ties with, as `index` orders them:
                    // first appearance, then a stable sort by specificity.
                    self.wildcards.push((name.clone(), vec![index]));
                    self.wildcards.sort_by_key(|(pattern, _)| {
                        std::cmp::Reverse(pattern_specificity(pattern))
                    });
                }
            }
        }
        self.deployments.push(deployment);
        self.version += 1;
        Ok(())
    }

    /// The last deployment added.
    pub fn last(&self) -> Option<&Arc<Deployment>> {
        self.deployments.last()
    }

    /// Take `set_model_list`'s `model_names`: the names it was given.
    pub fn set_model_names(&mut self, names: Vec<String>) {
        Arc::make_mut(&mut self.host).set_model_names(names);
        self.version += 1;
    }

    /// The indexes the host hands out.
    pub fn host_indexes(&self) -> &HostIndexes {
        &self.host
    }

    /// The deployment `spec` describes, keeping the provider the host resolved for the
    /// deployment it replaces (same id, same params) when the spec carries none: a host that
    /// hands back its whole list after a write does not resolve every provider again.
    fn carried(&self, mut spec: DeploymentSpec) -> Result<Deployment, ConfigError> {
        let deployment = Deployment::from_spec(spec.clone())?;
        if spec.provider.is_none()
            && let Some(previous) = self
                .by_id
                .get(&deployment.id)
                .and_then(|index| self.deployments.get(*index))
            && previous.raw_params == deployment.raw_params
            && previous.model_name == deployment.model_name
        {
            spec.provider.clone_from(&previous.resolved_provider);
            return Deployment::from_spec(spec);
        }
        Ok(deployment)
    }

    /// This registry with the providers `from` resolved for the same deployments (same id,
    /// same params) and `from`'s host indexes: a router rebuilt for new static settings over the
    /// same list keeps what its host resolved and the indexes their history built.
    #[must_use]
    pub fn with_providers_of(&self, from: &Self) -> Self {
        let deployments = self
            .deployments
            .iter()
            .map(|deployment| {
                let previous = from
                    .by_id
                    .get(&deployment.id)
                    .and_then(|index| from.deployments.get(*index))
                    .filter(|previous| {
                        previous.raw_params == deployment.raw_params
                            && previous.model_name == deployment.model_name
                    });
                match previous {
                    Some(previous) if deployment.resolved_provider.is_none() => {
                        Arc::new(Deployment {
                            resolved_provider: previous.resolved_provider.clone(),
                            ..Deployment::clone(deployment)
                        })
                    }
                    _ => deployment.clone(),
                }
            })
            .collect();
        Self {
            deployments,
            host: from.host.clone(),
            ..self.clone()
        }
    }

    fn at(&self, indices: &[usize]) -> Vec<Arc<Deployment>> {
        indices
            .iter()
            .filter_map(|index| self.deployments.get(*index).cloned())
            .collect()
    }

    /// `_core_deployments_named(name)`: the deployments the model-name index holds for `name`
    /// (whatever their own name), or `None` when it does not index the name.
    pub(crate) fn named(&self, name: &str) -> Option<Vec<Arc<Deployment>>> {
        match self.overrides.as_ref().and_then(|o| o.model_name.as_ref()) {
            Some(index) => index.get(name).map(|indices| self.at(indices)),
            None => self.by_group.get(name).map(|indices| self.at(indices)),
        }
    }

    /// Whether the model-name index holds `name` (`name in model_name_to_deployment_indices`).
    fn indexes_name(&self, name: &str) -> bool {
        match self.overrides.as_ref().and_then(|o| o.model_name.as_ref()) {
            Some(index) => index.contains_key(name),
            None => self.by_group.contains_key(name),
        }
    }

    /// `_team_deployments_named(team_id, name)`: the deployments the team index holds for the
    /// pair, or `None` when it does not index it.
    pub(crate) fn team_named(&self, team_id: &str, name: &str) -> Option<Vec<Arc<Deployment>>> {
        match self.overrides.as_ref().and_then(|o| o.team_model.as_ref()) {
            Some(entries) => entries
                .iter()
                .find(|(team, public, _)| team == team_id && public == name)
                .map(|(_, _, indices)| self.at(indices)),
            None => {
                let own: Vec<Arc<Deployment>> = self
                    .deployments
                    .iter()
                    .filter(|d| {
                        d.model_info.team_id.as_deref() == Some(team_id)
                            && d.model_info.team_public_model_name.as_deref() == Some(name)
                    })
                    .cloned()
                    .collect();
                (!own.is_empty()).then_some(own)
            }
        }
    }

    /// Every deployment the team index holds under public name `name`, any team's
    /// (`_team_deployments_across_teams`).
    pub(crate) fn team_named_across_teams(&self, name: &str) -> Vec<Arc<Deployment>> {
        match self.overrides.as_ref().and_then(|o| o.team_model.as_ref()) {
            Some(entries) => entries
                .iter()
                .filter(|(_, public, _)| public == name)
                .flat_map(|(_, _, indices)| self.at(indices))
                .collect(),
            None => self
                .deployments
                .iter()
                .filter(|d| {
                    d.model_info.team_id.is_some()
                        && d.model_info.team_public_model_name.as_deref() == Some(name)
                })
                .cloned()
                .collect(),
        }
    }

    /// Whether a written index entry names a position past the model list.
    fn overflows(&self, indices: &[usize]) -> bool {
        indices.iter().any(|index| *index >= self.deployments.len())
    }

    /// The written model-name index entry for `name` names a position past the model list.
    pub(crate) fn name_index_overflows(&self, name: &str) -> bool {
        self.overrides
            .as_ref()
            .and_then(|o| o.model_name.as_ref())
            .and_then(|index| index.get(name))
            .is_some_and(|indices| self.overflows(indices))
    }

    /// The written team index entry for `(team_id, name)` names a position past the model list.
    pub(crate) fn team_index_overflows(&self, team_id: &str, name: &str) -> bool {
        self.overrides
            .as_ref()
            .and_then(|o| o.team_model.as_ref())
            .and_then(|entries| {
                entries
                    .iter()
                    .find(|(team, public, _)| team == team_id && public == name)
            })
            .is_some_and(|(_, _, indices)| self.overflows(indices))
    }

    /// A written team index entry under public name `name` names a position past the list.
    pub(crate) fn across_teams_index_overflows(&self, name: &str) -> bool {
        self.overrides
            .as_ref()
            .and_then(|o| o.team_model.as_ref())
            .is_some_and(|entries| {
                entries
                    .iter()
                    .any(|(_, public, indices)| public == name && self.overflows(indices))
            })
    }

    /// The group a pick of `name` reads its strategy from, when the stale member map names a
    /// group that is no longer configured (`_get_routing_context` raises `KeyError`): `name`
    /// after its alias, unless it is a callable routing group.
    pub fn missing_member_group(&self, name: &str) -> Option<&str> {
        let members = self.member_groups.as_ref()?;
        let resolved = self.alias_of(name).unwrap_or(name);
        if self.routing_group_named(resolved).is_some() {
            return None;
        }
        let group = members.get(resolved)?;
        (!self.routing_groups.iter().any(|g| g.group_name == *group)).then_some(group.as_str())
    }

    /// The routing group a member call of `model_group` routes by: the stale member map's
    /// group when there is one (`_model_to_group`), else the non-priority group listing it.
    pub(crate) fn member_routing_group(&self, model_group: &str) -> Option<&RoutingGroupSpec> {
        match &self.member_groups {
            Some(members) => members
                .get(model_group)
                .and_then(|name| self.routing_groups.iter().find(|g| g.group_name == *name)),
            None => self
                .routing_groups
                .iter()
                .filter(|group| !crate::routing_groups::is_priority(group))
                .find(|group| group.models.iter().any(|model| model == model_group)),
        }
    }

    pub fn deployment(&self, id: &str) -> Option<Arc<Deployment>> {
        self.by_id
            .get(id)
            .map(|index| self.deployments[*index].clone())
    }

    pub fn deployments(&self, filter: &DeploymentFilter<'_>) -> Vec<Arc<Deployment>> {
        let candidates = match (filter.model_name, filter.team_id) {
            (Some(name), _) => self
                .by_group
                .get(name)
                .map(|i| self.at(i))
                .unwrap_or_default(),
            (None, Some(team)) => self
                .by_team
                .get(team)
                .map(|i| self.at(i))
                .unwrap_or_default(),
            (None, None) => self.deployments.clone(),
        };
        candidates
            .into_iter()
            .filter(|d| {
                filter
                    .team_id
                    .is_none_or(|team| d.model_info.team_id.as_deref() == Some(team))
            })
            .filter(|d| {
                filter
                    .provider
                    .is_none_or(|provider| d.provider() == provider)
            })
            .collect()
    }

    /// The model group names, in name order, with aliases when asked.
    pub fn names(&self, filter: &NameFilter) -> Vec<String> {
        let mut names: Vec<String> = self.by_group.keys().cloned().collect();
        if filter.include_aliases {
            names.extend(self.aliases.keys().cloned());
        }
        names
    }

    pub fn group(&self, name: &str) -> Option<Vec<Arc<Deployment>>> {
        if self
            .overrides
            .as_ref()
            .is_some_and(|o| o.model_name.is_some())
        {
            return self.named(name).map(|deployments| {
                deployments
                    .into_iter()
                    .filter(|d| d.model_name == name)
                    .collect()
            });
        }
        self.by_group.get(name).map(|indices| self.at(indices))
    }

    /// How many rows `_core_get_model_list(model_name=name)` lists: the group's deployments,
    /// an alias's target group, a routing group's members, and only when none of those, the
    /// deployments of the wildcard the name matches.
    pub fn model_list_len(&self, name: &str) -> usize {
        let count = |group: &str| self.group(group).map_or(0, |deployments| deployments.len());
        let listed = count(name)
            + self.aliases.get(name).map_or(0, |target| count(target))
            + self
                .routing_groups
                .iter()
                .find(|group| group.group_name == name)
                .map_or(0, |group| {
                    group.models.iter().map(|model| count(model)).sum()
                });
        if listed > 0 {
            return listed;
        }
        self.wildcards
            .iter()
            .find(|(pattern, _)| wildcard_matches(pattern, name))
            .map_or(0, |(_, indices)| indices.len())
    }

    /// The routing group a model group belongs to, if any.
    pub fn routing_group_of(&self, model_name: &str) -> Option<&RoutingGroupSpec> {
        self.routing_groups
            .iter()
            .find(|group| group.models.iter().any(|model| model == model_name))
    }

    /// The strategy a routing group named `group_name` declares.
    pub fn resolve_routing_group_strategy(
        &self,
        group_name: &str,
    ) -> Option<crate::config::RoutingStrategy> {
        self.routing_groups
            .iter()
            .find(|group| group.group_name == group_name)
            .map(|group| group.routing_strategy.clone())
    }

    pub(crate) fn routing_groups(&self) -> &[RoutingGroupSpec] {
        &self.routing_groups
    }

    /// Every deployment, in configured order.
    pub(crate) fn all(&self) -> &[Arc<Deployment>] {
        &self.deployments
    }

    /// Whether `name` is a deployment `model_name` (`name in _core_model_names`).
    pub(crate) fn has_name(&self, name: &str) -> bool {
        match self.overrides.as_ref().and_then(|o| o.model_names.as_ref()) {
            Some(names) => names.contains(name),
            None => self.by_group.contains_key(name),
        }
    }

    /// The group `model_group_alias` rewrites `name` to.
    pub(crate) fn alias_of(&self, name: &str) -> Option<&str> {
        self.aliases.get(name).map(String::as_str)
    }

    /// The callable routing group named `name`: a model group of that name shadows it.
    pub(crate) fn routing_group_named(&self, name: &str) -> Option<&RoutingGroupSpec> {
        (!self.indexes_name(name))
            .then(|| self.routing_groups.iter().find(|g| g.group_name == name))
            .flatten()
    }

    /// The most specific wildcard pattern matching `name`, with its deployments.
    pub(crate) fn wildcard_for(&self, name: &str) -> Option<(String, Vec<Arc<Deployment>>)> {
        self.wildcards
            .iter()
            .find(|(pattern, _)| wildcard_matches(pattern, name))
            .map(|(pattern, indices)| (pattern.clone(), self.at(indices)))
    }

    /// Resolve a requested name, in the order of `_common_checks_available_deployment`:
    /// a specific deployment, an id, an alias, a routing group, a model group, a wildcard.
    pub fn resolve(&self, name: &str, scope: ResolveScope) -> Option<Resolution> {
        let resolution = |kind, group: &str, deployments: Vec<Arc<Deployment>>| {
            (!deployments.is_empty()).then(|| Resolution {
                kind,
                group: group.to_owned(),
                deployments,
            })
        };
        if scope.specific_deployment {
            let matching = self
                .deployments
                .iter()
                .filter(|d| d.params.model == name)
                .cloned()
                .collect();
            // An empty match still resolves: the pick then finds no deployment rather than
            // falling back to a default (`_common_checks_available_deployment` returns `[]`).
            return Some(Resolution {
                kind: ResolveKind::SpecificDeployment,
                group: name.to_owned(),
                deployments: matching,
            });
        }
        if !self.by_group.contains_key(name)
            && let Some(deployment) = self.deployment(name)
        {
            let group = deployment.model_name.clone();
            return resolution(ResolveKind::Id, &group, vec![deployment]);
        }
        let routing_group_named = |name: &str| {
            // A model group of the same name shadows a routing group (`get_routing_group`).
            (!self.by_group.contains_key(name))
                .then(|| self.routing_groups.iter().find(|g| g.group_name == name))
                .flatten()
        };
        if let Some(target) = self.aliases.get(name) {
            if let Some(routing_group) = routing_group_named(target) {
                let deployments = crate::routing_groups::members(routing_group, |model| {
                    self.by_group
                        .get(model)
                        .map(|indices| self.at(indices))
                        .unwrap_or_default()
                });
                return resolution(ResolveKind::RoutingGroup, target, deployments);
            }
            return self
                .group(target)
                .and_then(|deployments| resolution(ResolveKind::Alias, target, deployments));
        }
        if let Some(routing_group) = routing_group_named(name) {
            let deployments = crate::routing_groups::members(routing_group, |model| {
                self.by_group
                    .get(model)
                    .map(|indices| self.at(indices))
                    .unwrap_or_default()
            });
            return resolution(ResolveKind::RoutingGroup, name, deployments);
        }
        if let Some(indices) = self.by_group.get(name) {
            return resolution(ResolveKind::Group, name, self.at(indices));
        }
        self.wildcards
            .iter()
            .find(|(pattern, _)| wildcard_matches(pattern, name))
            .and_then(|(pattern, indices)| {
                resolution(ResolveKind::Wildcard, pattern, self.at(indices))
            })
    }

    /// Access group name to the model groups in it.
    pub fn access_groups(&self) -> BTreeMap<String, BTreeSet<String>> {
        let mut groups: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for deployment in &self.deployments {
            for group in &deployment.model_info.access_groups {
                groups
                    .entry(group.clone())
                    .or_default()
                    .insert(deployment.model_name.clone());
            }
        }
        groups
    }

    pub fn credentials(&self, id: &str) -> Option<Credentials> {
        self.deployment(id).map(|deployment| Credentials {
            provider: deployment.provider().to_owned(),
            api_base: deployment.params.api_base.clone(),
            api_version: deployment.params.api_version.clone(),
            credential_name: deployment.params.litellm_credential_name.clone(),
            params: deployment.params.clone(),
        })
    }
}

/// `PatternUtils.calculate_pattern_specificity` over the pattern's regex
/// (`re.escape(pattern).replace(r"\*", "(.*)")`): its length, then how much regex syntax it has.
fn pattern_specificity(pattern: &str) -> (usize, usize) {
    let mut regex = String::new();
    for c in pattern.chars() {
        match c {
            '*' => regex.push_str("(.*)"),
            '(' | ')' | '[' | ']' | '{' | '}' | '?' | '+' | '-' | '|' | '^' | '$' | '\\' | '.'
            | '&' | '~' | '#' | ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c' => {
                regex.push('\\');
                regex.push(c);
            }
            _ => regex.push(c),
        }
    }
    let complexity = regex
        .chars()
        .filter(|c| matches!(c, '*' | '+' | '?' | '\\' | '^' | '$' | '|' | '(' | ')'))
        .count();
    (regex.chars().count(), complexity)
}

/// `*` matches any run of characters, as the pattern router's `(.*)` does under `re.match`:
/// anchored at the start only, so the name may run on past the pattern's last literal.
fn wildcard_matches(pattern: &str, name: &str) -> bool {
    let mut parts = pattern.split('*');
    let first = parts.next().unwrap_or_default();
    let Some(mut rest) = name.strip_prefix(first) else {
        return false;
    };
    let parts: Vec<&str> = parts.collect();
    for part in parts {
        match rest.find(part) {
            Some(at) => rest = &rest[at + part.len()..],
            None => return false,
        }
    }
    true
}

/// `deployment_id` in `litellm/router_api.py`: sha256 over the model group and every
/// `litellm_params` key and value in order. A string is taken as is, a dict as `json.dumps`, and
/// anything else as Python's `str()`.
pub fn deployment_id(model_group: &str, litellm_params: &Map<String, Value>) -> String {
    let mut text = String::from(model_group);
    for (key, value) in litellm_params {
        text.push_str(key);
        match value {
            Value::String(value) => text.push_str(value),
            Value::Object(_) => json_dumps(value, &mut text),
            Value::Null | Value::Bool(_) | Value::Number(_) | Value::Array(_) => {
                python_str(value, &mut text);
            }
        }
    }
    let digest = Sha256::digest(text.as_bytes());
    let mut hex = String::with_capacity(64);
    for byte in digest {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// Python's `repr(float)`: the shortest round-trip digits, positional for exponents in
/// `-4..16`, scientific otherwise.
fn python_float(value: f64, out: &mut String) {
    if value.is_nan() {
        out.push_str("nan");
        return;
    }
    if value.is_infinite() {
        out.push_str(if value > 0.0 { "inf" } else { "-inf" });
        return;
    }
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let (sign, mantissa) = mantissa
        .strip_prefix('-')
        .map_or(("", mantissa), |m| ("-", m));
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    out.push_str(sign);
    if (-4..16).contains(&exponent) {
        if exponent < 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-exponent - 1) as usize));
            out.push_str(&digits);
        } else {
            let point = exponent as usize + 1;
            if digits.len() <= point {
                out.push_str(&digits);
                out.extend(std::iter::repeat_n('0', point - digits.len()));
                out.push_str(".0");
            } else {
                out.push_str(&digits[..point]);
                out.push('.');
                out.push_str(&digits[point..]);
            }
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(
            out,
            "e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        );
    }
}

fn number(value: &serde_json::Number, out: &mut String) {
    match value.as_f64() {
        Some(float) if !value.is_i64() && !value.is_u64() => python_float(float, out),
        Some(_) | None => out.push_str(&value.to_string()),
    }
}

/// `json.dumps` with its defaults: `", "` and `": "` separators and `ensure_ascii`.
fn json_dumps(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Value::Number(value) => number(value, out),
        Value::String(text) => json_string(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                json_dumps(item, out);
            }
            out.push(']');
        }
        Value::Object(entries) => {
            out.push('{');
            for (index, (key, item)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                json_string(key, out);
                out.push_str(": ");
                json_dumps(item, out);
            }
            out.push('}');
        }
    }
}

fn json_string(text: &str, out: &mut String) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            ch if ch.is_ascii() && !ch.is_ascii_control() => out.push(ch),
            ch => {
                let mut units = [0u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}

/// Python's `str()` of a JSON value: `repr` for the items of a list or dict.
fn python_str(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("None"),
        Value::Bool(flag) => out.push_str(if *flag { "True" } else { "False" }),
        Value::Number(value) => number(value, out),
        Value::String(text) => python_repr_str(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                python_str(item, out);
            }
            out.push(']');
        }
        Value::Object(entries) => {
            out.push('{');
            for (index, (key, item)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                python_repr_str(key, out);
                out.push_str(": ");
                python_str(item, out);
            }
            out.push('}');
        }
    }
}

fn python_repr_str(text: &str, out: &mut String) {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(quote);
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch == quote => {
                out.push('\\');
                out.push(ch);
            }
            ch if (ch as u32) < 0x20 || ch as u32 == 0x7f => {
                let _ = write!(out, "\\x{:02x}", ch as u32);
            }
            ch => out.push(ch),
        }
    }
    out.push(quote);
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn spec(model_name: &str, params: Value, info: Value) -> DeploymentSpec {
        serde_json::from_value(json!({
            "model_name": model_name, "litellm_params": params, "model_info": info,
        }))
        .unwrap()
    }

    /// Expected ids computed with `litellm.router_api.deployment_id`.
    #[rstest]
    #[case::strings(
        "g",
        json!({"model": "openai/gpt-4o", "api_key": "sk-1"}),
        "30256ba697cc98ce2aff32c3b73c4b90514cb199876aef429ffcd91ae826d025"
    )]
    #[case::scalars(
        "claude",
        json!({"model": "anthropic/claude-3", "rpm": 10, "timeout": 1.5, "stream": true, "x": null}),
        "645fd1504089c1084316573ccfc5dd1dd9622623b169026657484fb64a2c89a2"
    )]
    #[case::list_and_dict(
        "m",
        json!({"model": "azure/x", "tags": ["a", "b'c"], "extra_body": {"k": [1, 2.0, "é"], "z": {"q": false}}}),
        "64f11e16dabed321268bd92132de2020290ef193c0cf554b2c74faee8badbb96"
    )]
    fn deployment_id_matches_python(
        #[case] group: &str,
        #[case] params: Value,
        #[case] expected: &str,
    ) {
        let Value::Object(params) = params else {
            unreachable!()
        };
        assert_eq!(deployment_id(group, &params), expected);
    }

    #[rstest]
    #[case::integral(2.0, "2.0")]
    #[case::small(0.00001, "1e-05")]
    #[case::large(1e16, "1e+16")]
    #[case::fraction(0.1, "0.1")]
    fn floats_print_as_python_repr(#[case] value: f64, #[case] expected: &str) {
        let mut out = String::new();
        python_float(value, &mut out);
        assert_eq!(out, expected);
    }

    #[rstest]
    #[case::group("gpt-4", ResolveKind::Group, "gpt-4")]
    #[case::alias("smart", ResolveKind::Alias, "gpt-4")]
    #[case::id("dep-b", ResolveKind::Id, "gpt-4")]
    #[case::wildcard("anthropic/claude-x", ResolveKind::Wildcard, "anthropic/*")]
    fn resolve_follows_python_order(
        #[case] name: &str,
        #[case] kind: ResolveKind,
        #[case] group: &str,
    ) {
        let registry = Registry::build(
            vec![
                spec(
                    "gpt-4",
                    json!({"model": "openai/gpt-4"}),
                    json!({"id": "dep-a"}),
                ),
                spec(
                    "gpt-4",
                    json!({"model": "azure/gpt-4"}),
                    json!({"id": "dep-b"}),
                ),
                spec("anthropic/*", json!({"model": "anthropic/*"}), json!({})),
            ],
            BTreeMap::from([("smart".to_owned(), "gpt-4".to_owned())]),
            Vec::new(),
        )
        .unwrap();
        let resolution = registry.resolve(name, ResolveScope::default()).unwrap();
        assert_eq!((resolution.kind, resolution.group.as_str()), (kind, group));
    }

    #[rstest]
    fn apply_moves_an_edited_deployment_to_the_end() {
        let registry = Registry::build(
            vec![
                spec("a", json!({"model": "openai/a"}), json!({"id": "1"})),
                spec("a", json!({"model": "openai/b"}), json!({"id": "2"})),
            ],
            BTreeMap::new(),
            Vec::new(),
        )
        .unwrap();
        let next = registry
            .apply(
                &ChangeSet {
                    upserts: vec![spec("a", json!({"model": "openai/c"}), json!({"id": "1"}))],
                    ..ChangeSet::default()
                },
                None,
            )
            .unwrap();
        let ids: Vec<String> = next
            .group("a")
            .unwrap()
            .iter()
            .map(|d| d.id.clone())
            .collect();
        assert_eq!(
            (next.version(), ids),
            (1, vec!["2".to_owned(), "1".to_owned()])
        );
    }
}
