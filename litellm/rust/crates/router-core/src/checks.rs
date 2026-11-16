//! The optional pre-call checks the core runs itself, for a host that does not select:
//! `encrypted_content_affinity` (`EncryptedContentAffinityCheck`) and
//! `responses_api_deployment_check` (`DeploymentAffinityCheck`'s Responses API continuity). Both pin
//! a follow-up request to the deployment that served the turn it continues, which the host reads
//! off the request's own ids into [`Affinity`]. A host that selects runs Python's checks through
//! [`crate::RouterHost::callback_filters`] instead.

use std::{collections::BTreeMap, str::FromStr, sync::Arc};

use crate::{
    host::RoutingContext,
    registry::{Deployment, Registry},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum NativeCheck {
    EncryptedContentAffinity,
    ResponsesApiDeploymentCheck,
}

/// `optional_pre_call_checks` and `model_group_affinity_config`.
#[derive(Clone, Debug, Default, PartialEq, serde::Deserialize)]
#[serde(default)]
pub struct PreCallChecks {
    pub global: Vec<String>,
    /// A group's own affinity checks, which replace the global ones for that group.
    pub groups: BTreeMap<String, Vec<String>>,
}

fn lists(names: &[String], check: NativeCheck) -> bool {
    let name: &'static str = check.into();
    names.iter().any(|listed| listed == name)
}

fn all_native(names: &[String]) -> bool {
    names.iter().all(|name| NativeCheck::from_str(name).is_ok())
}

impl PreCallChecks {
    /// Every `optional_pre_call_checks` entry is a check the core runs.
    pub fn global_is_native(&self) -> bool {
        all_native(&self.global)
    }

    /// Every `model_group_affinity_config` entry is a check the core runs.
    pub fn groups_are_native(&self) -> bool {
        self.groups.values().all(|names| all_native(names))
    }

    pub fn is_empty(&self) -> bool {
        self.global.is_empty() && self.groups.is_empty()
    }

    /// `EncryptedContentAffinityCheck.is_enabled_for_model_group`.
    pub fn encrypted_content_affinity(&self, model_group: &str) -> bool {
        lists(&self.global, NativeCheck::EncryptedContentAffinity)
            || self
                .groups
                .get(model_group)
                .is_some_and(|names| lists(names, NativeCheck::EncryptedContentAffinity))
    }

    /// `DeploymentAffinityCheck._get_effective_flags`: a group's own list wins over the global one.
    fn responses_api(&self, model_group: &str) -> bool {
        lists(
            self.groups.get(model_group).unwrap_or(&self.global),
            NativeCheck::ResponsesApiDeploymentCheck,
        )
    }
}

/// The deployments a request's own ids name, which the host decodes from them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Affinity {
    /// The deployment `previous_response_id` names (`get_model_id_from_response_id`).
    pub previous_response: Option<String>,
    /// The deployment the first encrypted-content marker in the request names
    /// (`_extract_model_id_from_input`).
    pub encrypted_content: Option<String>,
}

/// The upstream account a deployment's encrypted reasoning decrypts under
/// (`_deployment_encryption_boundary`): an operator-declared `model_info.encrypted_content_boundary`,
/// else its explicit `api_base` and `api_key`. A deployment that leaves either to the provider's
/// defaults has none here, so it is never a peer.
#[derive(Debug, PartialEq)]
struct Boundary<'a> {
    declared: Option<&'a str>,
    pair: Option<(&'a str, &'a str)>,
}

impl<'a> Boundary<'a> {
    fn of(deployment: &'a Deployment) -> Option<Self> {
        let declared = deployment
            .model_info
            .raw
            .get("encrypted_content_boundary")
            .and_then(serde_json::Value::as_str)
            .filter(|name| !name.is_empty());
        let params = &deployment.params;
        let pair = params
            .api_base
            .as_deref()
            .filter(|base| !base.is_empty())
            .zip(
                params
                    .api_key
                    .as_ref()
                    .map(|key| key.expose())
                    .filter(|key| !key.is_empty()),
            );
        (declared.is_some() || pair.is_some()).then_some(Self { declared, pair })
    }

    fn is_shared_with(&self, other: &Self) -> bool {
        match (self.declared, other.declared) {
            (Some(own), Some(theirs)) => own == theirs,
            _ => self.pair.is_some() && self.pair == other.pair,
        }
    }
}

/// Whether `target` decrypts reasoning `origin` minted: it is `origin`, or shares its boundary.
pub fn decrypts(registry: &Registry, origin: &str, target: &Deployment) -> bool {
    if origin == target.id {
        return true;
    }
    let Some(origin) = registry.deployment(origin) else {
        return false;
    };
    Boundary::of(&origin)
        .zip(Boundary::of(target))
        .is_some_and(|(origin, target)| origin.is_shared_with(&target))
}

/// The checks in the order Python registers them: encrypted-content affinity, then the
/// deployment affinity check that carries Responses API continuity.
pub(crate) fn filter(
    checks: &PreCallChecks,
    ctx: &RoutingContext<'_>,
    candidates: Vec<Arc<Deployment>>,
) -> Vec<Arc<Deployment>> {
    let candidates = encrypted_content_affinity(checks, ctx, candidates);
    responses_api_continuity(checks, ctx, candidates)
}

/// `EncryptedContentAffinityCheck.async_filter_deployments`, less the tag stage: tag filtering is
/// a host stage, refused on a host that does not select.
fn encrypted_content_affinity(
    checks: &PreCallChecks,
    ctx: &RoutingContext<'_>,
    candidates: Vec<Arc<Deployment>>,
) -> Vec<Arc<Deployment>> {
    let affinity = &ctx.request.affinity;
    let Some(origin) = affinity
        .encrypted_content
        .as_deref()
        .filter(|_| checks.encrypted_content_affinity(ctx.model_group))
    else {
        return candidates;
    };
    let candidates: Vec<_> = match ctx.target_order {
        Some(order) => candidates
            .into_iter()
            .filter(|deployment| deployment.order() == Some(order))
            .collect(),
        None => candidates,
    };
    let binds_origin = affinity.previous_response.as_deref() == Some(origin);
    let listed = |own: &[String], walked: &[String]| -> Vec<String> {
        if binds_origin {
            return Vec::new();
        }
        own.iter().chain(walked).cloned().collect()
    };
    let excluded = listed(&ctx.request.excluded_ids, ctx.excluded_ids);
    let skipped = listed(&ctx.request.retry_skipped_ids, ctx.skipped_ids);
    let not_excluded: Vec<_> = candidates
        .iter()
        .filter(|deployment| !excluded.contains(&deployment.id))
        .cloned()
        .collect();
    let untried: Vec<_> = not_excluded
        .iter()
        .filter(|deployment| !skipped.contains(&deployment.id))
        .cloned()
        .collect();
    let untried = if untried.is_empty() {
        not_excluded
    } else {
        untried
    };
    if let Some(pinned) = untried.iter().find(|deployment| deployment.id == origin) {
        return vec![pinned.clone()];
    }
    let peers: Vec<_> = untried
        .iter()
        .filter(|deployment| decrypts(ctx.registry, origin, deployment))
        .cloned()
        .collect();
    if !peers.is_empty() {
        return peers;
    }
    if excluded.is_empty() && skipped.is_empty() {
        candidates
    } else {
        untried
    }
}

/// `DeploymentAffinityCheck.async_filter_deployments`'s `previous_response_id` pin. Its user-key and
/// session pins need state and are refused on a host that does not select.
fn responses_api_continuity(
    checks: &PreCallChecks,
    ctx: &RoutingContext<'_>,
    candidates: Vec<Arc<Deployment>>,
) -> Vec<Arc<Deployment>> {
    if ctx.target_order.is_some() || !checks.responses_api(ctx.model_group) {
        return candidates;
    }
    let Some(previous) = ctx.request.affinity.previous_response.as_deref() else {
        return candidates;
    };
    match candidates
        .iter()
        .find(|deployment| deployment.id == previous)
    {
        Some(pinned) => vec![pinned.clone()],
        None => candidates,
    }
}
