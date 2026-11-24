//! The params stage of `_pre_call_checks`: while the live `litellm.drop_params` is off, a
//! deployment that does not support a param the request asks for is dropped, unless that leaves
//! none of what the earlier stages kept. Then only the deployments that reject the param are
//! dropped, those whose call settings neither drop nor allow it. A group whose every deployment
//! rejects it raises one `UnsupportedParams`.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use litellm_router_types::CallSettings;
use serde::Deserialize;

use super::PreCallFacts;
use crate::{error::Error, registry::Deployment};

/// The chat params a deployment must take when a request asks for them
/// (`_PRE_CALL_CHECK_FILTERED_PARAMS`), sorted.
pub const CHECKED_PARAMS: &[&str] = &["response_format"];

/// How a request sends a param, as its host reports it.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Sent {
    /// `None`, which is how a caller spells a param it does not set.
    Null,
    /// Text, as an image call's `response_format` is. It asks no chat deployment for anything.
    Text,
    /// Any other value.
    Structured,
}

/// The params each deployment does not support, by deployment id
/// (`_deployment_supports_param`). A deployment that is not listed supports every one.
pub type Unsupported = BTreeMap<String, BTreeSet<String>>;

/// The params the stage checks (`_pre_call_check_requested_chat_params`): each of
/// [`CHECKED_PARAMS`] the request sends as a structured value, while it carries input and
/// `litellm.drop_params` is off.
pub fn requested(facts: &PreCallFacts) -> Vec<&'static str> {
    if !facts.has_input || facts.drop_params != Some(false) {
        return Vec::new();
    }
    CHECKED_PARAMS
        .iter()
        .copied()
        .filter(|param| facts.sent_params.get(*param) == Some(&Sent::Structured))
        .collect()
}

/// One request's params check.
pub(super) struct Check<'a> {
    requested: Vec<&'static str>,
    sent: &'a CallSettings,
    unsupported: &'a Unsupported,
}

impl<'a> Check<'a> {
    pub(super) fn of(facts: &'a PreCallFacts, unsupported: &'a Unsupported) -> Self {
        Self {
            requested: requested(facts),
            sent: &facts.call_settings,
            unsupported,
        }
    }

    /// Which of `left` the stage drops: those lacking a requested param, unless every one lacks
    /// it, and then only those that reject it.
    pub(super) fn failing(&self, left: &[&Deployment]) -> Vec<bool> {
        let lacking: Vec<bool> = left
            .iter()
            .map(|deployment| self.lacks(deployment, self.requested.iter().copied()))
            .collect();
        if lacking.is_empty() || lacking.contains(&false) {
            return lacking;
        }
        left.iter()
            .map(|deployment| self.rejects(deployment))
            .collect()
    }

    /// The error for a group with nothing left, when every one of its deployments rejects a
    /// requested param.
    pub(super) fn unsupported(&self, model: &str, pool: &[Arc<Deployment>]) -> Option<Error> {
        let every_rejects =
            !pool.is_empty() && pool.iter().all(|deployment| self.rejects(deployment));
        every_rejects.then(|| Error::UnsupportedParams {
            model: model.to_owned(),
            params: self
                .requested
                .iter()
                .map(|param| (*param).to_owned())
                .collect(),
        })
    }

    /// `_deployment_lacks_params`: it does not support one of `params`.
    fn lacks(
        &self,
        deployment: &Deployment,
        mut params: impl Iterator<Item = &'static str>,
    ) -> bool {
        self.unsupported
            .get(&deployment.id)
            .is_some_and(|unsupported| params.any(|param| unsupported.contains(param)))
    }

    /// `_deployment_rejects_params`: it lacks a requested param its call settings neither drop
    /// nor allow. The request's settings replace the deployment's, one key at a time.
    fn rejects(&self, deployment: &Deployment) -> bool {
        let own = &deployment.params.call_settings;
        if self.sent.drop_params.or(own.drop_params) == Some(true) {
            return false;
        }
        let allowed = self
            .sent
            .allowed_openai_params
            .as_deref()
            .or(own.allowed_openai_params.as_deref())
            .unwrap_or_default();
        let dropped = self
            .sent
            .additional_drop_params
            .as_deref()
            .or(own.additional_drop_params.as_deref())
            .unwrap_or_default();
        let named = |names: &[String], param: &str| names.iter().any(|name| name == param);
        let kept = self
            .requested
            .iter()
            .copied()
            .filter(|param| !named(allowed, param) && !named(dropped, param));
        self.lacks(deployment, kept)
    }
}
