//! The `optional_pre_call_checks` the core runs itself, as stages of an attempt on the picked
//! deployment. A host runs the ones [`core_runs`] leaves out, and a host that runs them all
//! leaves [`PreCallChecks`] empty, so no check runs twice.

mod enforce_model_rate_limits;

use std::{sync::Arc, time::SystemTime};

use litellm_router_types::OptionalPreCallCheck;
use serde::Deserialize;

use crate::{cache::RouterCache, error::Error, registry::Deployment};

/// The `optional_pre_call_checks` entries the core runs itself, in the order listed.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(from = "Vec<OptionalPreCallCheck>")]
pub struct PreCallChecks(Arc<[OptionalPreCallCheck]>);

/// Whether the core has stages for `check`. A host runs the ones it does not.
pub fn core_runs(check: &OptionalPreCallCheck) -> bool {
    use OptionalPreCallCheck::*;
    match check {
        EnforceModelRateLimits => true,
        PromptCaching
        | RouterBudgetLimiting
        | ResponsesApiDeploymentCheck
        | DeploymentAffinity
        | SessionAffinity
        | ForwardClientHeadersByModelGroup
        | EncryptedContentAffinity
        | Other(_) => false,
    }
}

impl From<Vec<OptionalPreCallCheck>> for PreCallChecks {
    fn from(listed: Vec<OptionalPreCallCheck>) -> Self {
        Self::of(&listed)
    }
}

impl PreCallChecks {
    pub fn of(listed: &[OptionalPreCallCheck]) -> Self {
        Self(
            listed
                .iter()
                .filter(|check| core_runs(check))
                .cloned()
                .collect(),
        )
    }

    /// `async_pre_call_check` on the picked deployment. The first refusal ends the attempt.
    pub(crate) async fn admit(
        &self,
        cache: &impl RouterCache,
        deployment: &Deployment,
        now: SystemTime,
    ) -> Result<(), Error> {
        for check in self.0.iter() {
            #[expect(
                clippy::single_match,
                reason = "one arm per check with an admit stage, as succeeded has"
            )]
            match check {
                OptionalPreCallCheck::EnforceModelRateLimits => {
                    enforce_model_rate_limits::admit(cache, deployment, now).await?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// `async_log_success_event` of each check.
    pub(crate) async fn succeeded(
        &self,
        cache: &impl RouterCache,
        deployment: &Deployment,
        total_tokens: u64,
        now: SystemTime,
    ) {
        for check in self.0.iter() {
            #[expect(
                clippy::single_match,
                reason = "one arm per check with a success stage, as admit has"
            )]
            match check {
                OptionalPreCallCheck::EnforceModelRateLimits => {
                    enforce_model_rate_limits::count_tokens(cache, deployment, total_tokens, now)
                        .await;
                }
                _ => {}
            }
        }
    }
}
