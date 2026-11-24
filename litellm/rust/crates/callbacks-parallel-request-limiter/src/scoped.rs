//! The limiter's entry points outside the V2 pre-call: the body-free check, provider capacity, MCP
//! servers, Responses WebSocket turns, router fallback targets, proxy sub-calls, batches and the
//! Skills Hub. Each builds its descriptors with the pre-call's builder from the same typed key
//! records, then reads, charges or reserves them on the shared engine.

use litellm_callbacks::{Abort, CallbackError, KeyIdentity, KeyLimits, LimiterRequest};
use litellm_redis_store::RequestRedis;
use serde::Deserialize;
use serde_json::Value;

use crate::{
    builder::{AuthMetadata, BuildError, Lookups, Request, Selection},
    descriptor::Descriptor,
    engine::{Ctx, Unverifiable},
    limiter::{
        ParallelRequestLimiter, PolicyFacts, agent_policy, auth_members, customer_scopes,
        python_int_error, request_facts,
    },
    refusal,
    state::LimiterState,
    status::Response,
};

/// What an entry point does with the descriptors it built.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Action {
    /// Only build them.
    #[default]
    Build,
    /// `should_rate_limit`: read, or check and increment by one request.
    Check {
        #[serde(default)]
        read_only: bool,
        #[serde(default)]
        skip_tpm_check: bool,
        #[serde(default)]
        parallel_slot_id: Option<String>,
        #[serde(default)]
        skip_parallel_check: bool,
    },
    /// All-or-nothing check and increment of every descriptor by the same amounts.
    Charge {
        #[serde(default)]
        requests: i64,
        #[serde(default)]
        tokens: i64,
    },
    /// Reserve the input estimate on ITPM and the output estimate on OTPM.
    ReserveIo {
        input_tokens: i64,
        output_tokens: i64,
    },
    SettleIo {
        input_tokens: i64,
        output_tokens: i64,
    },
    /// Reserve a router fallback target's per-model limits, recording the model on the request.
    FallbackTarget { model: String },
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct ScopedInputs {
    pub key_identity: Option<KeyIdentity>,
    pub key_limits: Option<KeyLimits>,
    pub limiter_request: Option<LimiterRequest>,
    pub requested_model: Option<String>,
    pub metadata: AuthMetadata,
    pub lookups: Lookups,
    pub selection: Selection,
    pub action: Action,
    pub policy: PolicyFacts,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Built,
    Checked(Result<Response, Unverifiable>),
    Allowed(bool),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Scoped {
    pub descriptors: Vec<Descriptor>,
    pub outcome: Outcome,
}

fn failed(what: &str, error: impl std::fmt::Display) -> CallbackError {
    CallbackError::Failed(format!("the scoped limiter {what} do not decode: {error}"))
}

impl ParallelRequestLimiter {
    /// The descriptors an entry point checks, or the Python error the Python builder raised.
    pub fn scoped_descriptors(
        &self,
        inputs: &ScopedInputs,
    ) -> Result<Result<Vec<Descriptor>, Abort>, CallbackError> {
        let limits = inputs.key_limits.clone();
        let (rpm_limit_type, tpm_limit_type, model_has_failures) = match inputs.selection {
            Selection::Batch { .. } => (None, None, false),
            Selection::SubCall => (
                limits
                    .as_ref()
                    .and_then(|limits| limits.rpm_limit_type.clone()),
                limits
                    .as_ref()
                    .and_then(|limits| limits.tpm_limit_type.clone()),
                false,
            ),
            _ => (
                limits
                    .as_ref()
                    .and_then(|limits| limits.rpm_limit_type.clone()),
                limits
                    .as_ref()
                    .and_then(|limits| limits.tpm_limit_type.clone()),
                inputs.lookups.model_has_failures,
            ),
        };
        let lookups = Lookups {
            managed_agent_policy: agent_policy(
                limits
                    .as_ref()
                    .and_then(|limits| limits.managed_agent_policy.as_ref()),
            )?,
            invoked_agent_policy: agent_policy(
                limits
                    .as_ref()
                    .and_then(|limits| limits.invoked_agent_policy.as_ref()),
            )?,
            model_has_failures,
            ..inputs.lookups.clone()
        };
        let metadata = AuthMetadata {
            customer_scopes: customer_scopes(limits.as_ref()),
            ..inputs.metadata.clone()
        };
        let request = request_facts(inputs.limiter_request.clone())?;
        let auth = auth_members(inputs.key_identity.clone(), limits)?;
        let requested_model = inputs
            .requested_model
            .as_deref()
            .filter(|model| !model.is_empty());
        let built = match (Request {
            auth: &auth,
            rpm_limit_type: rpm_limit_type.as_deref(),
            tpm_limit_type: tpm_limit_type.as_deref(),
            metadata: &metadata,
            lookups: &lookups,
            request: &request,
            requested_model,
            window_size: self.engine().settings().window_size,
        })
        .select(&inputs.selection)
        {
            Ok(built) => built,
            Err(BuildError::Python { exception, message }) => {
                return Ok(Err(refusal::python_error(&exception, &message)));
            }
            Err(error) => return Err(CallbackError::Failed(error.to_string())),
        };
        match serde_json::from_value(Value::Array(built)) {
            Ok(descriptors) => Ok(Ok(descriptors)),
            Err(error) => {
                let message = error.to_string();
                match python_int_error(&message) {
                    Some((exception, raised)) => Ok(Err(refusal::python_error(exception, raised))),
                    None => Err(failed("descriptors", message)),
                }
            }
        }
    }

    /// Build an entry point's descriptors and run its action on the shared engine. An empty set
    /// is never sent to the engine: it admits.
    pub async fn scoped(
        &self,
        inputs: ScopedInputs,
        state: Option<&LimiterState>,
        request: Option<&RequestRedis>,
    ) -> Result<Result<Scoped, Abort>, CallbackError> {
        let descriptors = match self.scoped_descriptors(&inputs)? {
            Ok(descriptors) => descriptors,
            Err(abort) => return Ok(Err(abort)),
        };
        let ctx = Ctx {
            policy: inputs.policy.into(),
            request,
        };
        let empty = descriptors.is_empty();
        let outcome = match inputs.action {
            Action::Build => Outcome::Built,
            Action::FallbackTarget { model } => {
                match self
                    .fallback_target_allowed(
                        state,
                        descriptors.clone(),
                        &model,
                        inputs.policy,
                        request,
                    )
                    .await
                {
                    Ok(allowed) => Outcome::Allowed(allowed),
                    Err(abort) => return Ok(Err(*abort)),
                }
            }
            _ if empty => Outcome::Checked(Ok(Response::ok(Vec::new()))),
            Action::Check {
                read_only,
                skip_tpm_check,
                parallel_slot_id,
                skip_parallel_check,
            } => Outcome::Checked(
                self.engine()
                    .should_rate_limit(
                        &descriptors,
                        read_only,
                        skip_tpm_check,
                        parallel_slot_id.as_deref(),
                        skip_parallel_check,
                        ctx,
                    )
                    .await,
            ),
            Action::Charge { requests, tokens } => {
                let increments = vec![(requests, tokens); descriptors.len()];
                Outcome::Checked(
                    self.engine()
                        .atomic_check_and_increment(&descriptors, &increments, ctx)
                        .await,
                )
            }
            Action::ReserveIo {
                input_tokens,
                output_tokens,
            } => Outcome::Checked(
                self.engine()
                    .reserve_io_tokens(&descriptors, input_tokens, output_tokens, ctx)
                    .await
                    .map(|(response, _, _)| response),
            ),
            Action::SettleIo {
                input_tokens,
                output_tokens,
            } => {
                self.engine()
                    .settle_io_tokens(&descriptors, input_tokens, output_tokens, ctx)
                    .await;
                Outcome::Checked(Ok(Response::ok(Vec::new())))
            }
        };
        Ok(Ok(Scoped {
            descriptors,
            outcome,
        }))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::builder::{
        TpdLimit, batch_descriptors, skill_hub_publish_descriptor, socket_turn_tpm_windows,
        sub_call_descriptor,
    };

    fn descriptor(key: &str, value: &str, requests: Value, tokens: Value) -> Value {
        json!({"key": key, "value": value, "rate_limit": {
            "requests_per_unit": requests, "tokens_per_unit": tokens, "max_parallel_requests": null,
            "window_size": 60}})
    }

    #[rstest]
    fn a_socket_turn_keeps_only_token_windows_and_adds_unbuilt_agent_windows() {
        let built = vec![
            descriptor("api_key", "k", json!(5), json!(100)),
            descriptor("user", "u", json!(5), json!(null)),
            descriptor("model_per_project_itpm", "p:m", json!(null), json!(9)),
            descriptor("agent", "a", json!(null), json!(7)),
        ];
        let agent_windows = vec![
            descriptor("agent", "a", json!(null), json!(70)),
            descriptor("agent_session", "a:s", json!(null), json!(8)),
        ];
        assert_eq!(
            socket_turn_tpm_windows(built, &agent_windows, false),
            vec![
                json!({"key": "api_key", "value": "k", "rate_limit": {"tokens_per_unit": 100, "window_size": 60}}),
                json!({"key": "agent", "value": "a", "rate_limit": {"tokens_per_unit": 7, "window_size": 60}}),
                json!({"key": "agent_session", "value": "a:s", "rate_limit": {"tokens_per_unit": 8, "window_size": 60}}),
            ]
        );
    }

    #[rstest]
    fn a_socket_turn_that_charges_project_io_keeps_its_project_windows_but_not_ptu() {
        let built = vec![
            descriptor("model_per_project_itpm", "p:m", json!(null), json!(9)),
            descriptor("model_per_project_otpm", "p:m", json!(null), json!(4)),
            descriptor("model_per_team_ptu", "t:m", json!(null), json!(6)),
        ];
        assert_eq!(
            socket_turn_tpm_windows(built, &[], true),
            vec![
                json!({"key": "model_per_project_itpm", "value": "p:m",
                    "rate_limit": {"tokens_per_unit": 9, "window_size": 60}}),
                json!({"key": "model_per_project_otpm", "value": "p:m",
                    "rate_limit": {"tokens_per_unit": 4, "window_size": 60}}),
            ]
        );
    }

    #[rstest]
    #[case::caller_wide_rpm_is_dropped("team", json!(null))]
    #[case::model_rpm_is_kept("model_per_team", json!(3))]
    fn a_sub_call_keeps_token_limits_and_only_model_rpm(
        #[case] key: &str,
        #[case] requests: Value,
    ) {
        assert_eq!(
            sub_call_descriptor(&descriptor(key, "t", json!(3), json!(40))),
            json!({"key": key, "value": "t", "rate_limit": {
                "requests_per_unit": requests, "tokens_per_unit": 40, "window_size": 60}})
        );
    }

    #[rstest]
    fn a_daily_token_limit_replaces_the_scopes_minute_descriptor() {
        let descriptors = batch_descriptors(
            vec![
                descriptor("api_key", "k", json!(5), json!(100)),
                descriptor("team", "t", json!(5), json!(100)),
            ],
            &[
                TpdLimit {
                    key: "api_key".to_owned(),
                    value: json!("k"),
                    limit: json!(1000),
                },
                TpdLimit {
                    key: "end_user".to_owned(),
                    value: json!(null),
                    limit: json!(10),
                },
            ],
        );
        assert_eq!(
            descriptors,
            vec![
                descriptor("team", "t", json!(5), json!(100)),
                json!({"key": "api_key_tpd", "value": "k", "rate_limit": {
                    "requests_per_unit": null, "tokens_per_unit": 1000, "window_size": 86_400}}),
            ]
        );
    }

    #[rstest]
    fn an_mcp_call_charges_the_key_every_granting_team_and_the_server() {
        let Value::Object(auth) = json!({"api_key": "k", "team_id": "t"}) else {
            unreachable!("the auth facts are an object")
        };
        let built = Request {
            auth: &auth,
            rpm_limit_type: None,
            tpm_limit_type: None,
            metadata: &AuthMetadata::default(),
            lookups: &Lookups::default(),
            request: &crate::builder::RequestFacts::default(),
            requested_model: None,
            window_size: 60,
        }
        .select(&Selection::Mcp(Box::new(crate::builder::McpScope {
            server_name: Some("s".to_owned()),
            server_id: json!("sid"),
            server_rpm: json!(9),
            key_rpm_limit: json!({"s": 3}),
            team_rpm_limit: json!(null),
            source_team_rpm_limits: vec![
                ("t".to_owned(), json!({"s": 4})),
                ("u".to_owned(), json!({"s": 5})),
            ],
        })))
        .unwrap();
        let rpm = |key: &str, value: &str, limit: i64| {
            json!({"key": key, "value": value, "rate_limit": {
                "requests_per_unit": limit, "tokens_per_unit": null, "window_size": 60}})
        };
        assert_eq!(
            built,
            vec![
                rpm("mcp_per_key", "k:s", 3),
                rpm("mcp_per_team", "t:s", 4),
                rpm("mcp_per_team", "u:s", 5),
                rpm("mcp_server", "sid", 9),
            ]
        );
    }

    #[rstest]
    fn the_skill_hub_budget_is_ten_a_minute() {
        assert_eq!(
            skill_hub_publish_descriptor("h"),
            json!({"key": "skill_hub_publish", "value": "h", "rate_limit": {
                "requests_per_unit": 10, "window_size": 60}})
        );
    }
}
