use litellm_callbacks::{Abort, CallbackError, EstimatedOutputTokens, Host, HostCall};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

use crate::{
    engine::{Ctx, Unverifiable},
    estimate::{self, EstimateRequest, Out},
    limiter::{Estimate, ParallelRequestLimiter, PreCallFacts},
    refusal,
    state::LimiterState,
};

const GENAI_CALL_TYPES: [&str; 4] = estimate::GENAI_CALL_TYPES;
const RESPONSES_CALL_TYPES: [&str; 2] = ["aresponses", "responses"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tokens(pub i64);

impl<'de> Deserialize<'de> for Tokens {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        crate::python::saturating_int(deserializer).map(Self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WideTokens(pub i64, pub f64);

impl<'de> Deserialize<'de> for WideTokens {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let saturated = crate::python::saturating_int(&value).map_err(serde::de::Error::custom)?;
        let exact = crate::python::int_as_float(&value)
            .ok_or_else(|| serde::de::Error::custom(format!("an int was expected, not {value}")))?;
        Ok(Self(saturated, exact))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OptionalTokens(pub Option<i64>);

impl<'de> Deserialize<'de> for OptionalTokens {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        crate::python::saturating_optional_int(deserializer).map(Self)
    }
}

#[derive(Serialize)]
pub struct ConfiguredOutputTokens {}

impl HostCall for ConfiguredOutputTokens {
    const NAME: &'static str = "configured_output_tokens";
    type Reply = OptionalTokens;
}

#[derive(Serialize)]
pub struct WriteOutputCap {
    pub cap: i64,
}

impl HostCall for WriteOutputCap {
    const NAME: &'static str = "write_output_cap";
    type Reply = bool;
}

#[derive(Serialize)]
pub struct EstimateTokens {
    pub min_configured_tpm_limit: i64,
    pub configured_output_tokens: Option<i64>,
}

impl HostCall for EstimateTokens {
    const NAME: &'static str = "estimate_tokens";
    type Reply = Tokens;
}

#[derive(Serialize)]
pub struct EstimatePtuTokens {
    pub min_configured_tpm_limit: i64,
    pub configured_output_tokens: Option<i64>,
}

impl HostCall for EstimatePtuTokens {
    const NAME: &'static str = "estimate_ptu_tokens";
    type Reply = (WideTokens, WideTokens);
}

#[derive(Serialize)]
pub struct EstimateIoOutputTokens {
    pub min_configured_tpm_limit: Option<i64>,
}

impl HostCall for EstimateIoOutputTokens {
    const NAME: &'static str = "estimate_io_output_tokens";
    type Reply = Tokens;
}

#[derive(Serialize)]
pub struct PreciseInputTokens {}

impl HostCall for PreciseInputTokens {
    const NAME: &'static str = "precise_input_tokens";
    type Reply = Tokens;
}

pub struct Reservation<'a> {
    pub host: &'a Host,
    pub body: Option<Value>,
    pub call_type: &'a str,
    pub model: Option<&'a str>,
    pub configured: Option<&'a EstimatedOutputTokens>,
}

fn saturated(count: i128) -> i64 {
    i64::try_from(count).unwrap_or(if count < 0 { i64::MIN } else { i64::MAX })
}

fn typed_configured(configured: &EstimatedOutputTokens, model: Option<&str>) -> Option<i64> {
    let per_model = |found: Option<i64>| found.filter(|_| model.is_some());
    per_model(configured.key_per_model)
        .or(configured.key_default)
        .or(per_model(configured.team_per_model))
        .or(configured.team_default)
}

fn written(body: &mut Value, call_type: &str, cap: i64) {
    let Value::Object(request) = body else {
        return;
    };
    if GENAI_CALL_TYPES.contains(&call_type) {
        let field = if request.contains_key("config") || !request.contains_key("generationConfig") {
            "config"
        } else {
            "generationConfig"
        };
        let mut config = match request.get(field) {
            Some(Value::Object(config)) => config.clone(),
            _ => Map::new(),
        };
        config.insert("maxOutputTokens".to_owned(), Value::from(cap));
        request.insert(field.to_owned(), Value::Object(config));
        return;
    }
    let field = if RESPONSES_CALL_TYPES.contains(&call_type) {
        "max_output_tokens"
    } else {
        "max_tokens"
    };
    request.insert(field.to_owned(), Value::from(cap));
}

enum Stop {
    Refused(Box<Abort>),
    Raised(CallbackError),
}

impl From<Unverifiable> for Stop {
    fn from(Unverifiable: Unverifiable) -> Self {
        Self::Refused(Box::new(refusal::unverifiable()))
    }
}

impl From<litellm_callbacks::HostCallError> for Stop {
    fn from(error: litellm_callbacks::HostCallError) -> Self {
        Self::Raised(error.into())
    }
}

fn refused(abort: Option<Abort>) -> Result<(), Stop> {
    abort.map_or(Ok(()), |abort| Err(Stop::Refused(Box::new(abort))))
}

impl Reservation<'_> {
    fn request(
        &self,
        min_configured_tpm_limit: Option<i64>,
        configured: Option<i64>,
        include_media: bool,
    ) -> Option<EstimateRequest<'_>> {
        Some(EstimateRequest {
            data: self.body.as_ref()?,
            call_type: self.call_type,
            min_configured_tpm_limit,
            configured_output_tokens: configured.map(i128::from),
            include_media,
        })
    }

    async fn configured_output_tokens(&self) -> Result<Option<i64>, Stop> {
        match self.configured {
            Some(configured) if configured.rejected.as_ref().is_none_or(Vec::is_empty) => {
                Ok(typed_configured(configured, self.model))
            }
            Some(_) => Ok(self.host.call(ConfiguredOutputTokens {}).await?.0),
            None => Ok(None),
        }
    }

    async fn write_cap(&mut self, cap: Option<i64>) -> Result<bool, Stop> {
        let Some(cap) = cap else {
            return Ok(false);
        };
        let applied = self.host.call(WriteOutputCap { cap }).await?;
        if applied && let Some(body) = self.body.as_mut() {
            written(body, self.call_type, cap);
        }
        Ok(applied)
    }

    async fn answered<T, C: HostCall>(
        &self,
        answer: Option<Out<T>>,
        call: C,
    ) -> Result<Result<T, C::Reply>, Stop> {
        match answer {
            Some(Ok(answer)) => Ok(Ok(answer)),
            _ => Ok(Err(self.host.call(call).await?)),
        }
    }
}

impl ParallelRequestLimiter {
    pub(crate) async fn pre_call_hosted(
        &self,
        state: &LimiterState,
        facts: PreCallFacts,
        ctx: Ctx<'_>,
        reservation: Reservation<'_>,
    ) -> Result<Option<Abort>, CallbackError> {
        match self.admit_hosted(state, facts, ctx, reservation).await {
            Ok(()) => Ok(None),
            Err(Stop::Refused(abort)) => Ok(Some(*abort)),
            Err(Stop::Raised(error)) => Err(error),
        }
    }

    async fn admit_hosted(
        &self,
        state: &LimiterState,
        facts: PreCallFacts,
        ctx: Ctx<'_>,
        mut reservation: Reservation<'_>,
    ) -> Result<(), Stop> {
        refused(self.first_pass(state, &facts, ctx).await?)?;
        if !state.stash().first_pass_admitted {
            return Ok(());
        }
        let plan = self.plan(state);
        let cap = self.cap_settings();
        let configured = if plan.reserves {
            reservation.configured_output_tokens().await?
        } else {
            None
        };
        let cap_applied = plan.reserves
            && reservation
                .write_cap(self.implicit_output_cap(state, plan.min_tpm_limit, configured, false))
                .await?;
        let facts = match plan.min_tpm_limit.filter(|_| plan.reserves) {
            Some(min) => {
                let total = reservation
                    .request(Some(min), configured, true)
                    .map(|request| estimate::tokens_for_request(request, &cap));
                let estimated_tokens = match reservation
                    .answered(
                        total,
                        EstimateTokens {
                            min_configured_tpm_limit: min,
                            configured_output_tokens: configured,
                        },
                    )
                    .await?
                {
                    Ok(total) => saturated(total),
                    Err(Tokens(total)) => total,
                };
                if let Some(declared) = configured
                    && estimated_tokens > min
                {
                    tracing::debug!(
                        "Reserving {estimated_tokens} tokens for model {} (declared default_estimated_output_tokens={declared} plus the input estimate) exceeds the smallest TPM limit this request is charged against ({min}), so it cannot be admitted even against an empty window. Lower the declared estimate or raise the TPM limit.",
                        facts.requested_model.as_deref().unwrap_or("None"),
                    );
                }
                let (ptu_estimate, ptu_exact) = if plan.ptu {
                    let split = reservation
                        .request(Some(min), configured, true)
                        .map(|request| estimate::input_and_output_tokens(request, &cap));
                    let (input_tokens, output_tokens, exact) = match reservation
                        .answered(
                            split,
                            EstimatePtuTokens {
                                min_configured_tpm_limit: min,
                                configured_output_tokens: configured,
                            },
                        )
                        .await?
                    {
                        Ok((input, output)) => (
                            saturated(input),
                            saturated(output),
                            (input as f64, output as f64),
                        ),
                        Err((WideTokens(input, wide_input), WideTokens(output, wide_output))) => {
                            (input, output, (wide_input, wide_output))
                        }
                    };
                    (
                        Some(Estimate {
                            input_tokens,
                            output_tokens,
                            explicit_output_cap: false,
                        }),
                        Some(exact),
                    )
                } else {
                    (None, None)
                };
                PreCallFacts {
                    estimated_tokens: Some(estimated_tokens),
                    ptu_estimate,
                    ptu_exact,
                    ..facts
                }
            }
            None => facts,
        };
        if plan.reserves {
            refused(self.reserve_tokens(state, &facts, ctx).await?)?;
        }
        if !plan.has_io {
            return Ok(());
        }
        let split = reservation
            .request(plan.min_otpm_limit, None, false)
            .map(|request| {
                estimate::input_and_output_tokens(request, &cap).map(|(_, output)| output)
            });
        let output_tokens = match reservation
            .answered(
                split,
                EstimateIoOutputTokens {
                    min_configured_tpm_limit: plan.min_otpm_limit,
                },
            )
            .await?
        {
            Ok(output) => saturated(output),
            Err(Tokens(output)) => output,
        };
        let Tokens(input_tokens) = reservation.host.call(PreciseInputTokens {}).await?;
        reservation
            .write_cap(self.implicit_output_cap(state, plan.min_otpm_limit, None, cap_applied))
            .await?;
        let explicit_output_cap = state.stash().cap.explicit || cap_applied;
        let facts = PreCallFacts {
            io_estimate: Some(Estimate {
                input_tokens,
                output_tokens,
                explicit_output_cap,
            }),
            ..facts
        };
        let amount = self.reservation_amount(&facts);
        refused(self.reserve_io(state, &facts, amount, ctx).await?)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::chat("acompletion", json!({"max_tokens": 900}), json!({"max_tokens": 25}))]
    #[case::responses("aresponses", json!({}), json!({"max_output_tokens": 25}))]
    #[case::genai_config("agenerate_content", json!({"config": {"a": 1}}), json!({"config": {"a": 1, "maxOutputTokens": 25}}))]
    #[case::genai_generation_config(
        "agenerate_content",
        json!({"generationConfig": {"a": 1}}),
        json!({"generationConfig": {"a": 1, "maxOutputTokens": 25}})
    )]
    #[case::genai_neither("generate_content", json!({}), json!({"config": {"maxOutputTokens": 25}}))]
    fn an_applied_cap_lands_where_python_writes_it(
        #[case] call_type: &str,
        #[case] body: Value,
        #[case] expected: Value,
    ) {
        let mut body = body;
        written(&mut body, call_type, 25);
        assert_eq!(body, expected);
    }

    #[rstest]
    #[case::key_model(Some("m"), Some(7))]
    #[case::no_model(None, Some(3))]
    fn the_configured_estimate_prefers_the_key_then_the_team(
        #[case] model: Option<&str>,
        #[case] expected: Option<i64>,
    ) {
        let configured = EstimatedOutputTokens {
            key_per_model: Some(7),
            key_default: None,
            team_per_model: Some(9),
            team_default: Some(3),
            ..EstimatedOutputTokens::default()
        };
        assert_eq!(typed_configured(&configured, model), expected);
    }
}
