//! Port of Python's `bedrock_invoke_guardrail_blocked` and `BedrockInvokeStreamGuardrail`: whether
//! a Bedrock Invoke reply a guardrail intervened in was blocked, or answered with masked text.
//!
//! Bedrock sends `amazon-bedrock-guardrailAction: INTERVENED` for a block and a mask alike. With
//! trace on, a block is a policy whose action is `BLOCKED`. With trace off, a block is a reply the
//! model never ran for, or one Bedrock wrote itself under a `bedrock-guardrails-` id.

use std::collections::BTreeSet;

use litellm_llms_types::providers::bedrock::{
    GUARDRAIL_ACTION_KEY, INVOCATION_METRICS_KEY, TRACE_KEY,
};
use litellm_python_compat::{BigInt, Value};

const INTERVENED: &str = "INTERVENED";
const BLOCKED: &str = "BLOCKED";
const GUARDRAIL_REPLY_ID_PREFIX: &str = "bedrock-guardrails-";
const TOKEN_COUNT_FIELDS: [&str; 8] = [
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "completion_tokens",
    "inputTokens",
    "outputTokens",
    "prompt_token_count",
    "generation_token_count",
];

type Fields = [(Value, Value)];

/// Port of Python's `BedrockInvokeStreamGuardrail`: the guardrail evidence collected over every
/// event of an Invoke stream, since each family sends it on a different event.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct InvokeStreamGuardrail {
    intervened: bool,
    trace_actions: Option<BTreeSet<String>>,
    model_ran: bool,
    guardrail_wrote_reply: bool,
}

impl InvokeStreamGuardrail {
    /// An event that is not an object adds no evidence.
    pub(crate) fn with_event(self, event: &Value) -> Self {
        let Value::Dict(fields) = event else {
            return self;
        };
        Self {
            intervened: self.intervened || intervened(fields),
            trace_actions: match (trace_actions(fields), self.trace_actions) {
                (None, held) => held,
                (Some(actions), held) => Some(
                    actions
                        .into_iter()
                        .chain(held.into_iter().flatten())
                        .collect(),
                ),
            },
            model_ran: self.model_ran || reports_model_run(fields),
            guardrail_wrote_reply: self.guardrail_wrote_reply || guardrail_wrote_reply(fields),
        }
    }

    pub(crate) fn blocked(&self) -> bool {
        if !self.intervened {
            return false;
        }
        match &self.trace_actions {
            Some(actions) => actions.contains(BLOCKED),
            None => self.guardrail_wrote_reply || !self.model_ran,
        }
    }
}

/// Port of Python's `bedrock_invoke_guardrail_blocked` for a whole reply body. Python falls back to
/// the `x-amzn-bedrock-invocation-latency` header for a body without token counts, which a Claude
/// Messages reply always carries, so a body without them counts as answered here.
pub(crate) fn invoke_guardrail_blocked(reply: &Fields) -> bool {
    if !intervened(reply) {
        return false;
    }
    match trace_actions(reply) {
        Some(actions) => actions.contains(BLOCKED),
        None => model_never_ran(reply),
    }
}

fn get<'a>(fields: &'a Fields, name: &str) -> Option<&'a Value> {
    fields
        .iter()
        .find(|(key, _)| matches!(key, Value::Str(key) if key == name))
        .map(|(_, value)| value)
}

fn dict(value: Option<&Value>) -> Option<&Fields> {
    match value {
        Some(Value::Dict(fields)) => Some(fields),
        _ => None,
    }
}

fn is_zero(value: &Value) -> bool {
    match value {
        Value::Int(number) => *number == BigInt::ZERO,
        Value::Float(number) => *number == 0.0,
        _ => false,
    }
}

fn intervened(fields: &Fields) -> bool {
    matches!(get(fields, GUARDRAIL_ACTION_KEY), Some(Value::Str(action)) if action == INTERVENED)
}

fn token_counts(fields: Option<&Fields>) -> impl Iterator<Item = &Value> {
    fields.into_iter().flat_map(|fields| {
        TOKEN_COUNT_FIELDS
            .iter()
            .filter_map(|name| get(fields, name))
    })
}

fn model_never_ran(reply: &Fields) -> bool {
    let counts: Vec<&Value> = token_counts(Some(reply))
        .chain(token_counts(dict(get(reply, "usage"))))
        .collect();
    !counts.is_empty() && counts.into_iter().all(is_zero)
}

fn reports_model_run(event: &Fields) -> bool {
    dict(get(event, INVOCATION_METRICS_KEY))
        .is_some_and(|metrics| !get(metrics, "invocationLatency").is_some_and(is_zero))
}

fn guardrail_wrote_reply(event: &Fields) -> bool {
    matches!(get(event, "id"), Some(Value::Str(id)) if id.starts_with(GUARDRAIL_REPLY_ID_PREFIX))
}

fn trace_actions(payload: &Fields) -> Option<BTreeSet<String>> {
    let guardrail = dict(get(dict(get(payload, TRACE_KEY))?, "guardrail"))?;
    Some(object_actions(guardrail))
}

fn object_actions(fields: &Fields) -> BTreeSet<String> {
    let own = match get(fields, "action") {
        Some(Value::Str(action)) => Some(action.clone()),
        _ => None,
    };
    own.into_iter()
        .chain(fields.iter().flat_map(|(_, value)| actions(value)))
        .collect()
}

fn actions(node: &Value) -> BTreeSet<String> {
    match node {
        Value::List(items) => items.iter().flat_map(actions).collect(),
        Value::Dict(fields) => object_actions(fields),
        _ => BTreeSet::new(),
    }
}

#[cfg(test)]
mod tests {
    use litellm_python_compat::json;
    use rstest::rstest;

    use super::*;

    fn fields(text: &str) -> Vec<(Value, Value)> {
        match json::loads(text).unwrap() {
            Value::Dict(fields) => fields,
            other => panic!("not an object: {other:?}"),
        }
    }

    fn streamed(events: &[Vec<&str>]) -> InvokeStreamGuardrail {
        events
            .iter()
            .map(|fragments| {
                json::loads(&format!(
                    r#"{{"type": "message_stop"{}}}"#,
                    fragments.concat()
                ))
                .unwrap()
            })
            .fold(InvokeStreamGuardrail::default(), |guardrail, event| {
                guardrail.with_event(&event)
            })
    }

    const BLOCKED_TRACE: &str = r#""amazon-bedrock-trace": {"guardrail": {"input": {"gr": {"wordPolicy": {"customWords": [{"match": "hexblade", "action": "BLOCKED"}]}}}}}"#;
    const MASKED_TRACE: &str = r#""amazon-bedrock-trace": {"guardrail": {"input": {"gr": {"sensitiveInformationPolicy": {"piiEntities": [{"type": "US_SOCIAL_SECURITY_NUMBER", "action": "ANONYMIZED"}]}}}}}"#;

    #[rstest]
    #[case::blocked_with_zero_usage(
        r#"{"amazon-bedrock-guardrailAction": "INTERVENED", "usage": {"input_tokens": 0, "output_tokens": 0}}"#,
        true
    )]
    #[case::masked_after_the_model_ran(
        r#"{"amazon-bedrock-guardrailAction": "INTERVENED", "usage": {"input_tokens": 11, "output_tokens": 7}}"#,
        false
    )]
    #[case::a_masking_trace_wins_over_zero_usage(
        &format!(r#"{{"amazon-bedrock-guardrailAction": "INTERVENED", "usage": {{"input_tokens": 0, "output_tokens": 0}}, {MASKED_TRACE}}}"#),
        false
    )]
    #[case::a_blocking_trace_wins_over_real_usage(
        &format!(r#"{{"amazon-bedrock-guardrailAction": "INTERVENED", "usage": {{"input_tokens": 11, "output_tokens": 7}}, {BLOCKED_TRACE}}}"#),
        true
    )]
    #[case::a_trace_without_a_guardrail_reads_the_usage(
        r#"{"amazon-bedrock-guardrailAction": "INTERVENED", "usage": {"input_tokens": 0}, "amazon-bedrock-trace": {"guardrail": []}}"#,
        true
    )]
    #[case::top_level_counts_count(
        r#"{"amazon-bedrock-guardrailAction": "INTERVENED", "prompt_token_count": 0, "generation_token_count": 0.0}"#,
        true
    )]
    #[case::no_counts_is_an_answer(r#"{"amazon-bedrock-guardrailAction": "INTERVENED"}"#, false)]
    #[case::a_passed_guardrail(
        r#"{"amazon-bedrock-guardrailAction": "NONE", "usage": {"input_tokens": 0, "output_tokens": 0}}"#,
        false
    )]
    #[case::no_guardrail(r#"{"usage": {"input_tokens": 0, "output_tokens": 0}}"#, false)]
    fn a_reply_is_blocked_only_when_the_guardrail_stopped_the_model(
        #[case] reply: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(invoke_guardrail_blocked(&fields(reply)), expected);
    }

    const INTERVENED: &str = r#", "amazon-bedrock-guardrailAction": "INTERVENED""#;
    const RAN: &str = r#", "amazon-bedrock-invocationMetrics": {"invocationLatency": 412}"#;
    const NEVER_RAN: &str = r#", "amazon-bedrock-invocationMetrics": {"invocationLatency": 0}"#;
    const NO_LATENCY: &str = r#", "amazon-bedrock-invocationMetrics": {}"#;
    const GUARDRAIL_REPLY_ID: &str = r#", "id": "bedrock-guardrails-19a3""#;

    #[rstest]
    #[case::intervened_without_metrics(vec![vec![], vec![INTERVENED]], true)]
    #[case::masked_with_metrics(vec![vec![INTERVENED, RAN]], false)]
    #[case::zero_latency_means_the_model_never_ran(vec![vec![INTERVENED, NEVER_RAN]], true)]
    #[case::metrics_without_latency_mean_it_ran(vec![vec![INTERVENED, NO_LATENCY]], false)]
    #[case::a_reply_bedrock_wrote(vec![vec![GUARDRAIL_REPLY_ID], vec![INTERVENED, RAN]], true)]
    #[case::a_masking_trace(vec![vec![INTERVENED, ", ", MASKED_TRACE]], false)]
    #[case::traces_add_up_across_events(vec![vec![", ", BLOCKED_TRACE], vec![INTERVENED, RAN, ", ", MASKED_TRACE]], true)]
    #[case::a_passed_guardrail(vec![vec![r#", "amazon-bedrock-guardrailAction": "NONE""#]], false)]
    #[case::no_guardrail(vec![vec![]], false)]
    fn a_stream_is_blocked_by_what_all_its_events_say(
        #[case] events: Vec<Vec<&str>>,
        #[case] expected: bool,
    ) {
        assert_eq!(streamed(&events).blocked(), expected);
    }

    #[rstest]
    #[case::a_list(Value::List(vec![Value::from(1)]))]
    #[case::a_string(Value::from("bedrock-guardrails-19a3"))]
    #[case::none(Value::None)]
    fn an_event_that_is_not_an_object_adds_no_evidence(#[case] event: Value) {
        let intervened = streamed(&[vec![INTERVENED, RAN]]);

        assert_eq!(intervened.clone().with_event(&event), intervened);
    }
}
