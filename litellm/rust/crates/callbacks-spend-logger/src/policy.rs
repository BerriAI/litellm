//! The cost callback's decisions over the request document (`_PROXY_track_cost_callback` and
//! its helpers in `litellm/proxy/hooks/proxy_track_cost_callback.py`): how a success is treated
//! before any charge, whether its key identity is looked up, who may be charged with no identity,
//! the tags a row charges and what a repeat video poll still pays for.

use litellm_python_compat::{json, truthy::json_truthy};
use serde_json::Value as Json;

use litellm_callbacks_spend_log::{VIDEO_STATUS_CALL_TYPES, guardrail_information_cost};

use crate::hook_input::Success;

/// `_UNATTRIBUTED_TRACKABLE_CALL_TYPES`: pass-through, MCP and a managed batch's cost, tracked
/// with no identity.
const UNATTRIBUTED_TRACKABLE_CALL_TYPES: [&str; 5] = [
    "pass_through_endpoint",
    "llm_passthrough_route",
    "allm_passthrough_route",
    "call_mcp_tool",
    "aretrieve_batch",
];
/// `_CAPTURED_IDENTITY_CALL_TYPES`, both spellings.
const CAPTURED_IDENTITY_CALL_TYPES: [&str; 2] = ["aretrieve_batch", "CallTypes.aretrieve_batch"];

/// How the success callback treats a request before any charge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Disposition {
    /// The cost is known: gate, then charge it.
    Charge(f64),
    /// `defers_batch_cost`: a batch whose cost is not final yet.
    DeferBatch,
    /// An in-progress background interaction the poll task settles later.
    InteractionInProgress,
    /// An interaction create with no usage that no poll task settles.
    InteractionUnbilled,
    /// No cost and nothing to defer: an error, since a priced call lost its cost.
    Unpriced,
    /// No cost and nothing to charge: a call with no logging payload, or a stream still running.
    Released,
}

pub(crate) fn payload(row: &Json) -> Option<&Json> {
    row.get("standard_logging_object")
        .filter(|payload| !payload.is_null())
}

/// The `kwargs` model the row input logged.
pub(crate) fn model(row: &Json) -> Option<&Json> {
    row.get("kwargs").and_then(|kwargs| kwargs.get("model"))
}

/// `cost` is the logged response cost, `None` when the call has none or it is not finite.
pub(crate) fn disposition(success: &Success, cost: Option<f64>) -> Disposition {
    if success.response.defers_batch_cost {
        return Disposition::DeferBatch;
    }
    if let Some(cost) = cost {
        return Disposition::Charge(cost);
    }
    if success.response.unbilled_interaction {
        return if success.settings.background_interaction_cost_polling
            && success.response.interaction_in_progress
        {
            Disposition::InteractionInProgress
        } else {
            Disposition::InteractionUnbilled
        };
    }
    let row = &success.row.row;
    if payload(row).is_none()
        && (!model(row).is_some_and(json_truthy)
            || success.call_type.as_deref() == Some("_arealtime"))
    {
        return Disposition::Released;
    }
    if success.stream && !success.stream_complete {
        return Disposition::Released;
    }
    Disposition::Unpriced
}

/// Whether the key identity is looked up, and with `resolve_missing_key_identity` set: only for a
/// key that carried no user.
pub(crate) fn enrich(success: &Success) -> Option<bool> {
    let identity = &success.identity;
    let keyed = identity
        .token
        .as_deref()
        .is_some_and(|token| !token.is_empty());
    let has_user = identity
        .user_id
        .as_deref()
        .is_some_and(|user| !user.is_empty());
    (keyed && !has_user).then(|| {
        success
            .call_type
            .as_deref()
            .is_none_or(|kind| !CAPTURED_IDENTITY_CALL_TYPES.contains(&kind))
    })
}

pub(crate) fn tracks_unattributed(call_type: Option<&str>) -> bool {
    call_type.is_some_and(|kind| UNATTRIBUTED_TRACKABLE_CALL_TYPES.contains(&kind))
}

fn strings(items: &[Json]) -> Vec<String> {
    items
        .iter()
        .filter_map(|item| item.as_str().map(str::to_owned))
        .collect()
}

/// `_get_request_tags_for_cost_tracking`: the payload's `request_tags`, else the metadata tags.
pub(crate) fn tags(row: &Json) -> Option<Vec<String>> {
    if let Some(Json::Array(tags)) = payload(row).and_then(|payload| payload.get("request_tags")) {
        return Some(strings(tags));
    }
    match row
        .get("metadata")
        .and_then(|metadata| metadata.get("tags"))
    {
        Some(Json::Array(tags)) => Some(strings(tags)),
        _ => None,
    }
}

/// `repeat_poll_cost`: once an earlier poll charged the job, a video poll still pays for its
/// guardrails and a batch poll pays nothing.
pub(crate) fn repeat_poll_cost(call_type: Option<&str>, cost: f64, row: &Json) -> f64 {
    if !call_type.is_some_and(|kind| VIDEO_STATUS_CALL_TYPES.contains(&kind)) {
        return 0.0;
    }
    let guardrails = payload(row)
        .and_then(|payload| payload.get("guardrail_information"))
        .map_or(0.0, |information| {
            guardrail_information_cost(&json::from_json(information.clone()))
        });
    let finite = if cost.is_finite() { cost } else { 0.0 };
    finite.min(guardrails).max(0.0)
}

/// `JOB_POLL_GUARDRAIL_COST_REQUEST_ID_SUFFIX` from `spend_tracking_utils.py`.
const JOB_POLL_GUARDRAIL_COST_REQUEST_ID_SUFFIX: &str = "_guardrail_cost";

/// `get_spend_logs_id` for a repeat video poll that pays only its guardrails: the poll's own
/// `litellm_call_id` with the guardrail cost suffix, so the row never settles the job and is
/// charged like any other request. With no call id it falls back to the logging payload's id.
pub(crate) fn guardrail_cost_request_id(row: &Json) -> Option<String> {
    let non_empty = |value: Option<&Json>| {
        value
            .and_then(Json::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_owned)
    };
    match non_empty(
        row.get("kwargs")
            .and_then(|kwargs| kwargs.get("litellm_call_id")),
    ) {
        Some(call_id) => Some(format!(
            "{call_id}{JOB_POLL_GUARDRAIL_COST_REQUEST_ID_SUFFIX}"
        )),
        None => non_empty(payload(row).and_then(|payload| payload.get("id"))),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::hook_input::{Identity, ResponseFacts, RowRequest};

    fn success(row: Json) -> Success {
        Success {
            row: RowRequest {
                row,
                ..RowRequest::default()
            },
            ..Success::default()
        }
    }

    #[rstest]
    #[case::no_payload_no_model(json!({"standard_logging_object": null, "kwargs": {"model": ""}}), false, false, Disposition::Released)]
    #[case::running_stream(json!({"standard_logging_object": {}, "kwargs": {"model": "m"}}), true, false, Disposition::Released)]
    #[case::finished_stream(json!({"standard_logging_object": {}, "kwargs": {"model": "m"}}), true, true, Disposition::Unpriced)]
    #[case::priced_model_lost_its_cost(json!({"standard_logging_object": null, "kwargs": {"model": "m"}}), false, false, Disposition::Unpriced)]
    fn an_unpriced_success_is_released_only_when_nothing_was_priced(
        #[case] row: Json,
        #[case] stream: bool,
        #[case] stream_complete: bool,
        #[case] expected: Disposition,
    ) {
        let request = Success {
            stream,
            stream_complete,
            ..success(row)
        };
        assert_eq!(disposition(&request, None), expected);
    }

    #[rstest]
    #[case::polling_on(true, Disposition::InteractionInProgress)]
    #[case::polling_off(false, Disposition::InteractionUnbilled)]
    fn an_in_progress_interaction_waits_only_for_a_poll_task(
        #[case] polling: bool,
        #[case] expected: Disposition,
    ) {
        let mut request = success(json!({}));
        request.response = ResponseFacts {
            unbilled_interaction: true,
            interaction_in_progress: true,
            ..ResponseFacts::default()
        };
        request.settings.background_interaction_cost_polling = polling;
        assert_eq!(disposition(&request, None), expected);
    }

    #[test]
    fn a_deferred_batch_is_not_charged_even_when_priced() {
        let mut request = success(json!({}));
        request.response.defers_batch_cost = true;
        assert_eq!(disposition(&request, Some(1.0)), Disposition::DeferBatch);
    }

    #[rstest]
    #[case::keyed_without_user(Some("key-1"), None, Some("acompletion"), Some(true))]
    #[case::batch_cost_keeps_captured_identity(
        Some("key-1"),
        None,
        Some("aretrieve_batch"),
        Some(false)
    )]
    #[case::keyed_with_user(Some("key-1"), Some("u"), None, None)]
    #[case::empty_key(Some(""), None, None, None)]
    fn only_a_key_with_no_user_is_enriched(
        #[case] token: Option<&str>,
        #[case] user_id: Option<&str>,
        #[case] call_type: Option<&str>,
        #[case] expected: Option<bool>,
    ) {
        let request = Success {
            identity: Identity {
                token: token.map(str::to_owned),
                user_id: user_id.map(str::to_owned),
                ..Identity::default()
            },
            call_type: call_type.map(str::to_owned),
            ..Success::default()
        };
        assert_eq!(enrich(&request), expected);
    }

    #[rstest]
    #[case::payload_tags(json!({"standard_logging_object": {"request_tags": ["a", 1]}, "metadata": {"tags": ["b"]}}), Some(vec!["a"]))]
    #[case::metadata_tags(json!({"standard_logging_object": {"request_tags": null}, "metadata": {"tags": ["b"]}}), Some(vec!["b"]))]
    #[case::none(json!({"metadata": {}}), None)]
    fn tags_come_from_the_payload_before_the_metadata(
        #[case] row: Json,
        #[case] expected: Option<Vec<&str>>,
    ) {
        assert_eq!(
            tags(&row),
            expected.map(|tags| tags.into_iter().map(str::to_owned).collect())
        );
    }

    #[rstest]
    #[case::video_poll_pays_its_guardrail(Some("avideo_status"), 1.0, 0.25)]
    #[case::guardrail_capped_at_the_cost(Some("video_status"), 0.1, 0.1)]
    #[case::batch_poll_is_free(Some("aretrieve_batch"), 1.0, 0.0)]
    fn a_repeat_poll_pays_only_for_its_guardrails(
        #[case] call_type: Option<&str>,
        #[case] cost: f64,
        #[case] expected: f64,
    ) {
        let row = json!({"standard_logging_object": {"guardrail_information": [{"guardrail_cost": 0.25}]}});
        assert!((repeat_poll_cost(call_type, cost, &row) - expected).abs() < 1e-12);
    }

    #[rstest]
    #[case::call_id(json!({"kwargs": {"litellm_call_id": "call-1"}, "standard_logging_object": {"id": "p"}}), Some("call-1_guardrail_cost"))]
    #[case::payload_id(json!({"kwargs": {"litellm_call_id": ""}, "standard_logging_object": {"id": "p"}}), Some("p"))]
    #[case::none(json!({"kwargs": {}}), None)]
    fn a_guardrail_only_poll_is_charged_under_its_own_id(
        #[case] row: Json,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(guardrail_cost_request_id(&row).as_deref(), expected);
    }
}
