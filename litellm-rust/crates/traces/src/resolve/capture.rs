use std::collections::{BTreeMap, HashSet};

use super::{actors, resolution::Resolution};
use crate::{
    ObservationType,
    view::{ActorCapture, TraceCapture},
};

pub(super) fn coverage(resolution: &Resolution<'_>) -> Option<TraceCapture> {
    let rows = resolution.graph.rows;
    if !rows.iter().any(actors::native) {
        return None;
    }
    let calls: HashSet<_> = resolution.model_calls.iter().copied().collect();
    let mut actors: BTreeMap<_, _> = resolution
        .actors
        .entries
        .values()
        .map(|actor| {
            (
                actor.id.as_str(),
                ActorCapture {
                    actor_id: actor.id.clone(),
                    name: actor.name.clone(),
                    ..Default::default()
                },
            )
        })
        .collect();
    let mut capture = TraceCapture::default();
    for (index, row) in rows.iter().enumerate() {
        let event = matches!(
            row.name.as_str(),
            "claude_code.user_prompt"
                | "claude_code.assistant_response"
                | "claude_code.tool_result"
                | "claude_code.api_request_body"
        );
        capture.content_events += u64::from(event);
        capture.warning_events += u64::from(row.capture_warning);
        let Some(actor) = resolution
            .actors
            .owner(index)
            .and_then(|owner| actors.get_mut(owner.id.as_str()))
        else {
            capture.unassigned_events += u64::from(event);
            continue;
        };
        actor.llm_calls += u64::from(calls.contains(&index));
        actor.tool_calls += u64::from(resolution.kind(index) == ObservationType::Tool);
        actor.reply_events += u64::from(row.name == "claude_code.assistant_response");
        actor.model_outputs += u64::from(calls.contains(&index) && row.has_output);
        actor.content_events += u64::from(event);
    }
    capture.actors = actors.into_values().collect();
    Some(capture)
}
