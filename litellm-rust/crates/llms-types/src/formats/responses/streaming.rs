use serde_json::{Map, Value};

/// A Responses stream event's `type`, for the events the codec reads or writes itself. Every
/// other event passes through under its own name.
#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    strum::AsRefStr,
    strum::EnumString,
    strum::Display,
    serde_with::DeserializeFromStr,
    serde_with::SerializeDisplay,
)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(with = "String"))]
pub enum ResponsesStreamEventType {
    #[strum(serialize = "response.created")]
    Created,
    #[strum(serialize = "response.in_progress")]
    InProgress,
    #[strum(serialize = "response.completed")]
    Completed,
    #[strum(serialize = "response.failed")]
    Failed,
    #[strum(serialize = "response.incomplete")]
    Incomplete,
    #[strum(serialize = "response.output_item.added")]
    OutputItemAdded,
    #[strum(serialize = "response.output_item.done")]
    OutputItemDone,
    #[strum(serialize = "response.content_part.added")]
    ContentPartAdded,
    #[strum(serialize = "response.content_part.done")]
    ContentPartDone,
    #[strum(serialize = "response.output_text.delta")]
    OutputTextDelta,
    #[strum(serialize = "response.output_text.annotation.added")]
    OutputTextAnnotationAdded,
    #[strum(serialize = "response.output_text.done")]
    OutputTextDone,
    #[strum(serialize = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta,
    #[strum(serialize = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone,
    #[strum(default, transparent)]
    Other(String),
}

impl ResponsesStreamEventType {
    /// Whether the event closes the stream and carries the complete response.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Incomplete)
    }
}

/// One server-sent event of a Responses stream: its `type` and the rest of its JSON object.
#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ResponsesStreamEvent {
    #[serde(rename = "type")]
    pub event_type: ResponsesStreamEventType,
    #[serde(flatten)]
    pub data: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::completed(json!({"type": "response.completed", "response": {"id": "r"}}), ResponsesStreamEventType::Completed, true)]
    #[case::incomplete(json!({"type": "response.incomplete", "response": {"id": "r"}}), ResponsesStreamEventType::Incomplete, true)]
    #[case::failed(json!({"type": "response.failed", "response": {"id": "r"}}), ResponsesStreamEventType::Failed, true)]
    #[case::delta(json!({"type": "response.output_text.delta", "delta": "hi"}), ResponsesStreamEventType::OutputTextDelta, false)]
    #[case::unknown(json!({"type": "response.audio.delta", "delta": "AA=="}), ResponsesStreamEventType::Other("response.audio.delta".into()), false)]
    fn an_event_keeps_its_type_and_fields_through_serde(
        #[case] wire: Value,
        #[case] event_type: ResponsesStreamEventType,
        #[case] terminal: bool,
    ) {
        let event: ResponsesStreamEvent = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(event.event_type, event_type);
        assert_eq!(event.event_type.is_terminal(), terminal);
        assert_eq!(serde_json::to_value(&event).unwrap(), wire);
    }
}
