typify::import_types!(
    schema = "../../../litellm/llms/anthropic/beta_detection.schema.json",
    patch = {
        BetaThinkingType = { derives = [strum::VariantArray] },
        BetaThinkingDisplay = { derives = [strum::VariantArray] },
        BetaContentType = { derives = [strum::VariantArray] },
        BetaToolTargetType = { derives = [strum::VariantArray] },
    },
    replace = {
        JsonValue = serde_json::Value,
        BetaThinkingValue = crate::recognized::Recognized<BetaThinking>,
        BetaDisplayValue = crate::recognized::Recognized<BetaThinkingDisplay>,
        BetaToolValue = crate::recognized::Recognized<BetaToolTarget>,
        BetaContentValue = crate::recognized::Recognized<BetaContentBlock>,
    },
);

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serde_json::{Value, json};

    #[rstest]
    #[case::future(json!({"type": "future", "display": "updates", "extension": {"keep": true}}))]
    #[case::scalar(json!("opaque thinking"))]
    #[case::array(json!(["unknown", 1]))]
    #[case::unknown_display(json!({"type": "enabled", "display": "future"}))]
    #[case::wrong_display_shape(json!({"type": "adaptive", "display": ["updates"]}))]
    fn unknown_thinking_survives_projection(#[case] thinking: Value) {
        let request: BetaDetectionRequest = serde_json::from_value(json!({
            "thinking": thinking,
            "messages": [],
        }))
        .unwrap();
        assert_eq!(
            serde_json::to_value(request.thinking.unwrap()).unwrap(),
            thinking
        );
    }

    #[rstest]
    fn unknown_tool_target_survives_projection() {
        let target = json!({"type": "future_tool", "extension": {"keep": true}});
        let block: BetaContentBlock = serde_json::from_value(json!({
            "type": "tool_addition",
            "tool": target,
        }))
        .unwrap();
        assert_eq!(serde_json::to_value(block.tool.unwrap()).unwrap(), target);
    }
}
