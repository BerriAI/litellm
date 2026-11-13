use litellm_llms_types::{
    formats::messages::{MessagesOptionalParams, MessagesRequest, MessagesTool},
    recognized::Recognized,
};
use serde_json::{Map, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolCustomField {
    Keep,
    Rewrite { hoist_defer_loading: bool },
}

impl ToolCustomField {
    pub fn apply(self, request: MessagesRequest, drop_params: bool) -> MessagesRequest {
        match self {
            Self::Keep => request,
            Self::Rewrite {
                hoist_defer_loading,
            } => tools_without_custom_field(request, hoist_defer_loading, drop_params),
        }
    }
}

fn tools_without_custom_field(
    request: MessagesRequest,
    hoist: bool,
    drop_custom: bool,
) -> MessagesRequest {
    let tools = request.params.tools.map(|tools| {
        tools
            .into_iter()
            .map(|tool| tool_without_custom_field(tool, hoist, drop_custom))
            .collect()
    });
    MessagesRequest {
        params: MessagesOptionalParams {
            tools,
            ..request.params
        },
        ..request
    }
}

fn tool_without_custom_field(
    tool: Recognized<MessagesTool>,
    hoist: bool,
    drop_custom: bool,
) -> Recognized<MessagesTool> {
    let Ok(Value::Object(fields)) = serde_json::to_value(&tool) else {
        return tool;
    };
    let Some(custom) = fields.get("custom").cloned() else {
        return tool;
    };
    let defer_loading = hoist
        .then(|| hoistable_defer_loading(&fields, &custom))
        .flatten();
    let without_custom = fields
        .iter()
        .filter(|(key, _)| *key != "custom")
        .map(|(key, value)| (key.clone(), value.clone()));
    let rebuilt: Map<String, Value> = match defer_loading {
        None if drop_custom => without_custom.collect(),
        None => return tool,
        Some(defer_loading) => {
            let unhoisted: Map<String, Value> = custom
                .as_object()
                .into_iter()
                .flatten()
                .filter(|(key, _)| *key != "defer_loading")
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            let kept_custom = (!unhoisted.is_empty() && !drop_custom)
                .then(|| ("custom".to_string(), Value::Object(unhoisted)));
            without_custom
                .chain([("defer_loading".to_string(), Value::Bool(defer_loading))])
                .chain(kept_custom)
                .collect()
        }
    };
    serde_json::from_value(Value::Object(rebuilt)).unwrap_or(tool)
}

fn hoistable_defer_loading(tool: &Map<String, Value>, custom: &Value) -> Option<bool> {
    if tool.contains_key("defer_loading") {
        return None;
    }
    custom.as_object()?.get("defer_loading")?.as_bool()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    const WEATHER: &str = "get_weather";

    fn tool(fields: Value) -> Value {
        merged(
            json!({"name": WEATHER, "input_schema": {"type": "object"}}),
            fields,
        )
    }

    fn tools_after(policy: ToolCustomField, tools: Value, drop_custom: bool) -> Value {
        let request: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [tools]
        }))
        .unwrap();
        serde_json::to_value(policy.apply(request, drop_custom)).unwrap()["tools"][0].clone()
    }

    fn merged(base: Value, fields: Value) -> Value {
        let mut merged = base.as_object().unwrap().clone();
        merged.extend(fields.as_object().unwrap().clone());
        Value::Object(merged)
    }

    const HOIST: ToolCustomField = ToolCustomField::Rewrite {
        hoist_defer_loading: true,
    };
    const NO_HOIST: ToolCustomField = ToolCustomField::Rewrite {
        hoist_defer_loading: false,
    };

    #[rstest]
    #[case::hoists_defer_loading(
        HOIST,
        json!({"custom": {"defer_loading": true}}),
        false,
        json!({"defer_loading": true})
    )]
    #[case::keeps_other_custom_keys(
        HOIST,
        json!({"custom": {"defer_loading": false, "note": "kept"}}),
        false,
        json!({"defer_loading": false, "custom": {"note": "kept"}})
    )]
    #[case::drops_other_custom_keys_with_drop_params(
        HOIST,
        json!({"custom": {"defer_loading": true, "note": "dropped"}}),
        true,
        json!({"defer_loading": true})
    )]
    #[case::keeps_an_opaque_custom(
        HOIST,
        json!({"custom": "opaque"}),
        false,
        json!({"custom": "opaque"})
    )]
    #[case::drops_an_opaque_custom_with_drop_params(
        HOIST,
        json!({"custom": "opaque"}),
        true,
        json!({})
    )]
    #[case::own_defer_loading_wins(
        HOIST,
        json!({"defer_loading": false, "custom": {"defer_loading": true}}),
        false,
        json!({"defer_loading": false, "custom": {"defer_loading": true}})
    )]
    #[case::not_hoisted_off_claude_hosts(
        NO_HOIST,
        json!({"custom": {"defer_loading": true}}),
        false,
        json!({"custom": {"defer_loading": true}})
    )]
    fn tool_custom_field_follows_the_python_handler(
        #[case] policy: ToolCustomField,
        #[case] fields: Value,
        #[case] drop_custom: bool,
        #[case] expected: Value,
    ) {
        assert_eq!(
            tools_after(policy, tool(fields), drop_custom),
            tool(expected)
        );
    }

    #[rstest]
    fn keep_leaves_the_custom_field() {
        let custom = tool(json!({"custom": {"defer_loading": true}}));
        assert_eq!(
            tools_after(ToolCustomField::Keep, custom.clone(), true),
            custom
        );
    }
}
