//! Request shaping shared by every host speaking the OpenAI Responses wire format, matching
//! `litellm/llms/openai/responses/transformation.py`

use serde_json::{Map, Value};
use uuid::Uuid;

use crate::{Error, ErrorDetail};

/// The Responses API rejects `max_output_tokens` below this for every model
pub const MIN_MAX_OUTPUT_TOKENS: i64 = 16;

const REASONING_ITEM_ID_PREFIX: &str = "rs_";
/// Python's `DEFAULT_MAX_RECURSE_DEPTH` for `filter_value_from_dict`
const MAX_STRIP_DEPTH: usize = 100;
const REASONING_FIELDS_DROPPED_WHEN_NULL: [&str; 3] = ["status", "content", "encrypted_content"];

/// The host behind the OpenAI wire format. OpenAI and Azure run OpenAI's item validator, which
/// rejects reasoning it did not mint and tool-call item ids without its prefixes, and Azure also
/// rejects `status` on replayed reasoning and message items. xAI replaces OpenAI's param mapping,
/// so it keeps `max_output_tokens` as sent
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponsesHost {
    OpenAi,
    Azure,
    Compatible,
    Xai,
}

impl ResponsesHost {
    fn validates_items(self) -> bool {
        matches!(self, Self::OpenAi | Self::Azure)
    }

    fn raises_max_output_tokens(self) -> bool {
        !matches!(self, Self::Xai)
    }
}

pub fn shape(
    input: Value,
    mut params: Map<String, Value>,
    host: ResponsesHost,
) -> Result<(Value, Map<String, Value>), Error> {
    if host.raises_max_output_tokens()
        && let Some(max_output_tokens) = params.get_mut("max_output_tokens")
        && max_output_tokens
            .as_i64()
            .is_some_and(|tokens| (1..MIN_MAX_OUTPUT_TOKENS).contains(&tokens))
    {
        *max_output_tokens = MIN_MAX_OUTPUT_TOKENS.into();
    }
    if let Some(Value::Array(tools)) = params.get_mut("tools") {
        tools
            .iter_mut()
            .filter_map(Value::as_object_mut)
            .for_each(|tool| strip_cache_control(tool, 0));
        tools
            .iter_mut()
            .enumerate()
            .try_for_each(|(index, tool)| decode_tool_parameters(index, tool))?;
    }
    let Value::Array(items) = input else {
        return Ok((input, params));
    };
    let items = items
        .into_iter()
        .filter(|item| !host.validates_items() || !is_minted_reasoning(item))
        .map(|item| without_null_reasoning_fields(item, host))
        .map(|mut item| {
            if let Some(fields) = item.as_object_mut() {
                strip_cache_control(fields, 0);
            }
            item
        })
        .map(|item| match host.validates_items() {
            true => without_foreign_tool_call_id(item),
            false => item,
        })
        .collect();
    Ok((Value::Array(items), params))
}

/// Anthropic-only `cache_control` marks, which OpenAI rejects as unknown parameters, removed at
/// any depth like Python's `filter_value_from_dict`
fn strip_cache_control(fields: &mut Map<String, Value>, depth: usize) {
    if depth > MAX_STRIP_DEPTH {
        return;
    }
    fields.remove("cache_control");
    for value in fields.values_mut() {
        match value {
            Value::Object(nested) => strip_cache_control(nested, depth + 1),
            Value::Array(items) => items
                .iter_mut()
                .filter_map(Value::as_object_mut)
                .for_each(|nested| strip_cache_control(nested, depth + 1)),
            _ => {}
        }
    }
}

fn decode_tool_parameters(index: usize, tool: &mut Value) -> Result<(), Error> {
    let Some(parameters) = tool
        .as_object_mut()
        .and_then(|tool| tool.get_mut("parameters"))
    else {
        return Ok(());
    };
    let decoded = match parameters {
        Value::Null | Value::Object(_) => return Ok(()),
        Value::String(encoded) => serde_json::from_str::<Value>(encoded)
            .ok()
            .filter(Value::is_object),
        _ => None,
    };
    let Some(decoded) = decoded else {
        return Err(Error::InvalidRequest(ErrorDetail::InvalidToolParameters {
            index,
            actual: python_type_name(parameters),
        }));
    };
    *parameters = decoded;
    Ok(())
}

fn python_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(number) if number.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// Reasoning LiteLLM minted while bridging another provider, which OpenAI cannot decrypt
fn is_minted_reasoning(item: &Value) -> bool {
    let Some(item) = item.as_object() else {
        return false;
    };
    item.get("type").and_then(Value::as_str) == Some("reasoning")
        && (item
            .get("id")
            .and_then(Value::as_str)
            .is_some_and(is_minted_reasoning_item_id)
            || item
                .get("encrypted_content")
                .and_then(Value::as_str)
                .is_some_and(carries_thinking_blocks))
}

fn is_minted_reasoning_item_id(id: &str) -> bool {
    id.strip_prefix(REASONING_ITEM_ID_PREFIX)
        .is_some_and(|suffix| {
            Uuid::try_parse(suffix).is_ok_and(|parsed| {
                parsed.get_version_num() == 4 && parsed.hyphenated().to_string() == suffix
            })
        })
}

fn carries_thinking_blocks(encrypted_content: &str) -> bool {
    let Ok(Value::Array(blocks)) = serde_json::from_str::<Value>(encrypted_content) else {
        return false;
    };
    blocks.iter().filter_map(Value::as_object).any(|block| {
        let truthy = |field: &str| {
            block.get(field).is_some_and(|value| {
                !matches!(value, Value::Null | Value::Bool(false)) && value != ""
            })
        };
        match block.get("type").and_then(Value::as_str) {
            Some("thinking") => truthy("signature"),
            Some("redacted_thinking") => truthy("data"),
            _ => false,
        }
    })
}

/// OpenAI rejects `status: null` on a replayed reasoning item. Python round-trips a valid item
/// through the SDK model with `exclude_none`, and otherwise drops only the nullable fields
fn without_null_reasoning_fields(item: Value, host: ResponsesHost) -> Value {
    let Value::Object(mut fields) = item else {
        return item;
    };
    match fields.get("type").and_then(Value::as_str) {
        Some("message") if host == ResponsesHost::Azure => {
            fields.remove("status");
            return Value::Object(fields);
        }
        Some("reasoning") => {}
        _ => return Value::Object(fields),
    }
    if is_sdk_reasoning_item(&fields) {
        drop_nulls(&mut fields);
        if host == ResponsesHost::Azure {
            fields.remove("status");
        }
        fields
            .iter_mut()
            .filter(|(name, _)| matches!(name.as_str(), "summary" | "content"))
            .filter_map(|(_, parts)| parts.as_array_mut())
            .flatten()
            .filter_map(Value::as_object_mut)
            .for_each(drop_nulls);
    } else {
        fields.retain(|name, value| {
            !(value.is_null() && REASONING_FIELDS_DROPPED_WHEN_NULL.contains(&name.as_str()))
        });
    }
    Value::Object(fields)
}

/// Whether `ResponseReasoningItem.model_validate` accepts the item
fn is_sdk_reasoning_item(fields: &Map<String, Value>) -> bool {
    let parts = |name: &str, part_type: &str, required: bool| match fields.get(name) {
        None | Some(Value::Null) => !required,
        Some(Value::Array(parts)) => parts.iter().all(|part| {
            part.get("type").and_then(Value::as_str) == Some(part_type)
                && part.get("text").is_some_and(Value::is_string)
        }),
        Some(_) => false,
    };
    fields.get("id").is_some_and(Value::is_string)
        && parts("summary", "summary_text", true)
        && parts("content", "reasoning_text", false)
        && fields
            .get("encrypted_content")
            .is_none_or(|value| value.is_null() || value.is_string())
        && fields.get("status").is_none_or(|status| {
            status.is_null()
                || matches!(
                    status.as_str(),
                    Some("in_progress" | "completed" | "incomplete")
                )
        })
}

fn drop_nulls(fields: &mut Map<String, Value>) {
    fields.retain(|_, value| !value.is_null());
}

fn without_foreign_tool_call_id(item: Value) -> Value {
    let Value::Object(mut fields) = item else {
        return item;
    };
    let prefix = match fields.get("type").and_then(Value::as_str) {
        Some("function_call") => "fc",
        Some("custom_tool_call") => "ctc",
        _ => return Value::Object(fields),
    };
    if fields
        .get("id")
        .and_then(Value::as_str)
        .is_some_and(|id| !id.starts_with(prefix))
    {
        fields.remove("id");
    }
    Value::Object(fields)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn shaped(input: Value, params: Value, host: ResponsesHost) -> (Value, Value) {
        let Value::Object(params) = params else {
            panic!("params must be an object")
        };
        let (input, params) = shape(input, params, host).unwrap();
        (input, Value::Object(params))
    }

    #[rstest]
    #[case::warm_up_probe(json!(1), json!(16))]
    #[case::just_below(json!(15), json!(16))]
    #[case::minimum(json!(16), json!(16))]
    #[case::zero_left_for_the_provider(json!(0), json!(0))]
    #[case::negative_left_for_the_provider(json!(-1), json!(-1))]
    #[case::null(Value::Null, Value::Null)]
    fn raises_max_output_tokens_to_the_minimum(#[case] requested: Value, #[case] sent: Value) {
        let (_, params) = shaped(
            json!("hi"),
            json!({"max_output_tokens": requested}),
            ResponsesHost::Compatible,
        );
        assert_eq!(params, json!({"max_output_tokens": sent}));
    }

    #[test]
    fn strips_cache_control_at_any_depth() {
        let mark = json!({"type": "ephemeral"});
        let (input, params) = shaped(
            json!([
                {"role": "user", "content": "hi", "cache_control": mark},
                {"role": "user", "content": [{"type": "input_text", "text": "a", "cache_control": mark}]},
                {"type": "function_call_output", "call_id": "c", "output": [{"type": "input_text", "text": "b", "cache_control": mark}]}
            ]),
            json!({"tools": [{"type": "function", "name": "a", "cache_control": mark, "parameters": {"type": "object"}}]}),
            ResponsesHost::Compatible,
        );
        assert_eq!(
            (input, params),
            (
                json!([
                    {"role": "user", "content": "hi"},
                    {"role": "user", "content": [{"type": "input_text", "text": "a"}]},
                    {"type": "function_call_output", "call_id": "c", "output": [{"type": "input_text", "text": "b"}]}
                ]),
                json!({"tools": [{"type": "function", "name": "a", "parameters": {"type": "object"}}]})
            )
        );
    }

    #[test]
    fn decodes_json_encoded_tool_parameters() {
        let (_, params) = shaped(
            json!("hi"),
            json!({"tools": [
                {"type": "function", "name": "a", "parameters": "{\"type\":\"object\"}"},
                {"type": "function", "name": "b", "parameters": {"type": "object"}},
                {"type": "function", "name": "c", "parameters": null},
                {"type": "web_search"}
            ]}),
            ResponsesHost::Compatible,
        );
        assert_eq!(
            params,
            json!({"tools": [
                {"type": "function", "name": "a", "parameters": {"type": "object"}},
                {"type": "function", "name": "b", "parameters": {"type": "object"}},
                {"type": "function", "name": "c", "parameters": null},
                {"type": "web_search"}
            ]})
        );
    }

    #[rstest]
    #[case::not_json(json!("{"), "str")]
    #[case::json_array(json!("[]"), "str")]
    #[case::number(json!(3), "int")]
    #[case::list(json!([]), "list")]
    fn rejects_tool_parameters_that_are_not_objects(
        #[case] parameters: Value,
        #[case] actual: &str,
    ) {
        let Value::Object(params) = json!({"tools": [{"type": "web_search"}, {"type": "function", "parameters": parameters}]})
        else {
            unreachable!()
        };
        assert_eq!(
            shape(json!("hi"), params, ResponsesHost::Compatible)
                .unwrap_err()
                .to_string(),
            format!(
                "invalid request: Invalid type for 'tools[1].parameters': expected an object, but got {actual} instead."
            )
        );
    }

    #[rstest]
    #[case::sdk_shaped(
        json!({"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "x"}], "status": null, "encrypted_content": null, "extra": null}),
        json!({"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "x"}]})
    )]
    #[case::missing_id_keeps_unknown_nulls(
        json!({"type": "reasoning", "summary": [], "status": null, "extra": null}),
        json!({"type": "reasoning", "summary": [], "extra": null})
    )]
    #[case::missing_summary(
        json!({"type": "reasoning", "id": "rs_1", "content": null}),
        json!({"type": "reasoning", "id": "rs_1"})
    )]
    #[case::not_reasoning(
        json!({"type": "message", "role": "user", "content": "hi", "status": null}),
        json!({"type": "message", "role": "user", "content": "hi", "status": null})
    )]
    fn drops_null_reasoning_fields(#[case] item: Value, #[case] expected: Value) {
        let (input, _) = shaped(json!([item]), json!({}), ResponsesHost::Compatible);
        assert_eq!(input, json!([expected]));
    }

    #[rstest]
    #[case::minted_id(json!({"type": "reasoning", "id": "rs_6f9619ff-8b86-4d01-b42d-00c04fc964ff", "summary": []}), true)]
    #[case::bridged_thinking(
        json!({"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "[{\"type\":\"thinking\",\"thinking\":\"x\",\"signature\":\"sig\"}]"}),
        true
    )]
    #[case::unsigned_thinking(
        json!({"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "[{\"type\":\"thinking\",\"thinking\":\"x\"}]"}),
        false
    )]
    #[case::openai_id(json!({"type": "reasoning", "id": "rs_0a1b2c3d4e5f", "summary": [], "encrypted_content": "gAAAA"}), false)]
    #[case::uppercase_uuid(json!({"type": "reasoning", "id": "rs_6F9619FF-8B86-4D01-B42D-00C04FC964FF", "summary": []}), false)]
    #[case::uuid_v1(json!({"type": "reasoning", "id": "rs_6f9619ff-8b86-11d0-b42d-00c04fc964ff", "summary": []}), false)]
    fn drops_minted_reasoning_for_openai(#[case] item: Value, #[case] minted: bool) {
        let input = json!([item, {"role": "user", "content": "hi"}]);
        let (openai, _) = shaped(input.clone(), json!({}), ResponsesHost::OpenAi);
        let (lenient, _) = shaped(input.clone(), json!({}), ResponsesHost::Compatible);
        assert_eq!(openai.as_array().unwrap().len(), if minted { 1 } else { 2 });
        assert_eq!(lenient, input);
    }

    #[rstest]
    #[case::foreign_function_call(json!({"type": "function_call", "id": "toolu_1", "call_id": "c"}), json!({"type": "function_call", "call_id": "c"}))]
    #[case::openai_function_call(json!({"type": "function_call", "id": "fc_1", "call_id": "c"}), json!({"type": "function_call", "id": "fc_1", "call_id": "c"}))]
    #[case::foreign_custom_tool_call(json!({"type": "custom_tool_call", "id": "x", "call_id": "c"}), json!({"type": "custom_tool_call", "call_id": "c"}))]
    #[case::other_item(json!({"type": "message", "id": "x", "role": "user", "content": "hi"}), json!({"type": "message", "id": "x", "role": "user", "content": "hi"}))]
    fn drops_foreign_tool_call_ids_for_openai(#[case] item: Value, #[case] expected: Value) {
        let (openai, _) = shaped(json!([item.clone()]), json!({}), ResponsesHost::OpenAi);
        let (lenient, _) = shaped(json!([item.clone()]), json!({}), ResponsesHost::Compatible);
        assert_eq!((openai, lenient), (json!([expected]), json!([item])));
    }
}
