//! Tools a client sends as `additional_tools` input items, matching
//! `litellm/responses/additional_tools.py`

use serde_json::Value;

pub struct HoistedAdditionalTools {
    pub input: Value,
    pub hoisted: Vec<Value>,
}

/// The input without its `additional_tools` items, and the tools those items carried. An item
/// whose `tools` is not a list of objects carries none
pub fn hoist_additional_tools(input: Value) -> HoistedAdditionalTools {
    let Value::Array(items) = input else {
        return HoistedAdditionalTools {
            input,
            hoisted: Vec::new(),
        };
    };
    let (additional, remaining): (Vec<Value>, Vec<Value>) = items
        .into_iter()
        .partition(|item| item.get("type").and_then(Value::as_str) == Some("additional_tools"));
    HoistedAdditionalTools {
        input: Value::Array(remaining),
        hoisted: additional.into_iter().flat_map(item_tools).collect(),
    }
}

fn item_tools(item: Value) -> Vec<Value> {
    match item.get("tools") {
        None => Vec::new(),
        Some(Value::Array(tools)) if tools.iter().all(Value::is_object) => tools.clone(),
        Some(_) => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::tools_are_hoisted_in_order(
        json!([
            {"type": "additional_tools", "tools": [{"type": "function", "name": "a"}]},
            {"role": "user", "content": "hi"},
            {"type": "additional_tools", "tools": [{"type": "mcp", "server_label": "b"}]},
        ]),
        json!([{"role": "user", "content": "hi"}]),
        vec![json!({"type": "function", "name": "a"}), json!({"type": "mcp", "server_label": "b"})]
    )]
    #[case::malformed_tools_carry_nothing_but_the_item_still_leaves(
        json!([
            {"type": "additional_tools", "tools": [{"type": "function"}, "bad"]},
            {"type": "additional_tools", "tools": null},
            {"type": "additional_tools"},
        ]),
        json!([]),
        vec![]
    )]
    #[case::string_input(json!("hi"), json!("hi"), vec![])]
    fn additional_tools_items_are_hoisted(
        #[case] input: Value,
        #[case] remaining: Value,
        #[case] hoisted: Vec<Value>,
    ) {
        let result = hoist_additional_tools(input);
        assert_eq!((result.input, result.hoisted), (remaining, hoisted));
    }
}
