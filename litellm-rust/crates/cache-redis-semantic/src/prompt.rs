use litellm_cache::SemanticCacheContext;
use serde_json::Value;

pub fn prompt_from_context(context: &SemanticCacheContext) -> Option<String> {
    if let Some(messages) = context.messages.as_ref().and_then(Value::as_array)
        && !messages.is_empty()
    {
        return Some(messages_text(messages));
    }
    let input = context.input.as_ref()?;
    let mut parts = Vec::new();
    collect_input_text(input, &mut parts);
    let prompt = parts.join("\n").trim().to_string();
    (!prompt.is_empty()).then_some(prompt)
}

fn messages_text(messages: &[Value]) -> String {
    let mut text = String::new();
    for message in messages {
        let Some(message) = message.as_object() else {
            continue;
        };
        match message.get("content") {
            Some(Value::String(content)) => text.push_str(content),
            Some(Value::Array(parts)) => {
                for part in parts {
                    if let Some(text_content) = part.get("text").and_then(Value::as_str) {
                        text.push_str(text_content);
                    }
                }
            }
            _ => {}
        }
        text.push_str(&search_results_text(message.get("search_results")));
    }
    text
}

fn search_results_text(search_results: Option<&Value>) -> String {
    let Some(Value::Array(results)) = search_results else {
        return String::new();
    };
    let mut text = String::new();
    for result in results {
        let Some(result) = result.as_object() else {
            continue;
        };
        for key in ["source", "title"] {
            if let Some(value) = result.get(key).and_then(Value::as_str) {
                text.push_str(value);
            }
        }
        if let Some(Value::Array(content)) = result.get("content") {
            for block in content {
                if let Some(value) = block.get("text").and_then(Value::as_str) {
                    text.push_str(value);
                }
            }
        }
        if let Some(citations) = result.get("citations") {
            text.push_str(&citations.to_string());
        }
    }
    text
}

fn collect_input_text(value: &Value, parts: &mut Vec<String>) {
    match value {
        Value::String(text) => {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                parts.push(trimmed.to_string());
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_input_text(item, parts);
            }
        }
        Value::Object(map) => {
            if let Some(content) = map.get("content").filter(|content| !content.is_null()) {
                collect_input_text(content, parts);
                return;
            }
            for key in ["text", "output", "input_text", "output_text"] {
                if let Some(Value::String(text)) = map.get(key) {
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        parts.push(trimmed.to_string());
                        return;
                    }
                }
            }
        }
        _ => {}
    }
}
