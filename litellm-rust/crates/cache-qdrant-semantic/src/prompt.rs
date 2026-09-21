use serde_json::Value;

fn search_results_text(search_results: Option<&Value>) -> String {
    let Some(Value::Array(results)) = search_results else {
        return String::new();
    };
    results
        .iter()
        .filter_map(Value::as_object)
        .flat_map(|result| {
            let source = result
                .get("source")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let title = result
                .get("title")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let content = result
                .get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_object)
                .filter_map(|block| block.get("text").and_then(Value::as_str).map(str::to_owned));
            let citations = result
                .get("citations")
                .filter(|value| !value.is_null())
                .map(|value| serde_json::to_string(value).unwrap_or_default());
            source
                .into_iter()
                .chain(title)
                .chain(content)
                .chain(citations)
        })
        .collect()
}

pub fn prompt_from_messages(messages: &[Value]) -> String {
    messages
        .iter()
        .filter_map(Value::as_object)
        .map(|message| {
            let content = match message.get("content") {
                Some(Value::String(content)) => content.clone(),
                Some(Value::Array(parts)) => parts
                    .iter()
                    .filter_map(Value::as_object)
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .collect(),
                _ => String::new(),
            };
            format!(
                "{content}{}",
                search_results_text(message.get("search_results"))
            )
        })
        .collect()
}
