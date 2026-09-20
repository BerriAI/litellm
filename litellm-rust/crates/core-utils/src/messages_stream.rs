use serde_json::{Map, Value};

pub struct MessagesStream {
    pub usage: Value,
    pub cached_response: Option<Value>,
}

impl MessagesStream {
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        let text = std::str::from_utf8(bytes).ok()?;
        let mut usage = Map::new();
        let mut stopped = false;
        let mut failed = false;
        let events: Vec<_> = text.split_inclusive("\n\n").collect();
        for event in &events {
            for line in event.lines().filter_map(|line| line.strip_prefix("data:")) {
                let Ok(value) = serde_json::from_str::<Value>(line.trim()) else {
                    continue;
                };
                match value.get("type").and_then(Value::as_str) {
                    Some("message_stop") => stopped = true,
                    Some("error") => failed = true,
                    _ => {}
                }
                if let Some(fields) = value
                    .pointer("/message/usage")
                    .or_else(|| value.get("usage"))
                    .and_then(Value::as_object)
                {
                    usage.extend(fields.clone());
                }
            }
        }
        Some(Self {
            usage: Value::Object(usage),
            cached_response: (stopped && !failed)
                .then(|| serde_json::json!({"litellm_cached_anthropic_sse_events": events})),
        })
    }
}
