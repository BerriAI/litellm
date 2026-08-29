#![no_main]

use libfuzzer_sys::fuzz_target;
use serde_json::Value;

fuzz_target!(|data: &[u8]| {
    if let Ok(body) = serde_json::from_slice::<Value>(data) {
        // Fuzz chat completions request parsing
        // Validate that parsing doesn't panic on arbitrary input
        let _model = body.get("model").and_then(Value::as_str);
        let _messages = body.get("messages").and_then(Value::as_array);
        let _stream = body.get("stream").and_then(Value::as_bool);
        let _temperature = body.get("temperature").and_then(Value::as_f64);
        let _top_p = body.get("top_p").and_then(Value::as_f64);
        let _max_tokens = body.get("max_tokens").and_then(Value::as_i64);

        // Ensure serialization round-trips don't panic
        let _ = serde_json::to_vec(&body);
        let _ = serde_json::to_string(&body);
    }
});
