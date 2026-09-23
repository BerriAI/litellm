use serde_json::Value;

pub fn get_transcription_model_name_from_results(results: &[Value]) -> Option<&str> {
    results.iter().find_map(|event| {
        let event_type = event.get("type")?.as_str()?;
        if !matches!(
            event_type,
            "transcription_session.created"
                | "transcription_session.updated"
                | "session.created"
                | "session.updated"
        ) {
            return None;
        }
        let session = event.get("session")?;
        let nested_transcription = session.pointer("/audio/input/transcription");
        let transcription = nested_transcription
            .filter(|value| value.as_object().is_some_and(|object| !object.is_empty()))
            .or_else(|| session.get("input_audio_transcription"));
        [
            transcription.and_then(|value| value.get("model")),
            session.get("model"),
        ]
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .find(|model| !model.is_empty())
    })
}

pub fn transcription_usage_cost(usage: &Value, model_info: Option<&Value>) -> f64 {
    let Some(model_info) = model_info else {
        return 0.0;
    };
    let number = |value: Option<&Value>| value.and_then(Value::as_f64).unwrap_or(0.0);
    match usage.get("type").and_then(Value::as_str) {
        Some("duration") => {
            number(usage.get("seconds")) * number(model_info.get("input_cost_per_second"))
        }
        Some("tokens") => {
            let audio_tokens = number(usage.pointer("/input_token_details/audio_tokens"));
            let text_tokens = number(usage.pointer("/input_token_details/text_tokens"));
            let output_tokens = number(usage.get("output_tokens"));
            let input_rate = number(model_info.get("input_cost_per_token"));
            let audio_rate = model_info
                .get("input_cost_per_audio_token")
                .and_then(Value::as_f64)
                .filter(|rate| *rate != 0.0)
                .unwrap_or(input_rate);
            let output_rate = number(model_info.get("output_cost_per_token"));
            audio_tokens * audio_rate + text_tokens * input_rate + output_tokens * output_rate
        }
        _ => 0.0,
    }
}
