use litellm_llms_types::formats::messages::CacheControl;
use serde::{Deserialize, Serialize};

/// One entry of a deployment's `cache_control_injection_points`: where LiteLLM puts a
/// `cache_control` breakpoint the caller did not send. Python's `CacheControlInjectionPoint`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "location")]
pub enum CacheControlInjectionPoint {
    /// A breakpoint on the system prompt (`role: system`), on the message at `index`, counted
    /// from the end when negative, or on every message of `role`.
    #[serde(rename = "message")]
    Message {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        role: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        index: Option<i64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        control: Option<CacheControl>,
    },
    /// A breakpoint on the last tool the provider caches.
    #[serde(rename = "tool_config")]
    ToolConfig {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        control: Option<CacheControl>,
    },
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use litellm_llms_types::recognized::Recognized;

    use super::CacheControlInjectionPoint;

    #[rstest]
    #[case::system(json!({"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "1h"}}))]
    #[case::last_message(json!({"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}))]
    #[case::no_control(json!({"location": "message", "role": "user"}))]
    #[case::tool_config(json!({"location": "tool_config"}))]
    fn a_point_python_reads_is_known(#[case] written: Value) {
        let point: Recognized<CacheControlInjectionPoint> =
            serde_json::from_value(written.clone()).unwrap();

        assert!(point.known().is_some());
        assert_eq!(serde_json::to_value(point).unwrap(), written);
    }

    #[rstest]
    #[case::an_index_written_as_text(json!({"location": "message", "index": "-1"}))]
    #[case::a_control_that_is_no_object(json!({"location": "message", "role": "system", "control": "ephemeral"}))]
    #[case::another_location(json!({"location": "tools"}))]
    #[case::no_location(json!({"role": "system"}))]
    #[case::no_object(json!("message"))]
    fn any_other_entry_is_kept_as_written(#[case] written: Value) {
        let point: Recognized<CacheControlInjectionPoint> =
            serde_json::from_value(written.clone()).unwrap();

        assert_eq!(point, Recognized::Unrecognized(written));
    }
}
