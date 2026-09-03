use crate::payload::payload_from_py;
use litellm_core::http_utils::body::{JsonPayload, SharedText};
use litellm_core::messages::types::{
    AnthropicMessage, AnthropicMessagesRequest, CacheControl, ContentBlock, MessageContent,
    SystemPrompt,
};
use litellm_python_interop::{from_py, text_bytes_from_py};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use serde::de::DeserializeOwned;
use std::collections::BTreeMap;

fn invalid(message: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("invalid Anthropic messages request: {message}"))
}

fn required<'py>(dict: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(name)?
        .ok_or_else(|| invalid(format!("missing field `{name}`")))
}

fn optional<T: DeserializeOwned>(dict: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<T>> {
    dict.get_item(name)?
        .filter(|value| !value.is_none())
        .map(|value| from_py(&value))
        .transpose()
}

fn finite_number(dict: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<f64>> {
    optional::<f64>(dict, name).map(|number| number.filter(|value| value.is_finite()))
}

fn shared(dict: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<JsonPayload>> {
    dict.get_item(name)?
        .filter(|value| !value.is_none())
        .map(|value| payload_from_py(&value))
        .transpose()
}

fn shared_array(dict: &Bound<'_, PyDict>, name: &str) -> PyResult<Option<Vec<JsonPayload>>> {
    dict.get_item(name)?
        .filter(|value| !value.is_none())
        .map(|value| sequence(&value, payload_from_py))
        .transpose()
}

fn extras(dict: &Bound<'_, PyDict>, known: &[&str]) -> PyResult<BTreeMap<String, JsonPayload>> {
    dict.iter()
        .filter_map(|(key, value)| match key.extract::<String>() {
            Ok(key) if known.contains(&key.as_str()) => None,
            Ok(key) => Some(payload_from_py(&value).map(|value| (key, value))),
            Err(error) => Some(Err(error)),
        })
        .collect()
}

fn sequence<T>(
    value: &Bound<'_, PyAny>,
    extract: impl Fn(&Bound<'_, PyAny>) -> PyResult<T>,
) -> PyResult<Vec<T>> {
    if !value.is_instance_of::<PyList>() && !value.is_instance_of::<PyTuple>() {
        return Err(invalid("expected an array"));
    }
    value.try_iter()?.map(|value| extract(&value?)).collect()
}

fn content(value: &Bound<'_, PyAny>) -> PyResult<MessageContent> {
    if value.is_instance_of::<PyString>() {
        return SharedText::new(text_bytes_from_py(value)?)
            .map(MessageContent::Text)
            .map_err(invalid);
    }
    sequence(value, block).map(MessageContent::Blocks)
}

fn block(value: &Bound<'_, PyAny>) -> PyResult<ContentBlock> {
    let dict = value
        .cast::<PyDict>()
        .map_err(|_| invalid("content block must be an object"))?;
    let cache_control = dict
        .get_item("cache_control")?
        .filter(|value| !value.is_none())
        .map(|value| {
            let cache = value
                .cast::<PyDict>()
                .map_err(|_| invalid("cache_control must be an object"))?;
            Ok::<_, PyErr>(CacheControl {
                cache_type: optional(cache, "type")?,
                ttl: optional(cache, "ttl")?,
                scope: optional(cache, "scope")?,
                extra: extras(cache, &["type", "ttl", "scope"])?,
            })
        })
        .transpose()?;
    Ok(ContentBlock {
        cache_control,
        extra: extras(dict, &["cache_control"])?,
    })
}

fn message(value: &Bound<'_, PyAny>) -> PyResult<AnthropicMessage> {
    let dict = value
        .cast::<PyDict>()
        .map_err(|_| invalid("message must be an object"))?;
    Ok(AnthropicMessage {
        role: from_py(&required(dict, "role")?)?,
        content: content(&required(dict, "content")?)?,
        extra: extras(dict, &["role", "content"])?,
    })
}

pub(crate) fn messages_from_py(value: &Bound<'_, PyAny>) -> PyResult<AnthropicMessagesRequest> {
    let dict = value
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("body must be a dict"))?;
    let system = dict
        .get_item("system")?
        .filter(|value| !value.is_none())
        .map(|value| {
            content(&value).map(|value| match value {
                MessageContent::Text(text) => SystemPrompt::Text(text),
                MessageContent::Blocks(blocks) => SystemPrompt::Blocks(blocks),
            })
        })
        .transpose()?;
    Ok(AnthropicMessagesRequest {
        model: from_py(&required(dict, "model")?)?,
        messages: sequence(&required(dict, "messages")?, message)?,
        system,
        max_tokens: optional(dict, "max_tokens")?,
        stop_sequences: optional(dict, "stop_sequences")?,
        stream: optional(dict, "stream")?,
        temperature: finite_number(dict, "temperature")?,
        top_p: finite_number(dict, "top_p")?,
        top_k: optional(dict, "top_k")?,
        service_tier: optional(dict, "service_tier")?,
        speed: optional(dict, "speed")?,
        inference_geo: optional(dict, "inference_geo")?,
        metadata: shared(dict, "metadata")?,
        tool_choice: shared(dict, "tool_choice")?,
        thinking: shared(dict, "thinking")?,
        container: shared(dict, "container")?,
        context_management: shared(dict, "context_management")?,
        output_format: shared(dict, "output_format")?,
        output_config: shared(dict, "output_config")?,
        tools: shared_array(dict, "tools")?,
        mcp_servers: shared_array(dict, "mcp_servers")?,
        extra: extras(
            dict,
            &[
                "model",
                "messages",
                "system",
                "max_tokens",
                "stop_sequences",
                "stream",
                "temperature",
                "top_p",
                "top_k",
                "service_tier",
                "speed",
                "inference_geo",
                "metadata",
                "tool_choice",
                "thinking",
                "container",
                "context_management",
                "output_format",
                "output_config",
                "tools",
                "mcp_servers",
            ],
        )?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use litellm_core::messages::transformation::AnthropicMessagesProviderConfig;
    use litellm_core::providers::azure_ai::messages::transformation::AZURE_ANTHROPIC_MESSAGES_CONFIG;

    #[test]
    fn typed_extraction_retains_nested_media_and_extension_owners() {
        Python::initialize();
        let (request, pointer) = Python::attach(|py| {
            let globals = PyDict::new(py);
            py.run(c"media = 'A' * (1024 * 1024)\nbody = {'model': 'model', 'messages': [{'role': 'system', 'content': [{'type': 'tool_result', 'content': [{'type': 'image', 'source': {'type': 'base64', 'data': media}}]}]}], 'extension': {'data': media}}", Some(&globals), None).unwrap();
            let media = globals.get_item("media").unwrap().unwrap();
            let pointer = media.cast::<PyString>().unwrap().to_str().unwrap().as_ptr() as usize;
            let body = globals.get_item("body").unwrap().unwrap();
            (messages_from_py(&body).unwrap(), pointer)
        });
        let request = std::thread::spawn(move || {
            AZURE_ANTHROPIC_MESSAGES_CONFIG
                .transform_request(request)
                .unwrap()
        })
        .join()
        .unwrap();
        let SystemPrompt::Blocks(blocks) = request.system.unwrap() else {
            panic!("blocks")
        };
        assert_eq!(
            blocks[0].extra["content"][0]["source"]["data"]
                .as_text()
                .unwrap()
                .bytes()
                .as_ptr() as usize,
            pointer
        );
        assert_eq!(
            request.extra["extension"]["data"]
                .as_text()
                .unwrap()
                .bytes()
                .as_ptr() as usize,
            pointer
        );
    }

    #[test]
    fn direct_extraction_matches_serde_for_known_fields_and_extensions() {
        Python::initialize();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            py.run(c"body = {'model': 'model', 'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'héllo', 'cache_control': {'type': 'ephemeral', 'scope': 'global', 'unknown': 4}}], 'extension': [None, True]}], 'system': 'system', 'max_tokens': 3, 'tools': [{'name': 'tool'}], 'metadata': {'a': 1}, 'thinking': None, 'temperature': float('nan'), 'top_p': float('inf'), 'unknown': {'data': 'AAAA'}}", Some(&globals), None).unwrap();
            let value = globals.get_item("body").unwrap().unwrap();
            let direct = messages_from_py(&value).unwrap();
            let reference: AnthropicMessagesRequest =
                serde_json::from_value(from_py::<serde_json::Value>(&value).unwrap()).unwrap();
            assert_eq!(direct, reference);
        });
    }
}
