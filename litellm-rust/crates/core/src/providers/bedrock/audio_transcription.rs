use serde_json::{Map, Value, json};

use crate::audio_transcription::transformation::{
    AudioTranscriptionAuth, AudioTranscriptionProviderConfig,
};
use crate::audio_transcription::types::{
    AudioTranscriptionRequestData, AudioTranscriptionResponseData,
};
use crate::error::{CoreError, CoreResult, json_type_name};

use super::common_utils::{bedrock_model_id_and_region, resolve_bedrock_region};
use super::constants::{BEDROCK_RUNTIME_ENDPOINT_TEMPLATE, BEDROCK_SERVICE};

const SUPPORTED_PARAMS: &[&str] = &["language", "prompt", "temperature", "response_format"];

pub static BEDROCK_AUDIO_TRANSCRIPTION_CONFIG: BedrockAudioTranscriptionConfig =
    BedrockAudioTranscriptionConfig;

pub struct BedrockAudioTranscriptionConfig;

fn audio_fields(audio: Value) -> CoreResult<(String, String)> {
    let object = audio.as_object().ok_or_else(|| CoreError::InvalidType {
        expected: "object",
        actual: json_type_name(&audio),
    })?;
    let data = object
        .get("data")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or(CoreError::MissingField("audio.data"))?;
    let format = object
        .get("format")
        .and_then(Value::as_str)
        .filter(|value| matches!(*value, "wav" | "mp3" | "flac" | "ogg"))
        .ok_or_else(|| {
            CoreError::InvalidRequest("audio.format must be wav, mp3, flac, or ogg".to_string())
        })?;
    Ok((data.to_string(), format.to_string()))
}

fn optional_string<'a>(params: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    params
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
}

impl AudioTranscriptionProviderConfig for BedrockAudioTranscriptionConfig {
    fn supported_transcription_params(&self) -> &'static [&'static str] {
        SUPPORTED_PARAMS
    }

    fn transform_transcription_request(
        &self,
        _model: &str,
        audio: Value,
        optional_params: Map<String, Value>,
    ) -> CoreResult<AudioTranscriptionRequestData> {
        let (data, format) = audio_fields(audio)?;
        let mut instruction = "Transcribe the audio. Respond with only the transcript.".to_string();
        if let Some(language) = optional_string(&optional_params, "language") {
            instruction.push_str(&format!(" The audio language is {language}."));
        }
        if let Some(prompt) = optional_string(&optional_params, "prompt") {
            instruction.push_str(&format!(" Additional context: {prompt}"));
        }
        let mut inference_config = Map::from_iter([("maxTokens".to_string(), json!(4096))]);
        if let Some(temperature) = optional_params.get("temperature") {
            inference_config.insert("temperature".to_string(), temperature.clone());
        }
        Ok(AudioTranscriptionRequestData {
            body: json!({
                "messages": [{
                    "role": "user",
                    "content": [
                        {"audio": {"format": format, "source": {"bytes": data}}},
                        {"text": instruction}
                    ]
                }],
                "system": [{"text": "You are a transcription assistant."}],
                "inferenceConfig": inference_config,
            }),
        })
    }

    fn transform_transcription_response(
        &self,
        _model: &str,
        response_json: Value,
    ) -> CoreResult<AudioTranscriptionResponseData> {
        let content = response_json
            .get("output")
            .and_then(|value| value.get("message"))
            .and_then(|value| value.get("content"))
            .and_then(Value::as_array)
            .ok_or_else(|| {
                CoreError::InvalidResponse("Bedrock response has no output content".to_string())
            })?;
        let mut text = String::new();
        for block in content {
            if let Some(value) = block.get("text").and_then(Value::as_str) {
                text.push_str(value);
            }
        }
        Ok(AudioTranscriptionResponseData { text })
    }

    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        let (model_id, model_region) = bedrock_model_id_and_region(model);
        let region = resolve_bedrock_region(model_region.as_deref(), optional_params, env_lookup);
        let endpoint = optional_params
            .get("aws_bedrock_runtime_endpoint")
            .and_then(Value::as_str)
            .or(api_base)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| BEDROCK_RUNTIME_ENDPOINT_TEMPLATE.replace("{region}", &region));
        Ok(format!(
            "{}/model/{model_id}/converse",
            endpoint.trim_end_matches('/')
        ))
    }

    fn auth_strategy(
        &self,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<AudioTranscriptionAuth> {
        let (_, model_region) = bedrock_model_id_and_region(model);
        Ok(AudioTranscriptionAuth::AwsSigV4 {
            region: resolve_bedrock_region(model_region.as_deref(), optional_params, env_lookup),
            service: BEDROCK_SERVICE,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn request_matches_python_shape() {
        let params = Map::from_iter([
            ("language".to_string(), json!("en")),
            ("prompt".to_string(), json!("Speaker names")),
            ("temperature".to_string(), json!(0)),
            ("timestamp_granularities".to_string(), json!(["word"])),
        ]);
        let params = BEDROCK_AUDIO_TRANSCRIPTION_CONFIG.map_transcription_params(&params);
        let result = BEDROCK_AUDIO_TRANSCRIPTION_CONFIG
            .transform_transcription_request(
                "mistral.voxtral-mini-3b-2507",
                json!({"data": "AQI=", "format": "wav", "filename": "sample.wav"}),
                params,
            )
            .expect("request");
        assert_eq!(
            result.body,
            json!({
                "messages": [{
                    "role": "user",
                    "content": [
                        {"audio": {"format": "wav", "source": {"bytes": "AQI="}}},
                        {"text": "Transcribe the audio. Respond with only the transcript. The audio language is en. Additional context: Speaker names"}
                    ]
                }],
                "system": [{"text": "You are a transcription assistant."}],
                "inferenceConfig": {"maxTokens": 4096, "temperature": 0}
            })
        );
    }

    #[test]
    fn response_concatenates_content_blocks() {
        let result = BEDROCK_AUDIO_TRANSCRIPTION_CONFIG
            .transform_transcription_response(
                "model",
                json!({"output": {"message": {"content": [{"text": "hello "}, {"text": "world"}]}}}),
            )
            .expect("response");
        assert_eq!(result.text, "hello world");
        assert_eq!(result.into_json(), json!({"text": "hello world"}));
    }

    #[test]
    fn invalid_audio_is_rejected() {
        let result = BEDROCK_AUDIO_TRANSCRIPTION_CONFIG.transform_transcription_request(
            "model",
            json!({"data": "AQI="}),
            Map::new(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn region_and_url_precedence_match_python() {
        let params = Map::from_iter([("aws_region_name".to_string(), json!("eu-west-1"))]);
        let url = BEDROCK_AUDIO_TRANSCRIPTION_CONFIG
            .complete_url(
                None,
                "bedrock/us-east-1/mistral.voxtral-mini-3b-2507",
                &params,
                &no_env,
            )
            .expect("url");
        assert_eq!(
            url,
            "https://bedrock-runtime.eu-west-1.amazonaws.com/model/mistral.voxtral-mini-3b-2507/converse"
        );
    }
}
