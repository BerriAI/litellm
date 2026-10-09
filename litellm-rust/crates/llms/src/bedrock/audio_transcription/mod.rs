use litellm_auth_aws::{
    AwsCredentialSource, bedrock_model_id_and_region,
    constants::{BEDROCK_RUNTIME_ENDPOINT_TEMPLATE, BEDROCK_SERVICE},
    resolve_bedrock_region,
};
use litellm_core_utils::core_helpers::json_type_name;
use litellm_llms_types::formats::audio_transcription::AudioTranscriptionResponseData;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use strum::IntoStaticStr;

use crate::{
    Error,
    base_llm::{
        audio_transcription::transformation::{
            AudioTranscriptionRequestData, BaseAudioTranscriptionConfig, Headers,
            ValidatedEnvironment,
        },
        auth::AuthScheme,
    },
    bedrock::chat::converse_transformation::ConverseResponse,
};

const SUPPORTED_PARAMS: &[&str] = &["language", "prompt", "temperature", "response_format"];

pub static BEDROCK_AUDIO_TRANSCRIPTION_CONFIG: BedrockAudioTranscriptionConfig =
    BedrockAudioTranscriptionConfig;

pub struct BedrockAudioTranscriptionConfig;

#[derive(Clone, Copy, Deserialize, IntoStaticStr)]
#[serde(rename_all = "lowercase")]
enum AudioFormat {
    #[strum(serialize = "wav")]
    Wav,
    #[strum(serialize = "mp3")]
    Mp3,
    #[strum(serialize = "flac")]
    Flac,
    #[strum(serialize = "ogg")]
    Ogg,
}

struct AudioInput {
    data: String,
    format: AudioFormat,
}

fn audio_fields(audio: Value) -> Result<AudioInput, Error> {
    let object = audio.as_object().ok_or_else(|| Error::InvalidType {
        expected: "object",
        actual: json_type_name(&audio),
    })?;
    let data = object
        .get("data")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or(Error::MissingField("audio.data"))?;
    let format = object.get("format").cloned().ok_or_else(|| {
        Error::InvalidRequest(
            "audio.format must be wav, mp3, flac, or ogg"
                .to_string()
                .into(),
        )
    })?;
    let format: AudioFormat = serde_json::from_value(format).map_err(|_| {
        Error::InvalidRequest(
            "audio.format must be wav, mp3, flac, or ogg"
                .to_string()
                .into(),
        )
    })?;
    Ok(AudioInput {
        data: data.to_string(),
        format,
    })
}

fn optional_string<'a>(params: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    params
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
}

impl BaseAudioTranscriptionConfig for BedrockAudioTranscriptionConfig {
    fn secret_names(&self) -> Vec<&'static str> {
        litellm_auth_aws::constants::SECRET_NAMES.to_vec()
    }

    fn get_supported_openai_params(&self) -> &'static [&'static str] {
        SUPPORTED_PARAMS
    }

    fn transform_audio_transcription_request(
        &self,
        _model: &str,
        audio: Value,
        optional_params: Map<String, Value>,
    ) -> Result<AudioTranscriptionRequestData, Error> {
        let audio = audio_fields(audio)?;
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
                        {"audio": {"format": <&'static str>::from(audio.format), "source": {"bytes": audio.data}}},
                        {"text": instruction}
                    ]
                }],
                "system": [{"text": "You are a transcription assistant."}],
                "inferenceConfig": inference_config,
            }),
        })
    }

    fn transform_audio_transcription_response(
        &self,
        _model: &str,
        response_json: Value,
    ) -> Result<AudioTranscriptionResponseData, Error> {
        let response: ConverseResponse =
            serde_json::from_value(response_json).map_err(|error| {
                Error::InvalidResponse(format!("invalid Converse response: {error}").into())
            })?;
        if response.message_content_is_non_text() {
            return Err(Error::InvalidResponse(
                "Bedrock response contains non-text transcript content"
                    .to_string()
                    .into(),
            ));
        }
        Ok(AudioTranscriptionResponseData {
            text: response.content_text(),
        })
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
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

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[("Content-Type", "application/json")]
    }

    fn validate_environment(
        &self,
        headers: Headers,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let (_, model_region) = bedrock_model_id_and_region(model);
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::AwsSigV4 {
                region: resolve_bedrock_region(
                    model_region.as_deref(),
                    optional_params,
                    env_lookup,
                ),
                service: BEDROCK_SERVICE,
                credentials: Box::new(AwsCredentialSource::from_params(
                    optional_params,
                    env_lookup,
                )),
            },
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
            .transform_audio_transcription_request(
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
            .transform_audio_transcription_response(
                "model",
                json!({"output": {"message": {"content": [{"text": "hello "}, {"text": "world"}]}}, "usage": {"inputTokens": 1, "outputTokens": 2}}),
            )
            .expect("response");
        assert_eq!(result.text, "hello world");
        assert_eq!(result.into_json(), json!({"text": "hello world"}));
    }

    #[test]
    fn malformed_response_content_is_rejected() {
        let result = BEDROCK_AUDIO_TRANSCRIPTION_CONFIG.transform_audio_transcription_response(
            "model",
            json!({"output": {"message": {"content": [{"image": {}}]}}, "usage": {"inputTokens": 1, "outputTokens": 2}}),
        );
        assert!(matches!(result, Err(Error::InvalidResponse(_))));
    }

    #[test]
    fn invalid_audio_is_rejected() {
        let result = BEDROCK_AUDIO_TRANSCRIPTION_CONFIG.transform_audio_transcription_request(
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
            .get_complete_url(
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
