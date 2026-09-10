use super::converse::{
    ConverseAudio, ConverseAudioSource, ConverseContent, ConverseMappedParams, ConverseMessage,
    ConverseRequest, ConverseResponseContent, ConverseText,
};
use crate::auth::RequestAuth;
use crate::chat_completions::conversation::TurnRole;
use serde_json::{Map, Value};

use crate::audio_transcription::transformation::AudioTranscriptionProviderConfig;
use crate::audio_transcription::types::{
    AudioTranscriptionRequestData, AudioTranscriptionResponseData, ProviderTranscriptionResponse,
    TranscriptionAudio, TranscriptionParams,
};
use crate::error::Error;
use crate::url_utils::ApiUrl;

pub use super::aws_base::{aws_auth_config, bedrock_model_id_and_region, resolve_bedrock_region};
use super::constants::BEDROCK_RUNTIME_ENDPOINT_TEMPLATE;

pub static BEDROCK_AUDIO_TRANSCRIPTION_CONFIG: BedrockAudioTranscriptionConfig =
    BedrockAudioTranscriptionConfig;

pub struct BedrockAudioTranscriptionConfig;

impl AudioTranscriptionProviderConfig for BedrockAudioTranscriptionConfig {
    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_request(
        &self,
        _model: &str,
        audio: TranscriptionAudio,
        params: TranscriptionParams,
    ) -> Result<AudioTranscriptionRequestData, Error> {
        if audio.data.is_empty() {
            return Err(Error::MissingField("audio.data"));
        }
        let mut instruction = "Transcribe the audio. Respond with only the transcript.".to_string();
        if let Some(language) = params.language.as_deref().filter(|value| !value.is_empty()) {
            instruction.push_str(&format!(" The audio language is {language}."));
        }
        if let Some(prompt) = params.prompt.as_deref().filter(|value| !value.is_empty()) {
            instruction.push_str(&format!(" Additional context: {prompt}"));
        }
        Ok(AudioTranscriptionRequestData::Bedrock(ConverseRequest {
            messages: vec![ConverseMessage {
                role: TurnRole::User,
                content: vec![
                    ConverseContent::Audio(ConverseAudio {
                        format: audio.format,
                        source: ConverseAudioSource { bytes: audio.data },
                    }),
                    ConverseContent::Text(instruction),
                ],
            }],
            system: vec![ConverseText {
                text: "You are a transcription assistant.".into(),
            }],
            inference_config: ConverseMappedParams {
                max_tokens: Some(Some(4096)),
                temperature: params.temperature,
                ..Default::default()
            },
        }))
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_response(
        &self,
        _model: &str,
        response: ProviderTranscriptionResponse,
    ) -> Result<AudioTranscriptionResponseData, Error> {
        let ProviderTranscriptionResponse::Bedrock(response) = response;
        let content = response
            .output
            .and_then(|output| output.message)
            .and_then(|message| message.content)
            .ok_or_else(|| {
                Error::InvalidResponse("Bedrock response has no output content".into())
            })?;
        let text = content
            .into_iter()
            .filter_map(|block| match block {
                ConverseResponseContent::Object(block) => block.text.flatten(),
                ConverseResponseContent::Other(_) => None,
            })
            .collect();
        Ok(AudioTranscriptionResponseData { text })
    }

    fn complete_url(
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
        ApiUrl::parse(&endpoint)
            .and_then(|url| url.complete_path(&["model", &model_id, "converse"]))
            .map(|url| url.into_string())
            .map_err(|error| Error::InvalidRequest(format!("invalid api_base: {error}")))
    }

    fn auth(
        &self,
        _api_key: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<RequestAuth, Error> {
        let (_, model_region) = bedrock_model_id_and_region(model);
        Ok(RequestAuth::AwsSigV4 {
            region: resolve_bedrock_region(model_region.as_deref(), optional_params, env_lookup),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
        let params = serde_json::from_value(Value::Object(params)).unwrap();
        let result = BEDROCK_AUDIO_TRANSCRIPTION_CONFIG
            .transform_request(
                "mistral.voxtral-mini-3b-2507",
                serde_json::from_value(
                    json!({"data": "AQI=", "format": "wav", "filename": "sample.wav"}),
                )
                .unwrap(),
                params,
            )
            .expect("request");
        assert_eq!(
            serde_json::to_value(result).unwrap(),
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
            .transform_response(
                "model",
                serde_json::from_value(json!({"output": {"message": {"content": [{"text": "hello "}, {"text": "world"}]}}})).unwrap(),
            )
            .expect("response");
        assert_eq!(result.text, "hello world");
        assert_eq!(result.into_json(), json!({"text": "hello world"}));
    }

    #[test]
    fn invalid_audio_is_rejected() {
        let result = BEDROCK_AUDIO_TRANSCRIPTION_CONFIG.transform_request(
            "model",
            TranscriptionAudio {
                data: String::new(),
                format: crate::audio_transcription::types::AudioFormat::Wav,
            },
            TranscriptionParams::default(),
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
