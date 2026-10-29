use litellm_auth::AwsParams;
use litellm_auth_aws::{
    AwsCredentialSource, bedrock_model_id_and_region,
    constants::{BEDROCK_RUNTIME_ENDPOINT_TEMPLATE, BEDROCK_SERVICE},
    resolve_bedrock_region,
};
use litellm_core_utils::core_helpers::json_type_name;
use litellm_llms_types::{
    formats::audio_transcription::{AudioTranscriptionResponseData, AudioTranscriptionUsage},
    providers::bedrock::{ConverseResponse, ConverseUsage},
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use strum::IntoStaticStr;

use crate::{
    Error, ErrorDetail,
    base_llm::{
        audio_transcription::transformation::{
            AudioTranscriptionRequestData, BaseAudioTranscriptionConfig, Headers,
            ValidatedEnvironment,
        },
        auth::AuthScheme,
    },
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
    let format = object
        .get("format")
        .cloned()
        .ok_or(Error::MissingField("audio.format"))?;
    let format: AudioFormat = serde_json::from_value(format)
        .map_err(|error| Error::InvalidRequest(ErrorDetail::invalid("audio.format", error)))?;
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
        litellm_auth::AwsParams::secret_names().to_vec()
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
        let temperature = optional_params
            .get("temperature")
            .cloned()
            .unwrap_or(json!(0));
        Ok(AudioTranscriptionRequestData {
            body: json!({
                "messages": [{
                    "role": "user",
                    "content": [
                        {"audio": {"format": <&'static str>::from(audio.format), "source": {"bytes": audio.data}}},
                        {"text": instruction}
                    ]
                }],
                "inferenceConfig": {"maxTokens": 4096, "temperature": temperature},
            }),
        })
    }

    fn transform_audio_transcription_response(
        &self,
        _model: &str,
        response_json: Value,
    ) -> Result<AudioTranscriptionResponseData, Error> {
        let reports_usage = response_json.get("usage").is_some();
        let response: ConverseResponse =
            serde_json::from_value(response_json).map_err(|error| {
                Error::InvalidResponse(ErrorDetail::invalid("Converse response", error))
            })?;
        if response.message_content_is_non_text() {
            return Err(Error::InvalidResponse(ErrorDetail::Expected {
                subject: "Bedrock transcript content",
                expected: "text",
            }));
        }
        Ok(AudioTranscriptionResponseData {
            text: response.content_text(),
            usage: reports_usage.then(|| transcription_usage(&response.usage)),
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
        let region = resolve_bedrock_region(
            model_region.as_deref(),
            &AwsParams::from_optional_params(optional_params),
            env_lookup,
        );
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
        let params = AwsParams::from_optional_params(optional_params);
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::AwsSigV4 {
                region: resolve_bedrock_region(model_region.as_deref(), &params, env_lookup),
                service: BEDROCK_SERVICE,
                credentials: Box::new(AwsCredentialSource::from_params(&params, env_lookup)),
            },
        })
    }
}

fn transcription_usage(usage: &ConverseUsage) -> AudioTranscriptionUsage {
    AudioTranscriptionUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        total_tokens: usage
            .total_tokens
            .unwrap_or(usage.input_tokens + usage.output_tokens),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[rstest]
    #[case::caller_temperature(Some(json!(0.4)), json!(0.4))]
    #[case::default_temperature(None, json!(0))]
    fn request_sends_audio_and_instruction_without_system_block(
        #[case] temperature: Option<Value>,
        #[case] expected_temperature: Value,
    ) {
        let params = Map::from_iter(
            [
                ("language".to_string(), json!("en")),
                ("prompt".to_string(), json!("Speaker names")),
                ("timestamp_granularities".to_string(), json!(["word"])),
            ]
            .into_iter()
            .chain(temperature.map(|value| ("temperature".to_string(), value))),
        );
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
                "inferenceConfig": {"maxTokens": 4096, "temperature": expected_temperature}
            })
        );
    }

    #[rstest]
    #[case::reported_total(
        json!({"inputTokens": 390, "outputTokens": 60, "totalTokens": 450}),
        json!({"text": "hello world", "usage": {"type": "tokens", "input_tokens": 390, "output_tokens": 60, "total_tokens": 450}})
    )]
    #[case::derived_total(
        json!({"inputTokens": 1, "outputTokens": 2}),
        json!({"text": "hello world", "usage": {"type": "tokens", "input_tokens": 1, "output_tokens": 2, "total_tokens": 3}})
    )]
    fn response_carries_text_and_converse_usage(#[case] usage: Value, #[case] expected: Value) {
        let result = BEDROCK_AUDIO_TRANSCRIPTION_CONFIG
            .transform_audio_transcription_response(
                "model",
                json!({"output": {"message": {"content": [{"text": "hello "}, {"text": "world"}]}}, "usage": usage}),
            )
            .expect("response");
        assert_eq!(result.into_json(), expected);
    }

    #[rstest]
    fn response_without_usage_reports_none() {
        let result = BEDROCK_AUDIO_TRANSCRIPTION_CONFIG
            .transform_audio_transcription_response(
                "model",
                json!({"output": {"message": {"content": [{"text": "hello"}]}}}),
            )
            .expect("response");
        assert_eq!(result.into_json(), json!({"text": "hello"}));
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
