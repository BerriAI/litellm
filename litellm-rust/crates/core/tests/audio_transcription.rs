use litellm_core::audio_transcription::{
    Error, audio_transcription, types::AudioTranscriptionRequest,
};
use rstest::{fixture, rstest};
use serde_json::{Map, Value, json};
use wiremock::ResponseTemplate;

mod support;
use support::*;

const MODEL: &str = "mistral.voxtral-mini-3b-2507";

async fn transcribe(request: AudioTranscriptionRequest<'_>) -> Result<Value, Error> {
    audio_transcription(&support::resources(), &http_config(), request).await
}

fn transcript_response(text: &str) -> ResponseTemplate {
    json_response(json!({"output": {"message": {"content": [{"text": text}]}}}))
}

fn aws_params(region: &str) -> Map<String, Value> {
    Map::from_iter([
        ("aws_access_key_id".to_string(), json!("access-key")),
        ("aws_secret_access_key".to_string(), json!("secret-key")),
        ("aws_region_name".to_string(), json!(region)),
    ])
}

#[fixture]
fn request() -> AudioTranscriptionRequest<'static> {
    AudioTranscriptionRequest {
        model: MODEL,
        audio: json!({"data": "AQI=", "format": "wav", "filename": "audio.wav"}),
        api_key: None,
        api_base: None,
        custom_llm_provider: Some("bedrock"),
        extra_headers: None,
        optional_params: aws_params("us-east-1"),
        timeout: None,
    }
}

#[rstest]
#[case::us_east_1("us-east-1")]
#[case::eu_west_1("eu-west-1")]
#[tokio::test]
async fn bedrock_converse_request_is_signed_for_the_requested_region(
    request: AudioTranscriptionRequest<'static>,
    #[case] region: &str,
) {
    let upstream = upstream([transcript_response("hello")]).await;
    let base = upstream.uri();

    let response = transcribe(AudioTranscriptionRequest {
        api_base: Some(&base),
        optional_params: aws_params(region),
        ..request
    })
    .await
    .expect("transcription");

    assert_eq!(response, json!({"text": "hello"}));
    let sent = only_request(&upstream).await;
    assert_eq!(sent.method.as_str(), "POST");
    assert_eq!(sent.url.path(), format!("/model/{MODEL}/converse"));
    let authorization = sent.header("authorization").expect("request is signed");
    assert!(
        authorization.starts_with("AWS4-HMAC-SHA256 Credential=access-key/"),
        "{authorization}"
    );
    assert!(
        authorization.contains(&format!("/{region}/bedrock/aws4_request")),
        "{authorization}"
    );
    assert!(sent.header("x-amz-date").is_some());
    assert!(!sent.body_text().contains("secret-key"));
}

#[rstest]
#[tokio::test]
async fn the_provider_can_come_from_the_model_prefix(request: AudioTranscriptionRequest<'static>) {
    let upstream = upstream([transcript_response("hello")]).await;
    let base = upstream.uri();
    let model = format!("bedrock/{MODEL}");

    transcribe(AudioTranscriptionRequest {
        model: &model,
        custom_llm_provider: None,
        api_base: Some(&base),
        ..request
    })
    .await
    .expect("transcription");

    assert_eq!(
        only_request(&upstream).await.url.path(),
        format!("/model/{MODEL}/converse")
    );
}

#[rstest]
#[tokio::test]
async fn audio_and_transcription_params_reach_the_converse_body(
    request: AudioTranscriptionRequest<'static>,
    #[values("wav", "mp3", "flac", "ogg")] format: &str,
) {
    let upstream = upstream([transcript_response("hello")]).await;
    let base = upstream.uri();
    let optional_params = aws_params("us-east-1")
        .into_iter()
        .chain([
            ("language".to_string(), json!("fr")),
            ("temperature".to_string(), json!(0.2)),
        ])
        .collect();

    transcribe(AudioTranscriptionRequest {
        audio: json!({"data": "AQI=", "format": format}),
        api_base: Some(&base),
        optional_params,
        ..request
    })
    .await
    .expect("transcription");

    let body = only_request(&upstream).await.json();
    let content = &body["messages"][0]["content"];
    assert_eq!(
        content[0],
        json!({"audio": {"format": format, "source": {"bytes": "AQI="}}})
    );
    let instruction = content[1]["text"].as_str().expect("instruction text");
    assert!(instruction.contains("fr"), "{instruction}");
    assert_eq!(body["inferenceConfig"]["temperature"], 0.2);
}

#[rstest]
#[case::unknown_format(json!({"data": "AQI=", "format": "aac"}))]
#[case::missing_data(json!({"format": "wav"}))]
#[case::not_an_object(json!("AQI="))]
#[tokio::test]
async fn invalid_audio_is_rejected_before_sending(
    request: AudioTranscriptionRequest<'static>,
    #[case] audio: Value,
) {
    let upstream = upstream([transcript_response("hello")]).await;
    let base = upstream.uri();

    let error = transcribe(AudioTranscriptionRequest {
        audio,
        api_base: Some(&base),
        ..request
    })
    .await
    .expect_err("invalid audio is rejected");

    assert!(
        matches!(
            error,
            Error::InvalidRequest(_) | Error::MissingField(_) | Error::InvalidType { .. }
        ),
        "{error:?}"
    );
    assert!(received(&upstream).await.is_empty());
}

#[rstest]
#[case::unknown_provider(MODEL, Some("openai"), "openai")]
#[case::unresolvable_model(
    "no-such-model",
    None,
    "unable to resolve custom_llm_provider for audio transcription request"
)]
#[tokio::test]
async fn unsupported_providers_are_rejected_before_sending(
    request: AudioTranscriptionRequest<'static>,
    #[case] model: &'static str,
    #[case] provider: Option<&'static str>,
    #[case] reported: &str,
) {
    let error = transcribe(AudioTranscriptionRequest {
        model,
        custom_llm_provider: provider,
        api_base: Some(UNREACHABLE_BASE),
        ..request
    })
    .await
    .expect_err("unsupported provider errors");

    assert_eq!(error, Error::InvalidProvider(reported.into()));
}

#[rstest]
#[tokio::test]
async fn a_non_string_extra_header_is_rejected(request: AudioTranscriptionRequest<'static>) {
    let error = transcribe(AudioTranscriptionRequest {
        extra_headers: Some(Map::from_iter([("x-count".to_string(), json!(3))])),
        api_base: Some(UNREACHABLE_BASE),
        ..request
    })
    .await
    .expect_err("a non-string header is rejected");

    assert!(matches!(error, Error::Headers(_)), "{error:?}");
}

#[rstest]
#[case::throttled(429)]
#[case::server_error(500)]
#[tokio::test]
async fn an_upstream_error_keeps_its_status_and_body(
    request: AudioTranscriptionRequest<'static>,
    #[case] status: u16,
) {
    let upstream =
        upstream([ResponseTemplate::new(status).set_body_string("upstream said no")]).await;
    let base = upstream.uri();

    let error = transcribe(AudioTranscriptionRequest {
        api_base: Some(&base),
        ..request
    })
    .await
    .expect_err("upstream error propagates");

    assert_eq!(
        error,
        Error::Transport(litellm_http::transport::Error::Http {
            status,
            body: "upstream said no".into()
        })
    );
}

#[rstest]
#[case::not_json(ResponseTemplate::new(200).set_body_string("not json"))]
#[case::no_output(json_response(json!({"unexpected": true})))]
#[tokio::test]
async fn an_unreadable_success_body_is_an_invalid_response(
    request: AudioTranscriptionRequest<'static>,
    #[case] response: ResponseTemplate,
) {
    let upstream = upstream([response]).await;
    let base = upstream.uri();

    let error = transcribe(AudioTranscriptionRequest {
        api_base: Some(&base),
        ..request
    })
    .await
    .expect_err("an unreadable body fails");

    assert!(matches!(error, Error::InvalidResponse(_)), "{error:?}");
}
