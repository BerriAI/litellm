use litellm_inference::Connection;
use litellm_inference_transcription::{Error, types::AudioTranscriptionRequest};
use rstest::{fixture, rstest};
use serde_json::{Map, Value, json};
use wiremock::ResponseTemplate;

mod support;
use support::*;

const MODEL: &str = "mistral.voxtral-mini-3b-2507";

async fn transcribe(request: AudioTranscriptionRequest) -> Result<Value, Error> {
    audio_transcription_route().execute(request).await
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
fn request() -> AudioTranscriptionRequest {
    AudioTranscriptionRequest {
        model: MODEL.into(),
        custom_llm_provider: Some("bedrock".into()),
        audio: json!({"data": "AQI=", "format": "wav", "filename": "audio.wav"}),
        optional_params: aws_params("us-east-1"),
        connection: Connection::default(),
    }
}

fn at(base: &str, request: AudioTranscriptionRequest) -> AudioTranscriptionRequest {
    AudioTranscriptionRequest {
        connection: Connection {
            api_base: Some(base.into()),
            ..request.connection
        },
        ..request
    }
}

#[rstest]
#[case::us_east_1("us-east-1")]
#[case::eu_west_1("eu-west-1")]
#[tokio::test]
async fn bedrock_converse_request_is_signed_for_the_requested_region(
    request: AudioTranscriptionRequest,
    #[case] region: &str,
) {
    let upstream = upstream([transcript_response("hello")]).await;
    let base = upstream.uri();

    let response = transcribe(AudioTranscriptionRequest {
        optional_params: aws_params(region),
        ..at(&base, request)
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
async fn the_provider_can_come_from_the_model_prefix(request: AudioTranscriptionRequest) {
    let upstream = upstream([transcript_response("hello")]).await;
    let base = upstream.uri();
    let model = format!("bedrock/{MODEL}");

    transcribe(AudioTranscriptionRequest {
        model,
        custom_llm_provider: None,
        ..at(&base, request)
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
    request: AudioTranscriptionRequest,
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
        optional_params,
        ..at(&base, request)
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
    request: AudioTranscriptionRequest,
    #[case] audio: Value,
) {
    let upstream = upstream([transcript_response("hello")]).await;
    let base = upstream.uri();

    let error = transcribe(AudioTranscriptionRequest {
        audio,
        ..at(&base, request)
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
    request: AudioTranscriptionRequest,
    #[case] model: &'static str,
    #[case] provider: Option<&'static str>,
    #[case] reported: &str,
) {
    let error = transcribe(AudioTranscriptionRequest {
        model: model.into(),
        custom_llm_provider: provider.map(str::to_owned),
        ..at(UNREACHABLE_BASE, request)
    })
    .await
    .expect_err("unsupported provider errors");

    assert_eq!(error, Error::InvalidProvider(reported.into()));
}

#[rstest]
#[tokio::test]
async fn a_non_string_extra_header_is_rejected(request: AudioTranscriptionRequest) {
    let error = transcribe(AudioTranscriptionRequest {
        connection: Connection {
            api_base: Some(UNREACHABLE_BASE.into()),
            extra_headers: Some(Map::from_iter([("x-count".to_string(), json!(3))])),
            ..Connection::default()
        },
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
    request: AudioTranscriptionRequest,
    #[case] status: u16,
) {
    let upstream =
        upstream([ResponseTemplate::new(status).set_body_string("upstream said no")]).await;
    let base = upstream.uri();

    let error = transcribe(at(&base, request))
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
    request: AudioTranscriptionRequest,
    #[case] response: ResponseTemplate,
) {
    let upstream = upstream([response]).await;
    let base = upstream.uri();

    let error = transcribe(at(&base, request))
        .await
        .expect_err("an unreadable body fails");

    assert!(matches!(error, Error::InvalidResponse(_)), "{error:?}");
}

#[rstest]
#[tokio::test]
async fn transcription_records_route_and_resolved_provider(
    request: AudioTranscriptionRequest,
    traces: TraceCapture,
) {
    let upstream = upstream([transcript_response("hello")]).await;
    let base = upstream.uri();
    let model = request.model.clone();
    traces
        .logger()
        .instrument(transcribe(at(&base, request)))
        .await
        .unwrap();
    let summaries = traces.summaries("litellm.route");
    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0]["route"], "audio_transcription");
    assert_eq!(summaries[0]["model"], model);
    assert_eq!(summaries[0]["resolved_model"], model);
    assert_eq!(summaries[0]["provider"], "bedrock");
    assert_eq!(summaries[0]["outcome"], "success");
    assert_eq!(summaries[0]["stream"], false);
    assert!(!format!("{:?}", traces.records()).contains("secret-key"));
}
