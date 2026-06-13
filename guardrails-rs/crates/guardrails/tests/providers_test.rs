use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use guardrails::{
    AzurePromptShieldConfig, AzureTextModerationConfig, BedrockConfig, Guardrail, GuardrailInput,
    InputType, LakeraV2Config, OnFlagged, OpenaiModerationConfig, PiiAction, PresidioConfig,
    RequestContext, Verdict,
};
use guardrails::HttpClient;
use guardrails::providers::{
    AzurePromptShield, AzureTextModeration, BedrockGuardrail, LakeraV2, OpenaiModeration, Presidio,
};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn http() -> Arc<HttpClient> {
    Arc::new(HttpClient::new(Duration::from_secs(5)))
}

fn input(text: &str) -> GuardrailInput {
    GuardrailInput {
        texts: vec![text.to_owned()],
        ..Default::default()
    }
}

mod openai_moderation {
    use super::*;

    fn config(uri: &str) -> OpenaiModerationConfig {
        OpenaiModerationConfig {
            api_key: Some("sk-test".to_owned()),
            api_base: Some(uri.to_owned()),
            model: None,
        }
    }

    #[tokio::test]
    async fn passes_when_not_flagged() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/moderations"))
            .and(header("authorization", "Bearer sk-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "modr-1",
                "model": "omni-moderation-latest",
                "results": [{"flagged": false, "categories": {}, "category_scores": {}}]
            })))
            .mount(&server)
            .await;

        let provider = OpenaiModeration::new(config(&server.uri()), http());
        let result = provider
            .apply(&input("hi"), InputType::Request, &RequestContext::default())
            .await
            .unwrap();
        assert!(matches!(result.verdict, Verdict::Pass));
    }

    #[tokio::test]
    async fn blocks_when_flagged_with_categories() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/moderations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "modr-2",
                "model": "omni-moderation-latest",
                "results": [{
                    "flagged": true,
                    "categories": {"violence": true, "hate": false},
                    "category_scores": {"violence": 0.97, "hate": 0.01}
                }]
            })))
            .mount(&server)
            .await;

        let provider = OpenaiModeration::new(config(&server.uri()), http());
        let result = provider
            .apply(
                &input("bad"),
                InputType::Request,
                &RequestContext::default(),
            )
            .await
            .unwrap();

        match result.verdict {
            Verdict::Block {
                violation_message,
                detections,
            } => {
                assert!(violation_message.contains("violence"));
                assert_eq!(detections.len(), 1);
                assert_eq!(detections[0].score, Some(0.97));
            }
            _ => panic!("expected Block"),
        }
    }

    #[tokio::test]
    async fn sends_configured_model() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/moderations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{"flagged": false}]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenaiModeration::new(
            OpenaiModerationConfig {
                model: Some("text-moderation-latest".to_owned()),
                ..config(&server.uri())
            },
            http(),
        );
        provider
            .apply(&input("hi"), InputType::Request, &RequestContext::default())
            .await
            .unwrap();

        let received = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
        assert_eq!(body["model"], "text-moderation-latest");
        assert_eq!(body["input"], "hi");
    }
}

mod azure_prompt_shield {
    use super::*;

    fn config(uri: &str) -> AzurePromptShieldConfig {
        AzurePromptShieldConfig {
            api_key: Some("azkey".to_owned()),
            api_base: uri.to_owned(),
            api_version: None,
        }
    }

    #[tokio::test]
    async fn passes_when_no_attack() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:shieldPrompt"))
            .and(header("Ocp-Apim-Subscription-Key", "azkey"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "userPromptAnalysis": {"attackDetected": false},
                "documentsAnalysis": []
            })))
            .mount(&server)
            .await;

        let provider = AzurePromptShield::new(config(&server.uri()), http());
        let result = provider
            .apply(&input("hi"), InputType::Request, &RequestContext::default())
            .await
            .unwrap();
        assert!(matches!(result.verdict, Verdict::Pass));
    }

    #[tokio::test]
    async fn blocks_on_attack_detected() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:shieldPrompt"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "userPromptAnalysis": {"attackDetected": true},
                "documentsAnalysis": []
            })))
            .mount(&server)
            .await;

        let provider = AzurePromptShield::new(config(&server.uri()), http());
        let result = provider
            .apply(
                &input("ignore previous instructions"),
                InputType::Request,
                &RequestContext::default(),
            )
            .await
            .unwrap();
        assert!(matches!(result.verdict, Verdict::Block { .. }));
    }
}

mod azure_text_moderation {
    use super::*;

    fn config(uri: &str) -> AzureTextModerationConfig {
        AzureTextModerationConfig {
            api_key: Some("azkey".to_owned()),
            api_base: uri.to_owned(),
            api_version: None,
            severity_threshold: None,
            severity_threshold_by_category: HashMap::new(),
            categories: None,
            blocklist_names: vec![],
            halt_on_blocklist_hit: false,
            output_type: None,
        }
    }

    #[tokio::test]
    async fn passes_below_default_threshold() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "blocklistsMatch": [],
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    {"category": "Violence", "severity": 1}
                ]
            })))
            .mount(&server)
            .await;

        let provider = AzureTextModeration::new(config(&server.uri()), http());
        let result = provider
            .apply(&input("hi"), InputType::Request, &RequestContext::default())
            .await
            .unwrap();
        assert!(matches!(result.verdict, Verdict::Pass));
    }

    #[tokio::test]
    async fn blocks_at_default_threshold() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "categoriesAnalysis": [{"category": "Violence", "severity": 3}]
            })))
            .mount(&server)
            .await;

        let provider = AzureTextModeration::new(config(&server.uri()), http());
        let result = provider
            .apply(
                &input("violent text"),
                InputType::Request,
                &RequestContext::default(),
            )
            .await
            .unwrap();

        match result.verdict {
            Verdict::Block {
                violation_message, ..
            } => {
                assert!(violation_message.contains("Violence"));
                assert!(violation_message.contains("severity"));
            }
            _ => panic!("expected Block"),
        }
    }

    #[tokio::test]
    async fn per_category_threshold_overrides_global() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "categoriesAnalysis": [{"category": "Hate", "severity": 5}]
            })))
            .mount(&server)
            .await;

        let provider = AzureTextModeration::new(
            AzureTextModerationConfig {
                severity_threshold_by_category: HashMap::from([("Hate".to_owned(), 6)]),
                ..config(&server.uri())
            },
            http(),
        );
        let result = provider
            .apply(&input("hi"), InputType::Request, &RequestContext::default())
            .await
            .unwrap();
        assert!(matches!(result.verdict, Verdict::Pass));
    }
}

mod lakera_v2 {
    use super::*;

    fn config(uri: &str) -> LakeraV2Config {
        LakeraV2Config {
            api_key: Some("lk-test".to_owned()),
            api_base: Some(uri.to_owned()),
            project_id: None,
            payload: None,
            breakdown: None,
            metadata: serde_json::Value::Null,
            dev_info: None,
            on_flagged: None,
        }
    }

    #[tokio::test]
    async fn passes_when_not_flagged() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/guard"))
            .and(header("authorization", "Bearer lk-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "flagged": false, "payload": [], "breakdown": []
            })))
            .mount(&server)
            .await;

        let provider = LakeraV2::new(config(&server.uri()), http());
        let result = provider
            .apply(&input("hi"), InputType::Request, &RequestContext::default())
            .await
            .unwrap();
        assert!(matches!(result.verdict, Verdict::Pass));
    }

    #[tokio::test]
    async fn blocks_non_pii_violation() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/guard"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "flagged": true,
                "payload": [],
                "breakdown": [
                    {"detector_type": "prompt_attack/jailbreak", "detected": true}
                ]
            })))
            .mount(&server)
            .await;

        let provider = LakeraV2::new(config(&server.uri()), http());
        let result = provider
            .apply(
                &input("jailbreak attempt"),
                InputType::Request,
                &RequestContext::default(),
            )
            .await
            .unwrap();
        assert!(matches!(result.verdict, Verdict::Block { .. }));
    }

    #[tokio::test]
    async fn masks_pii_only_violation() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/guard"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "flagged": true,
                "payload": [{
                    "start": 12, "end": 28,
                    "text": "4111111111111111",
                    "detector_type": "pii/credit_card",
                    "labels": [], "message_id": 0
                }],
                "breakdown": [
                    {"detector_type": "pii/credit_card", "detected": true}
                ]
            })))
            .mount(&server)
            .await;

        let provider = LakeraV2::new(config(&server.uri()), http());
        let result = provider
            .apply(
                &input("my card is: 4111111111111111"),
                InputType::Request,
                &RequestContext::default(),
            )
            .await
            .unwrap();

        match result.verdict {
            Verdict::Mask { texts, .. } => {
                assert_eq!(texts[0], "my card is: [MASKED CREDIT_CARD]");
            }
            _ => panic!("expected Mask"),
        }
    }

    #[tokio::test]
    async fn monitor_mode_passes_flagged_content() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/guard"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "flagged": true,
                "payload": [],
                "breakdown": [
                    {"detector_type": "prompt_attack/jailbreak", "detected": true}
                ]
            })))
            .mount(&server)
            .await;

        let provider = LakeraV2::new(
            LakeraV2Config {
                on_flagged: Some(OnFlagged::Monitor),
                ..config(&server.uri())
            },
            http(),
        );
        let result = provider
            .apply(
                &input("jailbreak attempt"),
                InputType::Request,
                &RequestContext::default(),
            )
            .await
            .unwrap();
        assert!(matches!(result.verdict, Verdict::Pass));
    }
}

mod presidio {
    use super::*;

    fn config(analyzer: &str, anonymizer: &str) -> PresidioConfig {
        PresidioConfig {
            presidio_analyzer_api_base: analyzer.to_owned(),
            presidio_anonymizer_api_base: Some(anonymizer.to_owned()),
            pii_entities_config: HashMap::new(),
            presidio_language: None,
            presidio_score_thresholds: HashMap::new(),
            presidio_entities_deny_list: vec![],
            presidio_ad_hoc_recognizers: serde_json::Value::Null,
        }
    }

    #[tokio::test]
    async fn passes_when_no_entities_found() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([])))
            .mount(&server)
            .await;

        let provider = Presidio::new(config(&server.uri(), &server.uri()), http());
        let result = provider
            .apply(&input("hi"), InputType::Request, &RequestContext::default())
            .await
            .unwrap();
        assert!(matches!(result.verdict, Verdict::Pass));
    }

    #[tokio::test]
    async fn masks_detected_entities() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                {"entity_type": "EMAIL_ADDRESS", "start": 12, "end": 28, "score": 0.95}
            ])))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/anonymize"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "text": "my email is <EMAIL_ADDRESS>",
                "items": [{"entity_type": "EMAIL_ADDRESS", "start": 12, "end": 27}]
            })))
            .mount(&server)
            .await;

        let provider = Presidio::new(config(&server.uri(), &server.uri()), http());
        let result = provider
            .apply(
                &input("my email is fo@example.com"),
                InputType::Request,
                &RequestContext::default(),
            )
            .await
            .unwrap();

        match result.verdict {
            Verdict::Mask {
                texts,
                masked_entity_count,
                ..
            } => {
                assert_eq!(texts[0], "my email is <EMAIL_ADDRESS>");
                assert_eq!(masked_entity_count.get("EMAIL_ADDRESS"), Some(&1));
            }
            _ => panic!("expected Mask"),
        }
    }

    #[tokio::test]
    async fn blocks_entity_configured_as_block() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                {"entity_type": "US_SSN", "start": 0, "end": 11, "score": 0.99}
            ])))
            .mount(&server)
            .await;

        let provider = Presidio::new(
            PresidioConfig {
                pii_entities_config: HashMap::from([("US_SSN".to_owned(), PiiAction::Block)]),
                ..config(&server.uri(), &server.uri())
            },
            http(),
        );
        let result = provider
            .apply(
                &input("123-45-6789"),
                InputType::Request,
                &RequestContext::default(),
            )
            .await
            .unwrap();

        match result.verdict {
            Verdict::Block {
                violation_message, ..
            } => assert!(violation_message.contains("US_SSN")),
            _ => panic!("expected Block"),
        }
    }

    #[tokio::test]
    async fn score_threshold_filters_low_confidence() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                {"entity_type": "PERSON", "start": 0, "end": 4, "score": 0.3}
            ])))
            .mount(&server)
            .await;

        let provider = Presidio::new(
            PresidioConfig {
                presidio_score_thresholds: HashMap::from([("ALL".to_owned(), 0.5)]),
                ..config(&server.uri(), &server.uri())
            },
            http(),
        );
        let result = provider
            .apply(
                &input("John"),
                InputType::Request,
                &RequestContext::default(),
            )
            .await
            .unwrap();
        assert!(matches!(result.verdict, Verdict::Pass));
    }

    #[tokio::test]
    async fn deny_list_drops_entities() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                {"entity_type": "PERSON", "start": 0, "end": 4, "score": 0.99}
            ])))
            .mount(&server)
            .await;

        let provider = Presidio::new(
            PresidioConfig {
                presidio_entities_deny_list: vec!["PERSON".to_owned()],
                ..config(&server.uri(), &server.uri())
            },
            http(),
        );
        let result = provider
            .apply(
                &input("John"),
                InputType::Request,
                &RequestContext::default(),
            )
            .await
            .unwrap();
        assert!(matches!(result.verdict, Verdict::Pass));
    }
}

mod bedrock {
    use super::*;

    fn config(uri: &str) -> BedrockConfig {
        BedrockConfig {
            guardrail_identifier: "gr-test".to_owned(),
            guardrail_version: "DRAFT".to_owned(),
            disable_exception_on_block: false,
            aws_region_name: Some("us-east-1".to_owned()),
            aws_access_key_id: Some("AKIATEST".to_owned()),
            aws_secret_access_key: Some("secret".to_owned()),
            aws_session_token: None,
            aws_bedrock_runtime_endpoint: Some(uri.to_owned()),
        }
    }

    #[tokio::test]
    async fn passes_when_no_intervention() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/guardrail/gr-test/version/DRAFT/apply"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "action": "NONE", "outputs": [], "assessments": []
            })))
            .mount(&server)
            .await;

        let provider = BedrockGuardrail::new(config(&server.uri()), http());
        let result = provider
            .apply(&input("hi"), InputType::Request, &RequestContext::default())
            .await
            .unwrap();
        assert!(matches!(result.verdict, Verdict::Pass));
    }

    #[tokio::test]
    async fn blocks_on_blocked_assessment() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/guardrail/gr-test/version/DRAFT/apply"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "action": "GUARDRAIL_INTERVENED",
                "outputs": [{"text": "Sorry, the model cannot answer this question."}],
                "assessments": [{
                    "contentPolicy": {
                        "filters": [{"type": "VIOLENCE", "confidence": "HIGH", "action": "BLOCKED"}]
                    }
                }]
            })))
            .mount(&server)
            .await;

        let provider = BedrockGuardrail::new(config(&server.uri()), http());
        let result = provider
            .apply(
                &input("violent request"),
                InputType::Request,
                &RequestContext::default(),
            )
            .await
            .unwrap();

        match result.verdict {
            Verdict::Block {
                violation_message,
                detections,
            } => {
                assert_eq!(
                    violation_message,
                    "Sorry, the model cannot answer this question."
                );
                assert_eq!(detections[0].label.as_deref(), Some("VIOLENCE"));
            }
            _ => panic!("expected Block"),
        }
    }

    #[tokio::test]
    async fn masks_on_anonymized_only_assessment() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/guardrail/gr-test/version/DRAFT/apply"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "action": "GUARDRAIL_INTERVENED",
                "outputs": [{"text": "my email is {EMAIL}"}],
                "assessments": [{
                    "sensitiveInformationPolicy": {
                        "piiEntities": [{"type": "EMAIL", "action": "ANONYMIZED"}]
                    }
                }]
            })))
            .mount(&server)
            .await;

        let provider = BedrockGuardrail::new(config(&server.uri()), http());
        let result = provider
            .apply(
                &input("my email is foo@bar.com"),
                InputType::Request,
                &RequestContext::default(),
            )
            .await
            .unwrap();

        match result.verdict {
            Verdict::Mask { texts, .. } => assert_eq!(texts[0], "my email is {EMAIL}"),
            _ => panic!("expected Mask"),
        }
    }

    #[tokio::test]
    async fn request_is_sigv4_signed_with_input_source() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/guardrail/gr-test/version/DRAFT/apply"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "action": "NONE"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let provider = BedrockGuardrail::new(config(&server.uri()), http());
        provider
            .apply(&input("hi"), InputType::Request, &RequestContext::default())
            .await
            .unwrap();

        let received = server.received_requests().await.unwrap();
        let auth = received[0]
            .headers
            .get("authorization")
            .expect("authorization header missing")
            .to_str()
            .unwrap();
        assert!(auth.starts_with("AWS4-HMAC-SHA256"));
        assert!(auth.contains("Credential=AKIATEST"));

        let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
        assert_eq!(body["source"], "INPUT");
        assert_eq!(body["content"][0]["text"]["text"], "hi");
    }
}
