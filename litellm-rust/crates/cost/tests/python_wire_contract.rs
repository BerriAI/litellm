#![allow(clippy::disallowed_types)]

// mirrors: crates/cost/tests/generate_python_fixtures.py
// (wire_usage + wire_response surfaces)

use serde_json::Value;

use litellm_cost::wire::{
    ChatUsageInput, HiddenParamsInput, ImageResponseInput, OptionalParamsInput, QueryCount,
    ReportedCostInput, ResponseInput, TranscriptionUsageInput, UsageInput, WireNumber,
};
use rstest::rstest;

fn fixture() -> Value {
    serde_json::from_str(include_str!("python_fixtures.json")).expect("fixture parses")
}

fn usage_input(payload: &Value) -> UsageInput {
    serde_json::from_value(payload.clone()).expect("wire usage payload deserializes")
}

#[rstest]
fn wire_usage_payloads_match_their_declared_format() {
    for row in fixture()["wire_usage"].as_array().expect("wire_usage rows") {
        let payload = &row["payload"];
        let input = usage_input(payload);
        let declared = payload["format"].as_str().expect("format declared");
        let actual = match input {
            UsageInput::Chat { .. } => "chat",
            UsageInput::Responses { .. } => "responses",
            UsageInput::Anthropic { .. } => "anthropic",
            UsageInput::Interactions { .. } => "interactions",
            UsageInput::Transcription { .. } => "transcription",
        };
        assert_eq!(actual, declared, "row {}", row["id"]);
    }
}

#[rstest]
fn chat_wire_usage_carries_every_field_the_cost_path_reads() {
    let input = usage_input(&fixture()["wire_usage"][0]["payload"]);
    let UsageInput::Chat { usage } = input else {
        panic!("chat tag selects chat payload");
    };
    assert_eq!(usage.prompt_tokens, 1_000);
    assert_eq!(usage.completion_tokens, 500);
    assert_eq!(usage.cost.map(ReportedCostInput::cost), Some(Some(0.75)));
    assert_eq!(usage.cache_read_input_tokens, Some(800));
    let details = usage.prompt_tokens_details.expect("prompt details present");
    assert_eq!(details.cached_tokens, 800);
    assert_eq!(details.cache_write_tokens, Some(300));
    assert_eq!(details.web_search_requests, Some(2));
    let creation = details
        .cache_creation_token_details
        .expect("cache creation details present");
    assert_eq!(creation.ephemeral_5m_input_tokens, Some(200));
    assert_eq!(creation.ephemeral_1h_input_tokens, Some(100));
    let completion = usage
        .completion_tokens_details
        .expect("completion details present");
    assert_eq!(completion.reasoning_tokens, Some(120));
    assert_eq!(completion.text_tokens, Some(380));
    let server_tool_use = usage.server_tool_use.expect("server tool use present");
    assert_eq!(server_tool_use.web_search_requests, Some(3));
    assert_eq!(usage.speed.as_deref(), Some("fast"));
    assert_eq!(usage.inference_geo.as_deref(), Some("us"));
    assert_eq!(usage._cache_read_input_tokens, Some(800));
    let server_side = usage
        .server_side_tool_usage_details
        .expect("server side details present");
    assert_eq!(server_side.web_search_calls, Some(4));
    assert_eq!(usage.citation_tokens, Some(1_500));
}

#[rstest]
fn chat_wire_usage_with_gateway_merged_token_keys_stays_chat() {
    let input = usage_input(&fixture()["wire_usage"][2]["payload"]);
    let UsageInput::Chat { usage } = input else {
        panic!("gateway merged usage must stay chat");
    };
    assert_eq!(usage.prompt_tokens, 100);
    assert_eq!(usage.completion_tokens, 50);
}

#[rstest]
fn responses_wire_usage_reads_both_detail_spellings() {
    let singular = usage_input(&fixture()["wire_usage"][3]["payload"]);
    let UsageInput::Responses { usage } = singular else {
        panic!("responses tag selects responses payload");
    };
    assert_eq!(usage.input_tokens, 1_200);
    assert_eq!(usage.output_tokens, 300);
    assert_eq!(
        usage.input_token_details.as_ref().map(|d| d.cached_tokens),
        Some(800)
    );
    let plural = usage_input(&fixture()["wire_usage"][4]["payload"]);
    let UsageInput::Responses { usage } = plural else {
        panic!("responses tag selects responses payload");
    };
    assert_eq!(
        usage.input_tokens_details.as_ref().map(|d| d.cached_tokens),
        Some(200)
    );
    assert_eq!(usage.cost.map(ReportedCostInput::cost), Some(Some(0.25)));
}

#[rstest]
fn anthropic_wire_usage_types_iterations_and_cache_breakdown() {
    let full = usage_input(&fixture()["wire_usage"][5]["payload"]);
    let UsageInput::Anthropic { usage } = full else {
        panic!("anthropic tag selects anthropic payload");
    };
    assert_eq!(usage.input_tokens, 5_000);
    assert_eq!(usage.cache_read_input_tokens, Some(3_000));
    let creation = usage
        .cache_creation
        .as_ref()
        .expect("cache creation present");
    assert_eq!(creation.ephemeral_5m_input_tokens, Some(1_500));
    assert_eq!(creation.ephemeral_1h_input_tokens, Some(500));
    assert_eq!(
        usage
            .server_tool_use
            .as_ref()
            .and_then(|s| s.web_search_requests),
        Some(2)
    );
    assert_eq!(usage.speed.as_deref(), Some("fast"));
    let iterations = usage_input(&fixture()["wire_usage"][6]["payload"]);
    let UsageInput::Anthropic { usage } = iterations else {
        panic!("anthropic tag selects anthropic payload");
    };
    let iterations = usage.iterations.expect("iterations present");
    assert_eq!(iterations.len(), 2);
    assert_eq!(iterations[0].cache_read_input_tokens, Some(20));
    assert_eq!(
        iterations[0]
            .output_tokens_details
            .as_ref()
            .and_then(|d| d.thinking_tokens),
        Some(90)
    );
    assert_eq!(
        iterations[0]
            .cache_creation
            .as_ref()
            .and_then(|d| d.ephemeral_5m_input_tokens),
        Some(10)
    );
}

#[rstest]
fn interactions_wire_usage_types_modality_and_grounding_counts() {
    let input = usage_input(&fixture()["wire_usage"][7]["payload"]);
    let UsageInput::Interactions { usage } = input else {
        panic!("interactions tag selects interactions payload");
    };
    assert_eq!(usage.total_input_tokens, 1_000);
    assert_eq!(usage.total_tool_use_tokens, 50);
    assert_eq!(usage.total_cached_tokens, 400);
    assert_eq!(usage.total_reasoning_tokens, 120);
    assert_eq!(usage.input_tokens_by_modality.len(), 2);
    assert_eq!(
        usage.input_tokens_by_modality[0].modality.as_deref(),
        Some("TEXT")
    );
    assert_eq!(usage.grounding_tool_count[0].count, 3);
    assert_eq!(
        usage.grounding_tool_count[0].kind.as_deref(),
        Some("google_search")
    );
}

#[rstest]
fn transcription_wire_usage_types_duration_and_tokens_variants() {
    let tokens = usage_input(&fixture()["wire_usage"][8]["payload"]);
    let UsageInput::Transcription { usage } = tokens else {
        panic!("transcription tag selects transcription payload");
    };
    let TranscriptionUsageInput::Tokens {
        input_tokens,
        output_tokens,
        total_tokens,
        input_token_details,
    } = usage
    else {
        panic!("type tag selects the tokens variant");
    };
    assert_eq!((input_tokens, output_tokens, total_tokens), (100, 200, 300));
    assert_eq!(input_token_details.text_tokens, 60);
    assert_eq!(input_token_details.audio_tokens, 40);
    let duration = usage_input(&fixture()["wire_usage"][9]["payload"]);
    let UsageInput::Transcription { usage } = duration else {
        panic!("transcription tag selects transcription payload");
    };
    let TranscriptionUsageInput::Duration { seconds } = usage else {
        panic!("type tag selects the duration variant");
    };
    assert_eq!(seconds, 12.5);
}

fn wire_response_rows() -> Vec<Value> {
    let fixture = fixture();
    fixture["wire_response"]
        .as_array()
        .expect("wire_response rows")
        .clone()
}

#[rstest]
fn wire_response_rows_deserialize_into_typed_structs() {
    let rows = wire_response_rows();
    let chat = serde_json::from_value::<ResponseInput>(rows[0]["response"].clone())
        .expect("chat response deserializes");
    assert_eq!(chat.model.as_deref(), Some("synthetic-chat"));
    assert_eq!(chat.created, Some(1_774_000_000.0));
    assert_eq!(chat.ended, Some(1_774_000_012.5));
    assert_eq!(chat.response_ms, Some(1_234.5));
    let UsageInput::Chat { usage } = chat.usage.expect("usage present") else {
        panic!("nested usage stays chat");
    };
    assert_eq!(usage.prompt_tokens, 2_000);
    assert_eq!(
        usage.server_tool_use.and_then(|s| s.web_search_requests),
        Some(1)
    );
    assert_eq!(usage.inference_geo.as_deref(), Some("us"));

    let hidden = serde_json::from_value::<HiddenParamsInput>(rows[0]["hidden_params"].clone())
        .expect("hidden params deserialize");
    assert_eq!(hidden.custom_llm_provider.as_deref(), Some("openai"));
    assert_eq!(hidden.region_name.as_deref(), Some("us-west-2"));
    assert_eq!(
        hidden.litellm_model_name.as_deref(),
        Some("synthetic-chat-alias")
    );
    let header = hidden
        .additional_headers
        .as_ref()
        .and_then(|headers| headers.get("llm_provider-x-litellm-response-cost"))
        .cloned();
    assert_eq!(header.and_then(WireNumber::cost), Some(0.0123));
    assert_eq!(
        hidden
            .provider_specific_fields
            .and_then(|fields| fields.traffic_type),
        Some("batch".to_string())
    );

    let optional =
        serde_json::from_value::<OptionalParamsInput>(rows[0]["optional_params"].clone())
            .expect("optional params deserialize");
    assert_eq!(optional.service_tier.as_deref(), Some("priority"));
    assert_eq!(optional.query.map(QueryCount::get), Some(2));

    let realtime = serde_json::from_value::<ResponseInput>(rows[1]["response"].clone())
        .expect("realtime response deserializes");
    assert_eq!(realtime.results.len(), 1);
    let result_response = realtime.results[0]
        .response
        .as_ref()
        .expect("response present");
    assert_eq!(result_response.service_tier.as_deref(), Some("priority"));
    let UsageInput::Responses { usage } = result_response.usage.clone().expect("usage present")
    else {
        panic!("realtime nested usage is responses-shaped");
    };
    assert_eq!(usage.input_tokens, 500);
    assert_eq!(
        usage.input_token_details.as_ref().map(|d| d.cached_tokens),
        Some(100)
    );

    let image = serde_json::from_value::<ImageResponseInput>(rows[2]["response"].clone())
        .expect("image response deserializes");
    assert_eq!(image.data.get(), 2);
    let usage = image.usage.expect("image usage present");
    assert_eq!(
        (usage.input_tokens, usage.output_tokens, usage.total_tokens,),
        (100, 50, 150)
    );
    assert_eq!(
        usage
            .input_tokens_details
            .as_ref()
            .and_then(|d| d.image_tokens),
        Some(60)
    );
    assert_eq!(
        usage
            .output_tokens_details
            .as_ref()
            .and_then(|d| d.image_tokens),
        Some(40)
    );
}

#[rstest]
fn chat_usage_input_leniently_accepts_bool_string_and_integral_float_counts() {
    let usage: ChatUsageInput = serde_json::from_value(serde_json::json!({
        "prompt_tokens": true,
        "completion_tokens": " 42 ",
        "total_tokens": 58.0,
        "cache_read_input_tokens": null,
    }))
    .expect("lenient counts parse");
    assert_eq!(usage.prompt_tokens, 1);
    assert_eq!(usage.completion_tokens, 42);
    assert_eq!(usage.total_tokens, 58);
    assert_eq!(usage.cache_read_input_tokens, None);
}
