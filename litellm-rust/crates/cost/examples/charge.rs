use std::collections::{BTreeMap, HashMap};

use jiff::Timestamp;
use litellm_cost::batch::{
    BatchCostRates, BatchPricing, BatchUsage, ModalityRates, batch_cost_calculator,
};
use litellm_cost::catalog::{
    BuiltInToolCharge, CompletionCostRequest, CostCatalog, ModelCostRequest, ModelInfoCatalog,
    ResponseCostRequest,
};
use litellm_cost::custom_pricing::{
    CustomPricing, CustomTokenRates, RawUsage, cost_per_token_custom_pricing_helper,
    normalize_cache_usage,
};
use litellm_cost::gemini_cost::cost_per_web_search_request;
use litellm_cost::generic_cost::{
    ResolvedTokenRates, calculate_generic_cost_from_model_info,
    calculate_generic_cost_from_model_info_with_region,
    calculate_generic_cost_from_model_info_without_off_peak,
    calculate_generic_cost_with_resolved_rates,
};
use litellm_cost::generic_input::{InputBaseRates, calculate_input_cost};
use litellm_cost::generic_usage::parse_prompt_tokens_details;
use litellm_cost::guardrail_cost::bedrock_guardrail_cost;
use litellm_cost::non_token::{
    ImageRates, ImageUsage, OcrBatchRates, OcrRates, OcrUsage, VideoRates, calculate_image,
    calculate_ocr, calculate_ocr_batch, calculate_video,
};
use litellm_cost::responses_usage::transform_response_api_usage_to_chat_usage;
use litellm_cost::tiered_pricing::{select_tier_for_input, tier_rate};
use litellm_cost::tool_call_cost_tracking::{
    DefaultToolRates, ResponseKind, get_cost_for_file_search,
};
use litellm_cost::tool_cost_dispatch::BuiltInToolCostRequest;
use litellm_cost::usage_dispatch::get_usage_object;
use litellm_cost::{
    Pricing, PromptConvention, Rate, Rates, Request, ServiceTier, ThresholdPolicy, Usage,
    calculate, compile,
};
use serde_json::json;

fn main() {
    let pricing = Pricing {
        standard: Rates {
            input: Rate::Value(2.0),
            output: Rate::Value(4.0),
            cache_read: Rate::Value(0.5),
            cache_write: Rate::Value(3.0),
            cache_write_1h: Rate::Missing,
        },
        tiers: &[],
        thresholds: &[],
        off_peak: None,
    };
    let request = Request {
        usage: Usage {
            prompt_tokens: 100,
            completion_tokens: 20,
            cache_read_tokens: 25,
            cache_write_tokens: 10,
            cache_write_5m_tokens: None,
            cache_write_1h_tokens: None,
            prompt_convention: PromptConvention::IncludesCache,
        },
        service_tier: ServiceTier::Standard,
        threshold_policy: ThresholdPolicy::Exclusive,
        region_multiplier: None,
        billed_at_utc_minute: None,
    };
    let cost = compile(&pricing).unwrap().calculate(&request).unwrap();
    println!(
        "input={} output={} total={}",
        cost.input(),
        cost.output(),
        cost.total()
    );
    let image = calculate_image(
        &[ImageRates {
            input_per_image: Rate::Value(0.04),
            output_per_image: Rate::Missing,
            input_per_pixel: Rate::Missing,
        }],
        ImageUsage {
            count: 2,
            width: 1024,
            height: 1024,
        },
    )
    .unwrap();
    let ocr = calculate_ocr(
        OcrRates {
            per_credit: Rate::Missing,
            per_page: Rate::Value(0.01),
            per_annotation_page: Rate::Missing,
        },
        OcrUsage {
            credits: None,
            pages: 3,
            annotation_pages: 0,
        },
    )
    .unwrap();
    println!("image={} ocr={}", image.total, ocr.total);
    let video = calculate_video(
        &VideoRates {
            per_video_second: Rate::Missing,
            per_second: Rate::Value(0.05),
            resolution_rates: &[("1080p", Rate::Value(0.08))],
        },
        10.0,
        Some("1080p"),
    )
    .unwrap();
    let ocr_batch = calculate_ocr_batch(
        Some(OcrBatchRates {
            per_page_batch: Rate::Value(0.012),
            per_page: Rate::Value(0.04),
            per_annotation_page_batch: Rate::Missing,
            per_annotation_page: Rate::Missing,
        }),
        None,
        3,
        0,
    )
    .unwrap();
    println!("video={} ocr_batch={}", video.total, ocr_batch.total);
    let batch = batch_cost_calculator(
        &BatchPricing {
            batch: BatchCostRates {
                input: Rate::Value(0.000001),
                output: Rate::Value(0.000004),
                cache_read: Rate::Missing,
                cache_creation: Rate::Missing,
            },
            regular: BatchCostRates::EMPTY,
            modalities: ModalityRates {
                audio: Rate::Missing,
                image: Rate::Missing,
                video: Rate::Missing,
            },
            tiers: &[],
            threshold_policy: ThresholdPolicy::Exclusive,
            regional_uplift: 1.0,
        },
        BatchUsage {
            prompt_tokens: 1000,
            completion_tokens: 500,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
            audio_tokens: 0,
            image_tokens: 0,
            video_tokens: 0,
        },
    )
    .unwrap();
    println!(
        "batch_input={} batch_output={}",
        batch.prompt, batch.completion
    );
    let normalized = normalize_cache_usage(RawUsage {
        prompt_tokens: 1000.0,
        completion_tokens: 100.0,
        details_cached_tokens: Some(400.0),
        details_cache_write_tokens: None,
        details_cache_creation_tokens: None,
        top_level_cache_read_tokens: None,
        top_level_cache_creation_tokens: None,
        fallback_cache_read_tokens: None,
        fallback_cache_creation_tokens: None,
    })
    .unwrap();
    let custom = cost_per_token_custom_pricing_helper(
        normalized,
        CustomPricing {
            token: Some(CustomTokenRates {
                input: 0.0000025,
                output: 0.000015,
                cache_read: Some(0.00000025),
                cache_creation: None,
            }),
            per_second: None,
        },
        None,
    )
    .unwrap()
    .unwrap();
    println!(
        "custom_input={} custom_output={}",
        custom.input, custom.output
    );
    let catalog = CostCatalog::new(HashMap::from([(
        "bedrock_mantle/us-gov-west-1/model".to_owned(),
        Rates {
            input: Rate::Value(5e-6),
            output: Rate::Value(6e-6),
            ..Rates::EMPTY
        },
    )]));
    let catalog_request = Request {
        usage: Usage {
            completion_tokens: 50,
            ..request.usage
        },
        ..request
    };
    let catalog_cost = catalog
        .cost_per_token(
            "bedrock_mantle/model",
            Some("bedrock_mantle"),
            Some("us-gov-west-1"),
            &catalog_request,
        )
        .unwrap();
    println!(
        "catalog_input={:.4} catalog_output={:.4}",
        catalog_cost.input(),
        catalog_cost.output()
    );
    let response_usage = transform_response_api_usage_to_chat_usage(&json!({
        "input_tokens": 1000,
        "output_tokens": 100,
        "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 100}
    }))
    .unwrap();
    let response_pricing = Pricing {
        standard: Rates {
            input: Rate::Value(1e-6),
            output: Rate::Value(2e-6),
            cache_read: Rate::Value(0.2e-6),
            cache_write: Rate::Value(1.25e-6),
            cache_write_1h: Rate::Missing,
        },
        tiers: &[],
        thresholds: &[],
        off_peak: None,
    };
    let response_request = Request {
        usage: response_usage.token_usage(),
        ..request
    };
    let response_cost = calculate(&response_pricing, &response_request).unwrap();
    println!(
        "responses_input={:.6} responses_output={:.6}",
        response_cost.input(),
        response_cost.output()
    );
    let anthropic_usage = get_usage_object(&json!({
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 25
        }
    }))
    .unwrap()
    .unwrap();
    println!(
        "anthropic_prompt={} cached={} write={}",
        anthropic_usage.prompt_tokens,
        anthropic_usage.token_usage().cache_read_tokens,
        anthropic_usage.token_usage().cache_write_tokens
    );
    let interactions_usage = get_usage_object(&json!({
        "usage": {
            "total_input_tokens": 100,
            "total_cached_tokens": 20,
            "input_tokens_by_modality": [{"modality": "text", "tokens": 100}],
            "total_output_tokens": 30,
            "total_thought_tokens": 5
        }
    }))
    .unwrap()
    .unwrap();
    println!(
        "interactions_prompt={} cached={} completion={}",
        interactions_usage.prompt_tokens,
        interactions_usage.token_usage().cache_read_tokens,
        interactions_usage.completion_tokens
    );
    let transcription_usage = get_usage_object(&json!({
        "usage": {
            "type": "tokens",
            "input_tokens": 20,
            "output_tokens": 5,
            "total_tokens": 25,
            "input_token_details": {"text_tokens": 4, "audio_tokens": 16}
        }
    }))
    .unwrap()
    .unwrap();
    println!(
        "transcription_prompt={} audio={} completion={}",
        transcription_usage.prompt_tokens,
        transcription_usage
            .prompt_tokens_details
            .unwrap()
            .audio_tokens
            .unwrap(),
        transcription_usage.completion_tokens
    );
    let guardrail = bedrock_guardrail_cost(
        &BTreeMap::from([("contentPolicyUnits".to_owned(), 2)]),
        Some("us-east-1"),
        &BTreeMap::from([(
            "bedrock/guardrails".to_owned(),
            json!({"guardrail_cost_per_unit": {"contentPolicyUnits": 0.00015}}),
        )]),
    );
    println!("guardrail={guardrail:.5}");
    let tiers = [
        json!({"range": [0, 32_000], "input_cost_per_token": "4e-07"}),
        json!({"range": [32_000, 128_000], "input_cost_per_token": "8e-07"}),
    ];
    let tier = select_tier_for_input(&tiers, 32_001).unwrap();
    println!(
        "tier_input_rate={}",
        tier_rate(tier, "input_cost_per_token", None)
    );
    let gemini_usage = get_usage_object(&json!({
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"web_search_requests": 3}
        }
    }))
    .unwrap()
    .unwrap();
    let search_cost = cost_per_web_search_request(
        &gemini_usage,
        &json!({
            "web_search_billing_unit": "per_query",
            "search_context_cost_per_query": {"search_context_size_medium": 0.01}
        }),
    );
    println!("gemini_search={search_cost:.2}");
    let parsed = parse_prompt_tokens_details(
        &get_usage_object(&json!({
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "total_tokens": 1100,
                "prompt_tokens_details": {
                    "cached_tokens": 200,
                    "cached_tokens_details": {"audio_tokens": 80, "text_tokens": 90},
                    "text_tokens": 500,
                    "audio_tokens": 250
                }
            }
        }))
        .unwrap()
        .unwrap(),
    );
    println!(
        "parsed_input_text={} audio={} cached_audio={}",
        parsed.text_tokens, parsed.audio_tokens, parsed.cache_hit_audio_tokens
    );
    let generic_input = calculate_input_cost(
        &parsed,
        &json!({"input_cost_per_audio_token": 3e-6}),
        InputBaseRates {
            prompt: 2e-6,
            cache_read: 0.5e-6,
            cache_creation: 2.5e-6,
            cache_creation_above_1hr: 4e-6,
        },
        None,
    );
    println!("generic_input={generic_input:.5}");
    let complete_usage = get_usage_object(&json!({
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "prompt_tokens_details": {
                "cached_tokens": 200,
                "cache_write_tokens": 100,
                "text_tokens": 1000,
                "audio_tokens": 100,
                "image_tokens": 50,
                "video_tokens": 50
            }
        }
    }))
    .unwrap()
    .unwrap();
    let (generic_prompt, generic_output) = calculate_generic_cost_with_resolved_rates(
        &complete_usage,
        &json!({
            "input_cost_per_audio_token": 4e-6,
            "input_cost_per_image_token": 5e-6,
            "input_cost_per_video_token": 6e-6
        }),
        ResolvedTokenRates {
            input: InputBaseRates {
                prompt: 2e-6,
                cache_read: 0.5e-6,
                cache_creation: 3e-6,
                cache_creation_above_1hr: 3e-6,
            },
            output: 7e-6,
            reasoning: None,
            multiplier: 1.0,
        },
        None,
    );
    println!("generic_prompt={generic_prompt:.5} generic_output={generic_output:.5}");
    let threshold_usage = get_usage_object(&json!({
        "usage": {
            "prompt_tokens": 128_001,
            "completion_tokens": 100,
            "total_tokens": 128_101,
            "prompt_tokens_details": {"cached_tokens": 20_000},
            "completion_tokens_details": {"reasoning_tokens": 40}
        }
    }))
    .unwrap()
    .unwrap();
    let (threshold_prompt, threshold_output) =
        calculate_generic_cost_from_model_info_without_off_peak(
            &threshold_usage,
            &json!({
                "input_cost_per_token": 2e-6,
                "output_cost_per_token": 4e-6,
                "output_cost_per_reasoning_token": 8e-6,
                "input_cost_per_token_above_128k_tokens": 5e-6,
                "output_cost_per_token_above_128k_tokens": 7e-6,
                "cache_read_input_token_cost_above_128k_tokens": 1e-6
            }),
            None,
            false,
            1.0,
        );
    println!("threshold_prompt={threshold_prompt:.6} threshold_output={threshold_output:.5}");
    let off_peak_usage = get_usage_object(&json!({
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 100,
            "total_tokens": 200,
            "prompt_tokens_details": {"cached_tokens": 40},
            "completion_tokens_details": {"reasoning_tokens": 20}
        }
    }))
    .unwrap()
    .unwrap();
    let at: Timestamp = "2026-01-01T18:00Z".parse().unwrap();
    let (off_peak_prompt, off_peak_output) = calculate_generic_cost_from_model_info(
        &off_peak_usage,
        &json!({
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 4e-6,
            "output_cost_per_reasoning_token": 8e-6,
            "off_peak_pricing": {
                "hours_utc": "16:30-00:30",
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "output_cost_per_reasoning_token": 3e-6
            }
        }),
        None,
        false,
        1.0,
        at,
    );
    println!("off_peak_prompt={off_peak_prompt:.5} off_peak_output={off_peak_output:.5}");
    let tool_defaults = DefaultToolRates {
        file_search_per_call: 0.25,
        azure_file_search_per_gb_day: 0.1,
        azure_vector_store_per_gb_day: 0.2,
        azure_computer_input_per_1k_tokens: 3.0,
        azure_computer_output_per_1k_tokens: 12.0,
        code_interpreter_per_session: Some(0.03),
        xai_web_search_per_call: 0.005,
        groq_browser_open_per_call: 0.001,
    };
    let file_search_cost = get_cost_for_file_search(
        Some(&json!({"type": "file_search"})),
        Some("azure"),
        Some(&json!({"file_search_cost_per_gb_per_day": 0.4})),
        Some(1.5),
        Some(10.0),
        tool_defaults,
    );
    println!("file_search_cost={file_search_cost:.2}");
    let (regional_prompt, regional_output) = calculate_generic_cost_from_model_info_with_region(
        &off_peak_usage,
        &json!({
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 4e-6,
            "regional_processing_uplift_multiplier_eu": 1.2
        }),
        None,
        false,
        Some("eu"),
        None,
        at,
    );
    println!("regional_prompt={regional_prompt:.5} regional_output={regional_output:.5}");
    let model_info_catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "openai/model".to_owned(),
            json!({
                "input_cost_per_token": 2e-6,
                "output_cost_per_token": 4e-6,
                "search_context_cost_per_query": {"search_context_size_medium": 0.01},
                "regional_processing_uplift_multiplier_eu": 1.2,
                "off_peak_pricing": {
                    "hours_utc": "16:30-00:30",
                    "input_cost_per_token": 1e-6,
                    "output_cost_per_token": 2e-6,
                    "output_cost_per_reasoning_token": 3e-6
                }
            }),
        ),
        (
            "provider/duration".to_owned(),
            json!({"mode": "responses", "input_cost_per_second": 0.5, "output_cost_per_second": 1.0}),
        ),
        (
            "openai/cache_savings_model".to_owned(),
            json!({"input_cost_per_token": 2e-6, "output_cost_per_token": 4e-6, "cache_read_input_token_cost": 0.5e-6}),
        ),
        (
            "openai/image_model".to_owned(),
            json!({"input_cost_per_token": 5e-6, "input_cost_per_image_token": 8e-6, "output_cost_per_image_token": 3e-5}),
        ),
        (
            "perplexity/research".to_owned(),
            json!({
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 1e-6,
                "output_cost_per_reasoning_token": 3e-6,
                "citation_cost_per_token": 2e-6,
                "search_context_cost_per_query": {"search_context_size_low": 0.005},
                "off_peak_pricing": {"hours_utc": "16:30-00:30", "input_cost_per_token": 1e-7, "output_cost_per_token": 2e-7}
            }),
        ),
        (
            "vertex_ai/rerank".to_owned(),
            json!({"input_cost_per_query": 0.25}),
        ),
        (
            "vertex_ai/search_api".to_owned(),
            json!({"input_cost_per_query": 0.4}),
        ),
    ]));
    let (catalog_prompt, catalog_output) = model_info_catalog
        .cost_per_token(ModelCostRequest {
            model: "openai/model",
            provider: Some("openai"),
            region: None,
            usage: &off_peak_usage,
            service_tier: None,
            data_residency: Some("eu"),
            vertex_location: None,
            at,
            response_time_ms: None,
        })
        .unwrap();
    println!("catalog_prompt={catalog_prompt:.6} catalog_output={catalog_output:.6}");
    let (duration_prompt, duration_output) = model_info_catalog
        .cost_per_token(ModelCostRequest {
            model: "duration",
            provider: Some("provider"),
            region: None,
            usage: &off_peak_usage,
            service_tier: None,
            data_residency: None,
            vertex_location: None,
            at,
            response_time_ms: Some(2000.0),
        })
        .unwrap();
    println!("duration_prompt={duration_prompt:.1} duration_output={duration_output:.1}");
    let rerank = model_info_catalog.rerank_cost(
        "rerank",
        "vertex_ai",
        None,
        Some(&json!({"search_units": 3})),
    );
    let search = model_info_catalog.vector_store_search_cost("vertex_ai", Some("search_api"));
    println!("rerank={} vector_search={}", rerank.0, search.0);
    let discount = json!({"openai": 0.1});
    let margin = json!({"global": {"percentage": 0.2, "fixed_amount": 0.001}});
    let completion_response = json!({"output": [{"type": "web_search_call"}]});
    let tool_params = json!({});
    let tool_request = BuiltInToolCostRequest {
        response: &completion_response,
        response_kind: ResponseKind::Responses,
        usage: Some(&off_peak_usage),
        provider: Some("openai"),
        params: &tool_params,
        defaults: tool_defaults,
    };
    let completion_request = CompletionCostRequest {
        token: ModelCostRequest {
            model: "openai/model",
            provider: Some("openai"),
            region: None,
            usage: &off_peak_usage,
            service_tier: None,
            data_residency: Some("eu"),
            vertex_location: None,
            at,
            response_time_ms: None,
        },
        built_in_tools: BuiltInToolCharge::FromResponse(tool_request),
        additional_costs: &[0.02],
        discount_config: &discount,
        margin_config: &margin,
    };
    let completion_total = model_info_catalog
        .completion_cost(completion_request)
        .unwrap();
    let provider_total = model_info_catalog
        .response_cost_calculator(ResponseCostRequest {
            completion: completion_request,
            cache_hit: false,
            hidden_params: &json!({
                "additional_headers": {"llm_provider-x-litellm-response-cost": "0.031"}
            }),
        })
        .unwrap();
    println!(
        "completion_total={:.8} provider_total={provider_total:.3}",
        completion_total.total
    );
    let dispatched_tool_cost =
        model_info_catalog.built_in_tool_cost("openai/model", Some("openai"), None, tool_request);
    println!("dispatched_tool_cost={dispatched_tool_cost:.2}");
    let breakdown =
        model_info_catalog.get_token_type_cost_breakdown(completion_request.token, None);
    println!(
        "breakdown_reasoning={:.6} breakdown_cache={:.6}",
        breakdown.reasoning_cost, breakdown.cache_read_cost
    );
    let savings = model_info_catalog
        .calculate_prompt_caching_savings(ModelCostRequest {
            model: "cache_savings_model",
            ..completion_request.token
        })
        .unwrap();
    println!("prompt_caching_savings={savings:.6}");
    let image_usage_cost = model_info_catalog
        .calculate_image_response_cost_from_usage(
            "image_model",
            Some("openai"),
            None,
            &json!({"usage": {
                "input_tokens": 531,
                "output_tokens": 158,
                "total_tokens": 689,
                "input_tokens_details": {"text_tokens": 19, "image_tokens": 512},
                "output_tokens_details": {"image_tokens": 158}
            }}),
            at,
        )
        .unwrap();
    println!("image_usage_cost={image_usage_cost:.6}");
    let perplexity_usage = get_usage_object(&json!({"usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "total_tokens": 1200,
        "citation_tokens": 100,
        "prompt_tokens_details": {"web_search_requests": 2},
        "completion_tokens_details": {"reasoning_tokens": 50}
    }}))
    .unwrap()
    .unwrap();
    let (perplexity_prompt, perplexity_output) = model_info_catalog
        .cost_per_token(ModelCostRequest {
            model: "research",
            provider: Some("perplexity"),
            region: None,
            usage: &perplexity_usage,
            service_tier: None,
            data_residency: None,
            vertex_location: None,
            at,
            response_time_ms: None,
        })
        .unwrap();
    println!("perplexity_prompt={perplexity_prompt:.6} perplexity_output={perplexity_output:.6}");
}
