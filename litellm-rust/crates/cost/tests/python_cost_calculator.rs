use std::collections::HashMap;

use litellm_cost::batch::{
    BatchCostRates, BatchPricing, BatchTier, BatchUsage, ModalityRates, batch_cost_calculator,
    batch_cost_from_model_info, get_batch_cost_rates,
};
use litellm_cost::catalog::{DefaultImageCostRequest, ModelInfoCatalog};
use litellm_cost::non_token::{
    ImageRates, ImageUsage, OcrBatchRates, OcrRates, OcrUsage, Unit, VideoRates, calculate_image,
    calculate_ocr_batch, calculate_ocr_with_tables, calculate_video,
};
use litellm_cost::responses_usage::{ChatUsage, PromptTokenDetails};
use litellm_cost::{Rate, ThresholdPolicy};
use rstest::rstest;
use serde_json::json;

fn ocr_rates(credit: Rate, page: Rate, annotation: Rate) -> OcrRates {
    OcrRates {
        per_credit: credit,
        per_page: page,
        per_annotation_page: annotation,
    }
}

fn ocr_usage(pages: u64, annotation_pages: u64, credits: Option<f64>) -> OcrUsage {
    OcrUsage {
        credits,
        pages,
        annotation_pages,
    }
}

#[rstest]
#[case(1, 0.004)]
#[case(3, 0.012)]
#[case(10, 0.04)]
fn ocr_cost_uses_deployment_per_page_pricing_for_unmapped_model(
    #[case] pages: u64,
    #[case] expected: f64,
) {
    let deployment = ocr_rates(Rate::Missing, Rate::Value(0.004), Rate::Missing);
    let cost = calculate_ocr_with_tables(&[deployment], ocr_usage(pages, 0, None)).unwrap();
    assert!((cost.total - expected).abs() < 1e-12);
}

#[rstest]
#[case(
    ocr_rates(Rate::Missing, Rate::Missing, Rate::Value(0.01)),
    ocr_rates(Rate::Missing, Rate::Missing, Rate::Missing),
    0.02
)]
#[case(
    ocr_rates(Rate::Missing, Rate::Missing, Rate::Value(0.01)),
    ocr_rates(Rate::Missing, Rate::Value(0.004), Rate::Missing),
    0.032
)]
fn ocr_cost_annotation_pricing_keeps_independent_page_rate(
    #[case] deployment: OcrRates,
    #[case] published: OcrRates,
    #[case] expected: f64,
) {
    let cost = calculate_ocr_with_tables(&[deployment, published], ocr_usage(3, 2, None)).unwrap();
    assert!((cost.total - expected).abs() < 1e-12);
}

#[rstest]
#[case(Some(4.0), 1.0, Unit::Credit)]
#[case(None, 0.008, Unit::Page)]
fn ocr_cost_uses_credits_only_when_response_reports_them(
    #[case] credits: Option<f64>,
    #[case] expected: f64,
    #[case] unit: Unit,
) {
    let deployment = ocr_rates(Rate::Value(0.25), Rate::Missing, Rate::Missing);
    let published = ocr_rates(Rate::Missing, Rate::Value(0.004), Rate::Missing);
    let cost =
        calculate_ocr_with_tables(&[deployment, published], ocr_usage(2, 0, credits)).unwrap();
    assert_eq!(cost.components[0].unit, unit);
    assert!((cost.total - expected).abs() < 1e-12);
}

#[rstest]
#[case(Rate::Value(0.05), 0.1)]
#[case(Rate::Missing, 0.008)]
#[case(Rate::Value(0.0), 0.0)]
fn ocr_cost_deployment_rate_overrides_or_falls_through_to_cost_map(
    #[case] deployment_rate: Rate,
    #[case] expected: f64,
) {
    let deployment = ocr_rates(Rate::Missing, deployment_rate, Rate::Missing);
    let published = ocr_rates(Rate::Missing, Rate::Value(0.004), Rate::Missing);
    let cost = calculate_ocr_with_tables(&[deployment, published], ocr_usage(2, 0, None)).unwrap();
    assert!((cost.total - expected).abs() < 1e-12);
}

#[rstest]
fn ocr_cost_unmapped_model_without_deployment_pricing_bills_zero() {
    let cost = calculate_ocr_with_tables(&[], ocr_usage(5, 0, None)).unwrap();
    assert_eq!(cost.total, 0.0);
    assert!(cost.components.is_empty());
}

#[rstest]
#[case(
    Rate::Missing,
    Rate::Missing,
    Rate::Value(10.0),
    Unit::Pixel,
    10_485_760.0
)]
#[case(Rate::Value(0.08), Rate::Missing, Rate::Value(10.0), Unit::Image, 0.08)]
#[case(Rate::Missing, Rate::Value(0.04), Rate::Value(10.0), Unit::Image, 0.04)]
fn default_image_cost_calculator_selects_first_priced_unit(
    #[case] input_image: Rate,
    #[case] output_image: Rate,
    #[case] input_pixel: Rate,
    #[case] unit: Unit,
    #[case] expected: f64,
) {
    let rates = ImageRates {
        input_per_image: input_image,
        output_per_image: output_image,
        input_per_pixel: input_pixel,
    };
    let usage = ImageUsage {
        count: 1,
        width: 1024,
        height: 1024,
    };
    let cost = calculate_image(&[rates], usage).unwrap();
    assert_eq!(cost.components[0].unit, unit);
    assert_eq!(cost.total, expected);
}

fn batch_rates(
    page_batch: Rate,
    page: Rate,
    annotation_batch: Rate,
    annotation: Rate,
) -> OcrBatchRates {
    OcrBatchRates {
        per_page_batch: page_batch,
        per_page: page,
        per_annotation_page_batch: annotation_batch,
        per_annotation_page: annotation,
    }
}

#[rstest]
#[case(
    batch_rates(Rate::Value(0.0123), Rate::Value(0.0456), Rate::Missing, Rate::Missing),
    batch_rates(
        Rate::Value(0.003),
        Rate::Value(0.01),
        Rate::Value(0.007),
        Rate::Value(0.02)
    ),
    3,
    2,
    0.0509
)]
#[case(
    batch_rates(Rate::Missing, Rate::Value(0.04), Rate::Missing, Rate::Missing),
    batch_rates(Rate::Value(0.003), Rate::Missing, Rate::Value(0.007), Rate::Missing),
    3,
    2,
    0.134
)]
#[case(
    batch_rates(Rate::Value(0.0), Rate::Value(0.04), Rate::Missing, Rate::Missing),
    batch_rates(Rate::Value(0.003), Rate::Missing, Rate::Missing, Rate::Missing),
    3,
    2,
    0.0
)]
fn ocr_batch_cost_selects_each_family_from_deployment_then_published(
    #[case] deployment: OcrBatchRates,
    #[case] published: OcrBatchRates,
    #[case] pages: u64,
    #[case] annotation_pages: u64,
    #[case] expected: f64,
) {
    let cost =
        calculate_ocr_batch(Some(deployment), Some(published), pages, annotation_pages).unwrap();
    assert!((cost.total - expected).abs() < 1e-12);
}

#[rstest]
#[case(None, None, 0.5)]
#[case(Some("1080p"), None, 0.8)]
#[case(Some(" 1080P "), None, 0.8)]
#[case(Some("unknown"), None, 0.5)]
#[case(Some("1080p"), Some(0.2), 2.0)]
fn default_video_cost_calculator_selects_video_then_resolution_then_base_rate(
    #[case] resolution: Option<&str>,
    #[case] video_rate: Option<f64>,
    #[case] expected: f64,
) {
    let rates = VideoRates {
        per_video_second: video_rate.map_or(Rate::Missing, Rate::Value),
        per_second: Rate::Value(0.05),
        resolution_rates: &[("1080p", Rate::Value(0.08))],
    };
    let cost = calculate_video(&rates, 10.0, resolution).unwrap();
    assert_eq!(cost.components[0].unit, Unit::Second);
    assert!((cost.total - expected).abs() < 1e-12);
}

#[rstest]
fn default_video_cost_calculator_returns_zero_for_known_model_without_a_rate() {
    let rates = VideoRates {
        per_video_second: Rate::Missing,
        per_second: Rate::Missing,
        resolution_rates: &[],
    };
    assert_eq!(
        calculate_video(&rates, 5.0, Some("1080p")).unwrap().total,
        0.0
    );
}

fn token_rates(
    input: Rate,
    output: Rate,
    cache_read: Rate,
    cache_creation: Rate,
) -> BatchCostRates {
    BatchCostRates {
        input,
        output,
        cache_read,
        cache_creation,
    }
}

fn batch_usage(prompt_tokens: u64, completion_tokens: u64) -> BatchUsage {
    BatchUsage {
        prompt_tokens,
        completion_tokens,
        cache_read_tokens: 0,
        cache_creation_tokens: 0,
        audio_tokens: 0,
        image_tokens: 0,
        video_tokens: 0,
    }
}

fn batch_pricing<'a>(
    batch: BatchCostRates,
    regular: BatchCostRates,
    tiers: &'a [BatchTier],
) -> BatchPricing<'a> {
    BatchPricing {
        batch,
        regular,
        modalities: ModalityRates {
            audio: Rate::Missing,
            image: Rate::Missing,
            video: Rate::Missing,
        },
        tiers,
        threshold_policy: ThresholdPolicy::Exclusive,
        regional_uplift: 1.0,
    }
}

#[rstest]
#[case(Rate::Value(0.0), 0.0, 0.0)]
#[case(Rate::Value(1e-6), 0.001, 0.0005)]
#[case(Rate::Missing, 0.0015, 0.00375)]
fn batch_cost_calculator_preserves_zero_and_half_price_fallback(
    #[case] batch_rate: Rate,
    #[case] expected_prompt: f64,
    #[case] expected_completion: f64,
) {
    let pricing = batch_pricing(
        token_rates(batch_rate, batch_rate, Rate::Missing, Rate::Missing),
        token_rates(
            Rate::Value(3e-6),
            Rate::Value(15e-6),
            Rate::Missing,
            Rate::Missing,
        ),
        &[],
    );
    let cost = batch_cost_calculator(&pricing, batch_usage(1000, 500)).unwrap();
    assert!((cost.prompt - expected_prompt).abs() < 1e-12);
    assert!((cost.completion - expected_completion).abs() < 1e-12);
}

#[rstest]
#[case(Rate::Value(3.75e-6), (1000.0 * 3e-6 + 8000.0 * 3e-7 + 2000.0 * 3.75e-6) / 2.0)]
#[case(Rate::Missing, (1000.0 * 3e-6 + 8000.0 * 3e-7 + 2000.0 * 3e-6) / 2.0)]
fn batch_cost_calculator_prices_cache_creation_in_regular_fallback(
    #[case] creation_rate: Rate,
    #[case] expected: f64,
) {
    let pricing = batch_pricing(
        BatchCostRates::EMPTY,
        token_rates(
            Rate::Value(3e-6),
            Rate::Value(15e-6),
            Rate::Value(3e-7),
            creation_rate,
        ),
        &[],
    );
    let usage = BatchUsage {
        cache_read_tokens: 8000,
        cache_creation_tokens: 2000,
        ..batch_usage(11000, 200)
    };
    let cost = batch_cost_calculator(&pricing, usage).unwrap();
    assert!((cost.prompt - expected).abs() < 1e-12);
    assert!((cost.completion - 200.0 * 15e-6 / 2.0).abs() < 1e-12);
}

#[rstest]
fn batch_cost_calculator_prices_modalities_and_cached_tokens_together() {
    let tiers = [BatchTier {
        above_prompt_tokens: 272_000,
        rates: token_rates(
            Rate::Value(3e-6),
            Rate::Missing,
            Rate::Value(3e-7),
            Rate::Missing,
        ),
    }];
    let pricing = BatchPricing {
        modalities: ModalityRates {
            audio: Rate::Value(5e-6),
            image: Rate::Missing,
            video: Rate::Missing,
        },
        ..batch_pricing(
            token_rates(
                Rate::Value(1e-6),
                Rate::Missing,
                Rate::Value(1e-7),
                Rate::Missing,
            ),
            BatchCostRates::EMPTY,
            &tiers,
        )
    };
    let usage = BatchUsage {
        cache_read_tokens: 1000,
        audio_tokens: 64,
        image_tokens: 10,
        ..batch_usage(300_000, 0)
    };
    let cost = batch_cost_calculator(&pricing, usage).unwrap();
    let expected = 298_926.0 * 3e-6 + 64.0 * 5e-6 + 10.0 * 3e-6 + 1000.0 * 3e-7;
    assert!((cost.prompt - expected).abs() < 1e-12);
}

#[rstest]
#[case(272_000, 272_000.0 * 1e-6, 64.0 * 4e-6)]
#[case(300_035, 300_035.0 * 3e-6, 64.0 * 7e-6)]
fn batch_cost_calculator_selects_long_context_tier(
    #[case] prompt_tokens: u64,
    #[case] expected_prompt: f64,
    #[case] expected_completion: f64,
) {
    let tiers = [BatchTier {
        above_prompt_tokens: 272_000,
        rates: token_rates(
            Rate::Value(3e-6),
            Rate::Value(7e-6),
            Rate::Missing,
            Rate::Missing,
        ),
    }];
    let pricing = batch_pricing(
        token_rates(
            Rate::Value(1e-6),
            Rate::Value(4e-6),
            Rate::Missing,
            Rate::Missing,
        ),
        BatchCostRates::EMPTY,
        &tiers,
    );
    let cost = batch_cost_calculator(&pricing, batch_usage(prompt_tokens, 64)).unwrap();
    assert!((cost.prompt - expected_prompt).abs() < 1e-12);
    assert!((cost.completion - expected_completion).abs() < 1e-12);
}

#[rstest]
fn get_batch_cost_rates_selects_each_field_at_its_own_threshold() {
    let tiers = [
        BatchTier {
            above_prompt_tokens: 100,
            rates: token_rates(
                Rate::Value(5.0),
                Rate::Missing,
                Rate::Value(7.0),
                Rate::Missing,
            ),
        },
        BatchTier {
            above_prompt_tokens: 200,
            rates: token_rates(
                Rate::Missing,
                Rate::Value(6.0),
                Rate::Missing,
                Rate::Value(8.0),
            ),
        },
    ];
    let pricing = batch_pricing(
        token_rates(
            Rate::Value(1.0),
            Rate::Value(2.0),
            Rate::Value(3.0),
            Rate::Value(4.0),
        ),
        BatchCostRates::EMPTY,
        &tiers,
    );
    assert_eq!(
        get_batch_cost_rates(&pricing, 300),
        token_rates(
            Rate::Value(5.0),
            Rate::Value(6.0),
            Rate::Value(7.0),
            Rate::Value(8.0)
        )
    );
}

#[rstest]
#[case(200_000, 1e-6, 4e-6)]
#[case(250_000, 1e-6, 5e-6)]
#[case(300_000, 2e-6, 5e-6)]
fn batch_model_info_crosses_input_and_output_thresholds_independently(
    #[case] prompt_tokens: u64,
    #[case] input_rate: f64,
    #[case] output_rate: f64,
) {
    let model_info = json!({
        "input_cost_per_token_batches": 1e-6,
        "input_cost_per_token_above_272k_tokens_batches": 2e-6,
        "output_cost_per_token_batches": 4e-6,
        "output_cost_per_token_above_200k_tokens_batches": 5e-6
    });
    let usage = ChatUsage {
        prompt_tokens,
        completion_tokens: 1,
        total_tokens: prompt_tokens + 1,
        ..ChatUsage::default()
    };
    let cost = batch_cost_from_model_info(&model_info, &usage, Some("openai"), None).unwrap();
    assert!((cost.prompt - prompt_tokens as f64 * input_rate).abs() < 1e-12);
    assert_eq!(cost.completion, output_rate);
}

#[rstest]
fn batch_model_info_bills_cache_writes_as_input_without_a_batch_cache_rate() {
    let model_info = json!({
        "input_cost_per_token_batches": 1e-7,
        "input_cost_per_token_above_272k_tokens_batches": 2e-7,
        "cache_creation_input_token_cost": 2.5e-7,
        "cache_creation_input_token_cost_above_272k_tokens": 5e-7
    });
    let usage = ChatUsage {
        prompt_tokens: 300_000,
        prompt_tokens_details: Some(PromptTokenDetails {
            cache_creation_tokens: Some(100_000),
            ..PromptTokenDetails::default()
        }),
        ..ChatUsage::default()
    };
    let cost = batch_cost_from_model_info(&model_info, &usage, Some("openai"), None).unwrap();
    assert_eq!(cost.prompt, 300_000.0 * 2e-7);
}

#[rstest]
#[case(1000, 900, false, 100.0 * 1e-6 + 900.0 * 1e-7)]
#[case(300_048, 300_045, false, 3.0 * 3e-6 + 300_045.0 * 3e-7)]
#[case(1000, 900, true, 100.0 * 1e-6 + 900.0 * 1.25e-6)]
#[case(300_048, 300_045, true, 3.0 * 3e-6 + 300_045.0 * 3.75e-6)]
fn batch_cost_calculator_bills_cache_at_the_selected_batch_tier(
    #[case] prompt_tokens: u64,
    #[case] cache_tokens: u64,
    #[case] is_write: bool,
    #[case] expected: f64,
) {
    let tiers = [BatchTier {
        above_prompt_tokens: 272_000,
        rates: token_rates(
            Rate::Value(3e-6),
            Rate::Missing,
            Rate::Value(3e-7),
            Rate::Value(3.75e-6),
        ),
    }];
    let pricing = batch_pricing(
        token_rates(
            Rate::Value(1e-6),
            Rate::Missing,
            Rate::Value(1e-7),
            Rate::Value(1.25e-6),
        ),
        BatchCostRates::EMPTY,
        &tiers,
    );
    let usage = BatchUsage {
        cache_read_tokens: if is_write { 0 } else { cache_tokens },
        cache_creation_tokens: if is_write { cache_tokens } else { 0 },
        ..batch_usage(prompt_tokens, 0)
    };
    let cost = batch_cost_calculator(&pricing, usage).unwrap();
    assert!((cost.prompt - expected).abs() < 1e-12);
}

#[rstest]
fn batch_cost_calculator_applies_regional_uplift_to_both_directions() {
    let pricing = BatchPricing {
        regional_uplift: 1.1,
        ..batch_pricing(
            token_rates(
                Rate::Value(1e-6),
                Rate::Value(4e-6),
                Rate::Missing,
                Rate::Missing,
            ),
            BatchCostRates::EMPTY,
            &[],
        )
    };
    let cost = batch_cost_calculator(&pricing, batch_usage(1000, 500)).unwrap();
    assert!((cost.prompt - 0.0011).abs() < 1e-12);
    assert!((cost.completion - 0.0022).abs() < 1e-12);
}

#[rstest]
#[case(Some("low"), Some("1024x1024"), 0.04)]
#[case(Some("low"), Some("1536-x-1024"), 0.05)]
#[case(Some("high"), Some("1024x1024"), 0.08)]
fn default_image_cost_calculator_selects_quality_and_normalized_size(
    #[case] quality: Option<&str>,
    #[case] size: Option<&str>,
    #[case] expected: f64,
) {
    let catalog = ModelInfoCatalog::new(HashMap::from([
        (
            "xai/model".to_owned(),
            json!({"input_cost_per_image": 0.06}),
        ),
        (
            "low/1024-x-1024/model".to_owned(),
            json!({"input_cost_per_image": 0.04}),
        ),
        (
            "low/1536-x-1024/model".to_owned(),
            json!({"input_cost_per_image": 0.05}),
        ),
        (
            "high/1024-x-1024/model".to_owned(),
            json!({"input_cost_per_image": 0.08}),
        ),
    ]));
    let cost = catalog
        .default_image_cost_calculator(DefaultImageCostRequest {
            model: "xai/model",
            provider: Some("xai"),
            quality,
            n: Some(1),
            size,
            supplied_model_info: None,
        })
        .unwrap();
    assert!((cost - expected).abs() < 1e-12);
}

#[rstest]
fn default_image_cost_calculator_prefers_deployment_rate_and_explicit_zero() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "openai/dall-e".to_owned(),
        json!({"output_cost_per_image": 0.03}),
    )]));
    for (rate, expected) in [(0.07, 0.14), (0.0, 0.0)] {
        let supplied = json!({"input_cost_per_image": rate});
        let cost = catalog
            .default_image_cost_calculator(DefaultImageCostRequest {
                model: "dall-e",
                provider: Some("openai"),
                quality: Some("standard"),
                n: Some(2),
                size: Some("1024x1024"),
                supplied_model_info: Some(&supplied),
            })
            .unwrap();
        assert_eq!(cost, expected);
    }
}

#[rstest]
fn default_image_cost_calculator_prices_pixels_after_image_rates() {
    let catalog = ModelInfoCatalog::new(HashMap::from([(
        "provider/model".to_owned(),
        json!({"input_cost_per_pixel": 0.0001}),
    )]));
    let cost = catalog
        .default_image_cost_calculator(DefaultImageCostRequest {
            model: "provider/model",
            provider: Some("provider"),
            quality: None,
            n: Some(2),
            size: Some("20x10"),
            supplied_model_info: None,
        })
        .unwrap();
    assert!((cost - 2.0 * 20.0 * 10.0 * 0.0001).abs() < 1e-12);
}
