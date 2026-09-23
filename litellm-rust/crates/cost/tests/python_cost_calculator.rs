use litellm_cost::Rate;
use litellm_cost::non_token::{
    ImageRates, ImageUsage, OcrBatchRates, OcrRates, OcrUsage, Unit, VideoRates, calculate_image,
    calculate_ocr_batch, calculate_ocr_with_tables, calculate_video,
};
use rstest::rstest;

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
