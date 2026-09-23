use litellm_cost::non_token::{
    ImageRates, ImageUsage, OcrBatchRates, OcrRates, OcrUsage, VideoRates, calculate_image,
    calculate_ocr, calculate_ocr_batch, calculate_video,
};
use litellm_cost::{
    Pricing, PromptConvention, Rate, Rates, Request, ServiceTier, ThresholdPolicy, Usage, compile,
};

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
}
