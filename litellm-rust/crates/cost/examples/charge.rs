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
}
