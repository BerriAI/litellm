use litellm_pricing::{
    OffPeakRates, Pricing, PricingError, PromptConvention, Rate, Rates, Request, ServiceTier,
    ThresholdPolicy, ThresholdRates, TierRates, Usage, calculate,
};

fn rates(input: Rate, output: Rate) -> Rates {
    Rates {
        input,
        output,
        ..Rates::EMPTY
    }
}

fn request() -> Request {
    Request {
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
    }
}

#[test]
fn ordinary_and_custom_cache_rates_match_python_arithmetic() {
    let standard = Rates {
        cache_read: Rate::Value(0.5),
        cache_write: Rate::Value(3.0),
        ..rates(Rate::Value(2.0), Rate::Value(4.0))
    };
    let pricing = Pricing {
        standard,
        tiers: &[],
        thresholds: &[],
        off_peak: None,
    };
    let cost = calculate(&pricing, &request()).unwrap();
    assert_eq!(cost.input, 65.0 * 2.0 + 25.0 * 0.5 + 10.0 * 3.0);
    assert_eq!(cost.output, 20.0 * 4.0);
}

#[test]
fn absent_null_and_explicit_zero_cache_rates_are_distinct() {
    let base = rates(Rate::Value(2.0), Rate::Value(4.0));
    for read in [Rate::Missing, Rate::Null] {
        let pricing = Pricing {
            standard: Rates {
                cache_read: read,
                ..base
            },
            tiers: &[],
            thresholds: &[],
            off_peak: None,
        };
        assert_eq!(calculate(&pricing, &request()).unwrap().input, 200.0);
    }
    let pricing = Pricing {
        standard: Rates {
            cache_read: Rate::Value(0.0),
            cache_write: Rate::Value(0.0),
            ..base
        },
        tiers: &[],
        thresholds: &[],
        off_peak: None,
    };
    assert_eq!(calculate(&pricing, &request()).unwrap().input, 130.0);
}

#[test]
fn excluded_cache_and_split_writes_use_distinct_rates() {
    let pricing = Pricing {
        standard: Rates {
            cache_read: Rate::Value(0.5),
            cache_write: Rate::Value(3.0),
            cache_write_1h: Rate::Value(5.0),
            ..rates(Rate::Value(2.0), Rate::Value(4.0))
        },
        tiers: &[],
        thresholds: &[],
        off_peak: None,
    };
    let mut request = request();
    request.usage.prompt_tokens = 65;
    request.usage.prompt_convention = PromptConvention::ExcludesCache;
    request.usage.cache_write_5m_tokens = Some(4);
    request.usage.cache_write_1h_tokens = Some(6);
    assert_eq!(
        calculate(&pricing, &request).unwrap().input,
        130.0 + 12.5 + 12.0 + 30.0
    );
    request.usage.prompt_convention = PromptConvention::IncludesCache;
    request.usage.prompt_tokens = 30;
    assert_eq!(
        calculate(&pricing, &request),
        Err(PricingError::CacheExceedsPrompt)
    );
    request.usage.cache_write_1h_tokens = Some(5);
    request.usage.prompt_tokens = 100;
    assert_eq!(
        calculate(&pricing, &request),
        Err(PricingError::InvalidCacheWriteDetails)
    );
}

#[test]
fn threshold_and_service_tier_boundaries() {
    let standard = rates(Rate::Value(2.0), Rate::Value(4.0));
    let tier = TierRates {
        tier: ServiceTier::Priority,
        rates: rates(Rate::Value(3.0), Rate::Missing),
    };
    let threshold = ThresholdRates {
        above_prompt_tokens: 100,
        standard: rates(Rate::Value(5.0), Rate::Value(8.0)),
        tier: Some(TierRates {
            tier: ServiceTier::Priority,
            rates: rates(Rate::Value(7.0), Rate::Missing),
        }),
    };
    let pricing = Pricing {
        standard,
        tiers: &[tier],
        thresholds: &[threshold],
        off_peak: None,
    };
    let mut request = request();
    request.usage.cache_read_tokens = 0;
    request.usage.cache_write_tokens = 0;
    assert_eq!(calculate(&pricing, &request).unwrap().input, 200.0);
    request.service_tier = ServiceTier::Fast;
    assert_eq!(calculate(&pricing, &request).unwrap().input, 300.0);
    request.threshold_policy = ThresholdPolicy::Inclusive;
    assert_eq!(
        calculate(&pricing, &request).unwrap(),
        litellm_pricing::Cost {
            input: 700.0,
            output: 160.0
        }
    );
    request.usage.prompt_tokens = 101;
    request.threshold_policy = ThresholdPolicy::Exclusive;
    assert_eq!(calculate(&pricing, &request).unwrap().input, 707.0);
}

#[test]
fn off_peak_and_region_apply_to_selected_rates() {
    let pricing = Pricing {
        standard: rates(Rate::Value(2.0), Rate::Value(4.0)),
        tiers: &[],
        thresholds: &[],
        off_peak: Some(OffPeakRates {
            start_utc_minute: 60,
            end_utc_minute: 120,
            rates: rates(Rate::Value(1.0), Rate::Value(2.0)),
        }),
    };
    let mut request = request();
    request.region_multiplier = Some(1.1);
    assert_eq!(
        calculate(&pricing, &request),
        Err(PricingError::InvalidBillingTime)
    );
    request.billed_at_utc_minute = Some(60);
    assert_eq!(
        calculate(&pricing, &request).unwrap().input,
        110.00000000000001
    );
    assert_eq!(calculate(&pricing, &request).unwrap().output, 44.0);
    request.billed_at_utc_minute = Some(120);
    assert_eq!(
        calculate(&pricing, &request).unwrap().input,
        220.00000000000003
    );
}

#[test]
fn missing_pricing_is_an_error_even_for_zero_tokens() {
    let pricing = Pricing {
        standard: Rates::EMPTY,
        tiers: &[],
        thresholds: &[],
        off_peak: None,
    };
    let mut request = request();
    request.usage.prompt_tokens = 0;
    request.usage.completion_tokens = 0;
    request.usage.cache_read_tokens = 0;
    request.usage.cache_write_tokens = 0;
    assert_eq!(
        calculate(&pricing, &request),
        Err(PricingError::MissingInputRate)
    );
    let pricing = Pricing {
        standard: rates(Rate::Value(0.0), Rate::Missing),
        ..pricing
    };
    assert_eq!(
        calculate(&pricing, &request),
        Err(PricingError::MissingOutputRate)
    );
    let pricing = Pricing {
        standard: rates(Rate::Value(0.0), Rate::Value(0.0)),
        ..pricing
    };
    assert_eq!(calculate(&pricing, &request).unwrap().input, 0.0);
}
