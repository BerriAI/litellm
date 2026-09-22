use litellm_cost::{
    OffPeakRates, Pricing, PricingError, PromptConvention, Rate, Rates, Request, ServiceTier,
    ThresholdPolicy, ThresholdRates, TierRates, Usage, calculate, compile,
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

fn pricing(standard: Rates) -> Pricing<'static> {
    Pricing {
        standard,
        tiers: &[],
        thresholds: &[],
        off_peak: None,
    }
}

#[test]
fn breakdown_and_total_agree() {
    let standard = Rates {
        cache_read: Rate::Value(0.5),
        cache_write: Rate::Value(3.0),
        ..rates(Rate::Value(2.0), Rate::Value(4.0))
    };
    let result = calculate(&pricing(standard), &request()).unwrap();
    assert_eq!(result.uncached_input, 65.0 * 2.0);
    assert_eq!(result.cache_read, 25.0 * 0.5);
    assert_eq!(result.cache_write_5m, 10.0 * 3.0);
    assert_eq!(result.output(), 20.0 * 4.0);
    assert_eq!(result.total(), result.input() + result.output());
    assert_eq!(result.rates.cache_read, 0.5);
}

#[test]
fn absent_null_and_zero_cache_rates_are_distinct() {
    let base = rates(Rate::Value(2.0), Rate::Value(4.0));
    for read in [Rate::Missing, Rate::Null] {
        let standard = Rates {
            cache_read: read,
            ..base
        };
        assert_eq!(
            calculate(&pricing(standard), &request()).unwrap().input(),
            200.0
        );
    }
    let standard = Rates {
        cache_read: Rate::Value(0.0),
        cache_write: Rate::Value(0.0),
        ..base
    };
    assert_eq!(
        calculate(&pricing(standard), &request()).unwrap().input(),
        130.0
    );
}

#[test]
fn equivalent_prompt_conventions_select_the_same_threshold() {
    let threshold = ThresholdRates {
        above_prompt_tokens: 90,
        standard: rates(Rate::Value(5.0), Rate::Value(8.0)),
        tiers: &[],
    };
    let specification = Pricing {
        standard: rates(Rate::Value(2.0), Rate::Value(4.0)),
        tiers: &[],
        thresholds: &[threshold],
        off_peak: None,
    };
    let included = request();
    let excluded = Request {
        usage: Usage {
            prompt_tokens: 65,
            prompt_convention: PromptConvention::ExcludesCache,
            ..included.usage
        },
        ..included
    };
    let plan = compile(&specification).unwrap();
    assert_eq!(plan.calculate(&included), plan.calculate(&excluded));
    assert_eq!(plan.calculate(&included).unwrap().rates.input, 5.0);
}

#[test]
fn split_writes_and_invalid_accounting() {
    let standard = Rates {
        cache_read: Rate::Value(0.5),
        cache_write: Rate::Value(3.0),
        cache_write_1h: Rate::Value(5.0),
        ..rates(Rate::Value(2.0), Rate::Value(4.0))
    };
    let base = request();
    let split = Request {
        usage: Usage {
            cache_write_5m_tokens: Some(4),
            cache_write_1h_tokens: Some(6),
            ..base.usage
        },
        ..base
    };
    let result = calculate(&pricing(standard), &split).unwrap();
    assert_eq!(result.cache_write_5m, 12.0);
    assert_eq!(result.cache_write_1h, 30.0);
    let overlapping = Request {
        usage: Usage {
            prompt_tokens: 30,
            ..split.usage
        },
        ..split
    };
    assert_eq!(
        calculate(&pricing(standard), &overlapping),
        Err(PricingError::CacheExceedsPrompt)
    );
    let incomplete = Request {
        usage: Usage {
            cache_write_1h_tokens: None,
            ..split.usage
        },
        ..split
    };
    assert_eq!(
        calculate(&pricing(standard), &incomplete),
        Err(PricingError::InvalidCacheWriteDetails)
    );
}

#[test]
fn threshold_tiers_and_boundaries() {
    let priority = TierRates {
        tier: ServiceTier::Priority,
        rates: rates(Rate::Value(3.0), Rate::Missing),
    };
    let threshold = ThresholdRates {
        above_prompt_tokens: 100,
        standard: rates(Rate::Value(5.0), Rate::Value(8.0)),
        tiers: &[
            TierRates {
                tier: ServiceTier::Priority,
                rates: rates(Rate::Value(7.0), Rate::Missing),
            },
            TierRates {
                tier: ServiceTier::Flex,
                rates: rates(Rate::Value(6.0), Rate::Missing),
            },
        ],
    };
    let specification = Pricing {
        standard: rates(Rate::Value(2.0), Rate::Value(4.0)),
        tiers: &[priority],
        thresholds: &[threshold],
        off_peak: None,
    };
    let base = request();
    let no_cache = Request {
        usage: Usage {
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            ..base.usage
        },
        ..base
    };
    let fast = Request {
        service_tier: ServiceTier::Fast,
        ..no_cache
    };
    let inclusive = Request {
        threshold_policy: ThresholdPolicy::Inclusive,
        ..fast
    };
    let flex = Request {
        service_tier: ServiceTier::Flex,
        ..inclusive
    };
    assert_eq!(calculate(&specification, &no_cache).unwrap().input(), 200.0);
    assert_eq!(calculate(&specification, &fast).unwrap().input(), 300.0);
    assert_eq!(
        calculate(&specification, &inclusive).unwrap().input(),
        700.0
    );
    assert_eq!(calculate(&specification, &flex).unwrap().input(), 600.0);
}

#[test]
fn compile_rejects_ambiguous_rates() {
    let duplicate = ThresholdRates {
        above_prompt_tokens: 100,
        standard: Rates::EMPTY,
        tiers: &[],
    };
    let specification = Pricing {
        standard: rates(Rate::Value(1.0), Rate::Value(1.0)),
        tiers: &[],
        thresholds: &[duplicate, duplicate],
        off_peak: None,
    };
    assert_eq!(
        compile(&specification).err(),
        Some(PricingError::DuplicateThreshold)
    );
    let invalid = pricing(rates(Rate::Value(f64::NAN), Rate::Value(1.0)));
    assert_eq!(compile(&invalid).err(), Some(PricingError::InvalidRate));
}

#[test]
fn off_peak_is_one_non_wrapping_utc_window() {
    let specification = Pricing {
        standard: rates(Rate::Value(2.0), Rate::Value(4.0)),
        tiers: &[],
        thresholds: &[],
        off_peak: Some(OffPeakRates {
            start_utc_minute: 60,
            end_utc_minute: 120,
            rates: rates(Rate::Value(1.0), Rate::Value(2.0)),
        }),
    };
    let base = request();
    let start = Request {
        billed_at_utc_minute: Some(60),
        ..base
    };
    let end = Request {
        billed_at_utc_minute: Some(120),
        ..base
    };
    assert_eq!(
        calculate(&specification, &base),
        Err(PricingError::InvalidBillingTime)
    );
    assert_eq!(calculate(&specification, &start).unwrap().input(), 100.0);
    assert_eq!(calculate(&specification, &end).unwrap().input(), 200.0);
}

#[test]
fn missing_rates_and_free_rates_remain_distinct() {
    let base = request();
    let empty = Request {
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            ..base.usage
        },
        ..base
    };
    assert_eq!(
        calculate(&pricing(Rates::EMPTY), &empty),
        Err(PricingError::MissingInputRate)
    );
    assert_eq!(
        calculate(&pricing(rates(Rate::Value(0.0), Rate::Missing)), &empty),
        Err(PricingError::MissingOutputRate)
    );
    assert_eq!(
        calculate(&pricing(rates(Rate::Value(0.0), Rate::Value(0.0))), &empty)
            .unwrap()
            .total(),
        0.0
    );
}

#[test]
fn matches_executed_python_reference_cases() {
    for row in include_str!("python_reference.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
    {
        let fields: Vec<_> = row.split('\t').collect();
        let count = |index: usize| fields[index].parse::<u64>().unwrap();
        let number = |index: usize| fields[index].parse::<f64>().unwrap();
        let optional_rate = |index: usize| {
            if fields[index].is_empty() {
                Rate::Missing
            } else {
                Rate::Value(number(index))
            }
        };
        let threshold = ThresholdRates {
            above_prompt_tokens: if fields[9].is_empty() { 0 } else { count(9) },
            standard: rates(optional_rate(10), optional_rate(11)),
            tiers: &[],
        };
        let thresholds = if fields[9].is_empty() {
            &[][..]
        } else {
            std::slice::from_ref(&threshold)
        };
        let specification = Pricing {
            standard: Rates {
                cache_read: optional_rate(7),
                cache_write: optional_rate(8),
                ..rates(Rate::Value(number(5)), Rate::Value(number(6)))
            },
            tiers: &[],
            thresholds,
            off_peak: None,
        };
        let base = request();
        let input = Request {
            usage: Usage {
                prompt_tokens: count(1),
                completion_tokens: count(2),
                cache_read_tokens: count(3),
                cache_write_tokens: count(4),
                ..base.usage
            },
            ..base
        };
        let actual = calculate(&specification, &input).unwrap();
        assert_eq!(actual.input(), number(12), "{}", fields[0]);
        assert_eq!(actual.output(), number(13), "{}", fields[0]);
    }
}

proptest::proptest! {
    #[test]
    fn equivalent_usage_conventions_and_breakdown_agree(
        regular in 0_u64..1000,
        read in 0_u64..1000,
        write in 0_u64..1000,
        output in 0_u64..1000,
    ) {
        let threshold = ThresholdRates {
            above_prompt_tokens: 1000,
            standard: rates(Rate::Value(5.0), Rate::Value(8.0)),
            tiers: &[],
        };
        let specification = Pricing {
            standard: rates(Rate::Value(2.0), Rate::Value(4.0)),
            tiers: &[],
            thresholds: &[threshold],
            off_peak: None,
        };
        let base = request();
        let included = Request {
            usage: Usage {
                prompt_tokens: regular + read + write,
                completion_tokens: output,
                cache_read_tokens: read,
                cache_write_tokens: write,
                ..base.usage
            },
            ..base
        };
        let excluded = Request {
            usage: Usage {
                prompt_tokens: regular,
                prompt_convention: PromptConvention::ExcludesCache,
                ..included.usage
            },
            ..included
        };
        let plan = compile(&specification).unwrap();
        let left = plan.calculate(&included).unwrap();
        let right = plan.calculate(&excluded).unwrap();
        proptest::prop_assert_eq!(left, right);
        proptest::prop_assert_eq!(left.total(), left.input() + left.output());
    }
}

#[test]
fn regional_multiplier_applies_after_input_components_are_summed() {
    let standard = Rates {
        cache_read: Rate::Value(0.5),
        cache_write: Rate::Value(3.0),
        ..rates(Rate::Value(2.0), Rate::Value(4.0))
    };
    let base = request();
    let regional = Request {
        region_multiplier: Some(1.1),
        ..base
    };
    let result = calculate(&pricing(standard), &regional).unwrap();
    assert_eq!(result.input(), (65.0 * 2.0 + 25.0 * 0.5 + 10.0 * 3.0) * 1.1);
    assert_eq!(result.output(), 20.0 * 4.0 * 1.1);
    let invalid = Request {
        region_multiplier: Some(f64::NAN),
        ..base
    };
    assert_eq!(
        calculate(&pricing(standard), &invalid),
        Err(PricingError::InvalidRegionMultiplier)
    );
}

#[test]
fn compilation_sorts_thresholds_and_rejects_duplicate_tiers() {
    let high = ThresholdRates {
        above_prompt_tokens: 200,
        standard: rates(Rate::Value(7.0), Rate::Missing),
        tiers: &[],
    };
    let low = ThresholdRates {
        above_prompt_tokens: 100,
        standard: rates(Rate::Value(5.0), Rate::Missing),
        tiers: &[],
    };
    let specification = Pricing {
        standard: rates(Rate::Value(2.0), Rate::Value(4.0)),
        tiers: &[],
        thresholds: &[high, low],
        off_peak: None,
    };
    let base = request();
    let above_both = Request {
        usage: Usage {
            prompt_tokens: 201,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            ..base.usage
        },
        ..base
    };
    assert_eq!(
        compile(&specification)
            .unwrap()
            .calculate(&above_both)
            .unwrap()
            .rates
            .input,
        7.0
    );
    let duplicate = TierRates {
        tier: ServiceTier::Flex,
        rates: Rates::EMPTY,
    };
    let invalid = Pricing {
        tiers: &[duplicate, duplicate],
        thresholds: &[],
        ..specification
    };
    assert_eq!(compile(&invalid).err(), Some(PricingError::DuplicateTier));
}
