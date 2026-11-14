use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Ceiling {
    pub model_group: String,
    pub tpm_limit: i64,
    pub output_to_input_ratio: f64,
    pub cached_input_ratio: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    #[serde(default)]
    pub cached_tokens: i64,
}

impl Usage {
    pub fn billable_input_tokens(&self) -> i64 {
        (self.prompt_tokens - self.cached_tokens).max(0)
    }
}

impl Ceiling {
    pub fn raw_output_token_limit(&self) -> i64 {
        ((self.tpm_limit as f64 / self.output_to_input_ratio.max(1.0)).trunc() as i64).max(1)
    }

    pub fn normalized(
        &self,
        prompt_tokens: i64,
        completion_tokens: i64,
        cache_read_tokens: i64,
    ) -> f64 {
        let cached = cache_read_tokens.max(0).min(prompt_tokens.max(0));
        let uncached = prompt_tokens.max(0) - cached;
        uncached as f64
            + self.cached_input_ratio * cached as f64
            + self.output_to_input_ratio * completion_tokens.max(0) as f64
    }

    pub fn estimate(&self, input_tokens: i64, output_tokens: i64) -> i64 {
        (self
            .normalized(input_tokens, output_tokens, 0)
            .round_ties_even() as i64)
            .max(1)
    }

    pub fn needed(&self, input_tokens: f64, output_tokens: f64) -> f64 {
        (input_tokens.max(0.0)
            + self.cached_input_ratio * 0.0
            + self.output_to_input_ratio * output_tokens.max(0.0))
        .round_ties_even()
        .max(1.0)
    }

    pub fn exceeded(&self, needed_tokens: &str) -> String {
        format!(
            "This request needs {needed_tokens} normalized tokens, more than your team's PTU share of {} allows in a minute ({}), so it can never be admitted. Lower max_tokens or shorten the prompt.",
            self.model_group, self.tpm_limit
        )
    }
}

const I64_END: f64 = 9_223_372_036_854_775_808.0;

pub fn beyond_i64(tokens: f64) -> bool {
    tokens >= I64_END
}

pub fn saturated(tokens: f64) -> i64 {
    if beyond_i64(tokens) {
        i64::MAX
    } else {
        tokens as i64
    }
}

pub fn settlement_tokens(ceiling: Option<&Ceiling>, usage: Option<&Usage>, raw_total: i64) -> i64 {
    match (ceiling, usage) {
        (Some(ceiling), Some(usage)) => ceiling
            .normalized(
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.cached_tokens,
            )
            .round_ties_even() as i64,
        _ => raw_total,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn ceiling(output_ratio: f64, cached_ratio: f64) -> Ceiling {
        Ceiling {
            model_group: "gpt".into(),
            tpm_limit: 1000,
            output_to_input_ratio: output_ratio,
            cached_input_ratio: cached_ratio,
        }
    }

    #[rstest]
    #[case::output_weighted(4.0, 0.0, Usage { prompt_tokens: 10, completion_tokens: 5, cached_tokens: 0 }, 30)]
    #[case::cached_input_discounted(1.0, 0.5, Usage { prompt_tokens: 10, completion_tokens: 0, cached_tokens: 3 }, 8)]
    #[case::half_rounds_to_even(1.0, 0.5, Usage { prompt_tokens: 1, completion_tokens: 0, cached_tokens: 1 }, 0)]
    #[case::cache_capped_by_prompt(1.0, 0.0, Usage { prompt_tokens: 2, completion_tokens: 1, cached_tokens: 9 }, 1)]
    fn settlement_weighs_usage_like_azure(
        #[case] output_ratio: f64,
        #[case] cached_ratio: f64,
        #[case] usage: Usage,
        #[case] expected: i64,
    ) {
        assert_eq!(
            settlement_tokens(Some(&ceiling(output_ratio, cached_ratio)), Some(&usage), 77),
            expected
        );
    }

    #[rstest]
    #[case::no_usage(Some(ceiling(4.0, 0.0)), None)]
    #[case::no_ceiling(None, Some(Usage { prompt_tokens: 1, completion_tokens: 1, cached_tokens: 0 }))]
    fn settlement_falls_back_to_the_raw_total(
        #[case] ceiling: Option<Ceiling>,
        #[case] usage: Option<Usage>,
    ) {
        assert_eq!(settlement_tokens(ceiling.as_ref(), usage.as_ref(), 77), 77);
    }

    #[rstest]
    #[case::weighted(4.0, 1000, 250)]
    #[case::ratio_below_one(0.5, 1000, 1000)]
    #[case::floored_at_one(4.0, 2, 1)]
    fn the_raw_output_limit_divides_by_the_output_ratio(
        #[case] ratio: f64,
        #[case] limit: i64,
        #[case] expected: i64,
    ) {
        let ceiling = Ceiling {
            tpm_limit: limit,
            ..ceiling(ratio, 0.0)
        };
        assert_eq!(ceiling.raw_output_token_limit(), expected);
    }

    #[rstest]
    #[case::zero_estimate_floors_at_one(0, 0, 1)]
    #[case::output_weighted(10, 5, 30)]
    fn the_estimate_is_at_least_one(
        #[case] input: i64,
        #[case] output: i64,
        #[case] expected: i64,
    ) {
        assert_eq!(ceiling(4.0, 0.0).estimate(input, output), expected);
    }
}
