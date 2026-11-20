//! Provider-keyed discount and margin, a port of `litellm_core_utils/cost_discount.py` and of `_apply_cost_discount`
//! and `_apply_cost_margin` in `cost_calculator.py`. Keys and rules keep the order the config wrote them in, since a
//! tie between two equally specific patterns goes to the one written first

use litellm_fnmatch::fnmatchcase;

const GLOB: [char; 3] = ['*', '?', '['];

/// `cost_discount_config` and `cost_margin_config`, which adjust a priced call
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CostConfig {
    pub discounts: Discounts,
    pub margins: Margins,
}

/// `cost_discount_config`: a fraction per `<provider>` or `<provider>/<model pattern>` key
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Discounts(Vec<(String, f64)>);

/// `cost_margin_config`: a rule per provider, and `global` for every provider without one
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Margins(Vec<(String, MarginRule)>);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MarginRule {
    /// A bare number, a fraction of the cost
    Percent(f64),
    /// `{"percentage": ..., "fixed_amount": ...}`, either key optional
    Split {
        percentage: Option<f64>,
        fixed_amount: Option<f64>,
    },
    /// A value of another type. Main still counts the provider as configured and adds nothing
    Ignored,
}

/// Who the call's adjustments are keyed on
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdjustmentKey<'a> {
    pub provider: Option<&'a str>,
    pub model: Option<&'a str>,
    pub region: Option<&'a str>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Discount {
    pub percent: f64,
    pub amount: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Margin {
    pub percent: f64,
    pub fixed: f64,
    pub total: f64,
}

/// A cost after discount and then margin, with both parts
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Adjusted {
    pub cost: f64,
    pub discount: Discount,
    pub margin: Margin,
}

impl Discounts {
    pub fn new(rules: impl IntoIterator<Item = (String, f64)>) -> Self {
        Self(rules.into_iter().collect())
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn get(&self, key: &str) -> Option<f64> {
        self.0
            .iter()
            .find(|(written, _)| written == key)
            .map(|(_, fraction)| *fraction)
    }

    /// `resolve_cost_discount`: an exact `<provider>/<model>` key, else the matching glob with the most literal
    /// characters, else the bare `<provider>` key. The model is tried with each run of repeated `<provider>/`
    /// prefixes stripped, and with its `<provider>/<region>/` segment stripped
    pub fn resolve(&self, key: &AdjustmentKey<'_>) -> Option<f64> {
        let provider = key.provider.filter(|provider| !provider.is_empty())?;
        let prefix = format!("{provider}/");
        let names: Vec<String> = key.model.map_or_else(Vec::new, |model| {
            let region_stripped = key
                .region
                .filter(|region| !region.is_empty())
                .and_then(|region| model.strip_prefix(&format!("{prefix}{region}/")))
                .map(|bare| names_without_prefix(&format!("{prefix}{bare}"), &prefix))
                .unwrap_or_default();
            names_without_prefix(model, &prefix)
                .into_iter()
                .chain(region_stripped)
                .collect()
        });
        let patterns: Vec<&str> = self
            .0
            .iter()
            .filter_map(|(written, _)| written.strip_prefix(&prefix))
            .collect();
        let exact = names.iter().find(|name| {
            patterns
                .iter()
                .any(|pattern| !is_glob(pattern) && pattern == name)
        });
        if let Some(name) = exact {
            return self.get(&format!("{prefix}{name}"));
        }
        let markers: Vec<String> = match key.region.filter(|region| !region.is_empty()) {
            Some(region) => vec![prefix.clone(), format!("{region}/")],
            None => vec![prefix.clone()],
        };
        let best = patterns
            .iter()
            .filter(|pattern| is_glob(pattern))
            .filter(|pattern| {
                names
                    .iter()
                    .filter(|name| {
                        pattern.contains('/')
                            || !markers.iter().any(|marker| name.starts_with(marker))
                    })
                    .any(|name| fnmatchcase(name, pattern))
            })
            .fold(None::<&str>, |best, pattern| match best {
                Some(kept) if literal_length(kept) >= literal_length(pattern) => Some(kept),
                _ => Some(pattern),
            });
        match best {
            Some(pattern) => self.get(&format!("{prefix}{pattern}")),
            None => self.get(provider),
        }
    }

    /// `_apply_cost_discount`, run only when the config is not empty
    pub fn apply(&self, cost: f64, key: &AdjustmentKey<'_>) -> (f64, Discount) {
        if self.is_empty() {
            return (cost, Discount::default());
        }
        let resolved = self.resolve(key);
        let percent = resolved.unwrap_or(0.0);
        let amount = cost * percent;
        let discounted = if resolved.is_some() {
            cost - amount
        } else {
            cost
        };
        (discounted, Discount { percent, amount })
    }
}

impl Margins {
    pub fn new(rules: impl IntoIterator<Item = (String, MarginRule)>) -> Self {
        Self(rules.into_iter().collect())
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn get(&self, key: &str) -> Option<MarginRule> {
        self.0
            .iter()
            .find(|(written, _)| written == key)
            .map(|(_, rule)| *rule)
    }

    /// `_apply_cost_margin`, run only when the config is not empty: the provider's rule, else `global`
    pub fn apply(&self, cost: f64, provider: Option<&str>) -> (f64, Margin) {
        if self.is_empty() {
            return (cost, Margin::default());
        }
        let rule = provider
            .filter(|provider| !provider.is_empty())
            .and_then(|provider| self.get(provider))
            .or_else(|| self.get("global"));
        let Some(rule) = rule else {
            return (cost, Margin::default());
        };
        let margin = match rule {
            MarginRule::Percent(percent) => Margin {
                percent,
                fixed: 0.0,
                total: cost * percent,
            },
            MarginRule::Split {
                percentage,
                fixed_amount,
            } => {
                let from_percent = percentage.map_or(0.0, |percent| 0.0 + cost * percent);
                Margin {
                    percent: percentage.unwrap_or(0.0),
                    fixed: fixed_amount.unwrap_or(0.0),
                    total: fixed_amount.map_or(from_percent, |fixed| from_percent + fixed),
                }
            }
            MarginRule::Ignored => Margin::default(),
        };
        (cost + margin.total, margin)
    }
}

/// Discount, then margin on the discounted cost
pub fn adjust(
    discounts: &Discounts,
    margins: &Margins,
    cost: f64,
    key: &AdjustmentKey<'_>,
) -> Adjusted {
    let (discounted, discount) = discounts.apply(cost, key);
    let (cost, margin) = margins.apply(discounted, key.provider);
    Adjusted {
        cost,
        discount,
        margin,
    }
}

fn is_glob(pattern: &str) -> bool {
    pattern.contains(GLOB)
}

/// `_model_names`: the model with each run of its leading `<provider>/` prefixes stripped, or the model itself
fn names_without_prefix(model: &str, prefix: &str) -> Vec<String> {
    let depth = std::iter::successors(Some(model), |rest| rest.strip_prefix(prefix))
        .skip(1)
        .count();
    if depth == 0 {
        return vec![model.to_owned()];
    }
    (1..=depth)
        .map(|level| model[prefix.len() * level..].to_owned())
        .collect()
}

/// `_literal_length`: characters a glob matches literally. A bracket class counts as none
fn literal_length(pattern: &str) -> usize {
    let chars: Vec<char> = pattern.chars().collect();
    std::iter::successors(Some(0usize), |&at| {
        (at < chars.len()).then(|| at + token_len(&chars[at..]))
    })
    .take_while(|&at| at < chars.len())
    .filter(|&at| token_len(&chars[at..]) == 1 && !matches!(chars[at], '*' | '?'))
    .count()
}

/// Length of the token starting `rest`, a closed bracket class or one character, as `_GLOB_TOKEN` splits it
fn token_len(rest: &[char]) -> usize {
    let class_end = |from: usize| {
        rest[from..]
            .iter()
            .position(|c| *c == ']')
            .map(|end| from + end + 1)
    };
    match rest {
        ['[', '!', ']', ..] => class_end(3).unwrap_or(1),
        ['[', ']', ..] => class_end(2).unwrap_or(1),
        ['[', '!', ..] => class_end(2).unwrap_or(1),
        ['[', ..] => class_end(1).unwrap_or(1),
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn discounts(rules: &[(&str, f64)]) -> Discounts {
        Discounts::new(
            rules
                .iter()
                .map(|(key, fraction)| ((*key).to_owned(), *fraction)),
        )
    }

    fn key<'a>(provider: &'a str, model: &'a str, region: Option<&'a str>) -> AdjustmentKey<'a> {
        AdjustmentKey {
            provider: Some(provider),
            model: Some(model),
            region,
        }
    }

    #[rstest]
    #[case::provider_only(&[("anthropic", 0.1)], key("anthropic", "claude-opus-4-8", None), Some(0.1))]
    #[case::another_provider(&[("openai", 0.1)], key("anthropic", "claude-opus-4-8", None), None)]
    #[case::exact_beats_glob(
        &[("anthropic/claude-*", 0.2), ("anthropic/claude-opus-4-8", 0.3)],
        key("anthropic", "claude-opus-4-8", None),
        Some(0.3)
    )]
    #[case::longer_literal_glob_wins(
        &[("anthropic", 0.1), ("anthropic/claude-*", 0.2), ("anthropic/claude-opus-*", 0.25)],
        key("anthropic", "claude-opus-4-8", None),
        Some(0.25)
    )]
    #[case::equal_globs_keep_the_first_written(
        &[("anthropic/claude-?pus*", 0.2), ("anthropic/claude-o?us*", 0.25)],
        key("anthropic", "claude-opus-4-8", None),
        Some(0.2)
    )]
    #[case::bracket_class_counts_no_literal(
        &[("anthropic/claude-[op]*", 0.2), ("anthropic/claude-o*", 0.25)],
        key("anthropic", "claude-opus-4-8", None),
        Some(0.25)
    )]
    #[case::unmatched_glob_falls_back_to_provider(
        &[("anthropic", 0.1), ("anthropic/claude-haiku-*", 0.5)],
        key("anthropic", "claude-opus-4-8", None),
        Some(0.1)
    )]
    #[case::repeated_provider_prefix_is_stripped(
        &[("bedrock/claude-*", 0.4)],
        key("bedrock", "bedrock/bedrock/claude-opus-5", None),
        Some(0.4)
    )]
    #[case::a_slashless_glob_skips_prefixed_names(
        &[("bedrock/*", 0.4)],
        key("bedrock", "bedrock/us-east-1/claude-opus-5", Some("us-east-1")),
        Some(0.4)
    )]
    #[case::region_segment_is_stripped_for_exact_keys(
        &[("bedrock/claude-opus-5", 0.4)],
        key("bedrock", "bedrock/us-east-1/claude-opus-5", Some("us-east-1")),
        Some(0.4)
    )]
    #[case::no_provider(&[("anthropic", 0.1)], AdjustmentKey::default(), None)]
    fn a_discount_resolves_exact_then_most_literal_glob_then_provider(
        #[case] rules: &[(&str, f64)],
        #[case] key: AdjustmentKey<'_>,
        #[case] expected: Option<f64>,
    ) {
        assert_eq!(discounts(rules).resolve(&key), expected);
    }

    #[rstest]
    #[case::glob(&"claude-*", 7)]
    #[case::question_mark(&"claude-?", 7)]
    #[case::class(&"[ab]c", 1)]
    #[case::negated_class(&"[!a]c", 1)]
    #[case::class_starting_with_bracket(&"[]a]c", 1)]
    #[case::unclosed_bracket(&"[abc", 4)]
    fn literal_length_counts_plain_characters(#[case] pattern: &&str, #[case] expected: usize) {
        assert_eq!(literal_length(pattern), expected);
    }

    #[rstest]
    #[case::percent(MarginRule::Percent(0.1), (11.0, Margin { percent: 0.1, fixed: 0.0, total: 1.0 }))]
    #[case::fixed_only(
        MarginRule::Split { percentage: None, fixed_amount: Some(0.5) },
        (10.5, Margin { percent: 0.0, fixed: 0.5, total: 0.5 })
    )]
    #[case::percent_and_fixed(
        MarginRule::Split { percentage: Some(0.1), fixed_amount: Some(0.5) },
        (11.5, Margin { percent: 0.1, fixed: 0.5, total: 1.5 })
    )]
    #[case::ignored(MarginRule::Ignored, (10.0, Margin::default()))]
    fn a_margin_adds_its_percent_and_fixed_amount(
        #[case] rule: MarginRule,
        #[case] expected: (f64, Margin),
    ) {
        let margins = Margins::new([("anthropic".to_owned(), rule)]);
        assert_eq!(margins.apply(10.0, Some("anthropic")), expected);
    }

    #[rstest]
    #[case::own_rule(Some("anthropic"), 12.0)]
    #[case::global(Some("openai"), 15.0)]
    #[case::no_provider_reads_global(None, 15.0)]
    fn a_provider_without_a_margin_rule_reads_global(
        #[case] provider: Option<&str>,
        #[case] expected: f64,
    ) {
        let margins = Margins::new([
            ("anthropic".to_owned(), MarginRule::Percent(0.2)),
            ("global".to_owned(), MarginRule::Percent(0.5)),
        ]);
        assert_eq!(margins.apply(10.0, provider).0, expected);
    }

    #[rstest]
    fn the_margin_applies_to_the_discounted_cost() {
        let adjusted = adjust(
            &discounts(&[("anthropic", 0.5)]),
            &Margins::new([("global".to_owned(), MarginRule::Percent(0.1))]),
            10.0,
            &key("anthropic", "claude-opus-4-8", None),
        );
        assert_eq!(adjusted.cost, 5.5);
        assert_eq!(
            adjusted.discount,
            Discount {
                percent: 0.5,
                amount: 5.0
            }
        );
        assert_eq!(adjusted.margin.total, 0.5);
    }

    #[rstest]
    fn an_unmatched_discount_leaves_the_cost() {
        assert_eq!(
            discounts(&[("openai", 0.5)]).apply(10.0, &key("anthropic", "claude", None)),
            (10.0, Discount::default())
        );
    }
}
