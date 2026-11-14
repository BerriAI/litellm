#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Amount {
    Int(i128),
    Float(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Limit {
    Amount(Amount),
    Other(String),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GroupLimits {
    pub tpm: Option<Limit>,
    pub rpm: Option<Limit>,
    pub itpm: Option<Limit>,
    pub otpm: Option<Limit>,
}

pub type Totals = (Option<i128>, Option<i128>);

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RemainingReads {
    pub limits: Option<GroupLimits>,
    pub io: Option<Totals>,
    pub usage: Option<Totals>,
    pub reset: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("unsupported operand type(s) for -: '{0}' and 'int'")]
pub struct Unsubtractable(pub String);

pub type Headers = Vec<(&'static str, Amount)>;

pub fn usage_total(values: impl IntoIterator<Item = Option<i128>>) -> Option<i128> {
    values
        .into_iter()
        .flatten()
        .fold(None, |total, value| Some(total.unwrap_or(0) + value))
}

pub fn usage_totals(values: &[Option<i128>], tpm_count: usize) -> Totals {
    let split = tpm_count.min(values.len());
    let (tpm, rpm) = values.split_at(split);
    (
        usage_total(tpm.iter().copied()),
        usage_total(rpm.iter().copied()),
    )
}

#[allow(clippy::cast_precision_loss)]
fn minus(limit: &Limit, used: Option<i128>) -> Result<Amount, Unsubtractable> {
    let used = used.unwrap_or(0);
    match limit {
        Limit::Amount(Amount::Int(limit)) => Ok(Amount::Int(limit - used)),
        Limit::Amount(Amount::Float(limit)) => Ok(Amount::Float(limit - used as f64)),
        Limit::Other(type_name) => Err(Unsubtractable(type_name.clone())),
    }
}

fn at_least_zero(amount: Amount) -> Amount {
    match amount {
        Amount::Int(value) if value > 0 => amount,
        Amount::Float(value) if value > 0.0 => amount,
        _ => Amount::Int(0),
    }
}

fn amount(limit: &Limit) -> Result<Amount, Unsubtractable> {
    match limit {
        Limit::Amount(amount) => Ok(*amount),
        Limit::Other(type_name) => Err(Unsubtractable(type_name.clone())),
    }
}

fn io_headers(
    limits: &GroupLimits,
    (itpm, otpm): Totals,
    reset: i64,
    headers: &mut Headers,
) -> Result<(), Unsubtractable> {
    let reset = Amount::Int(i128::from(reset));
    for (limit, used, names) in [
        (
            &limits.itpm,
            itpm,
            [
                "x-ratelimit-limit-input-tokens",
                "x-ratelimit-remaining-input-tokens",
                "x-ratelimit-reset-input-tokens",
            ],
        ),
        (
            &limits.otpm,
            otpm,
            [
                "x-ratelimit-limit-output-tokens",
                "x-ratelimit-remaining-output-tokens",
                "x-ratelimit-reset-output-tokens",
            ],
        ),
    ] {
        let Some(limit) = limit else {
            continue;
        };
        let remaining = at_least_zero(minus(limit, used)?);
        headers.push((names[0], amount(limit)?));
        headers.push((names[1], remaining));
        headers.push((names[2], reset));
    }
    Ok(())
}

pub fn remaining_usage(reads: &RemainingReads) -> Result<Headers, Unsubtractable> {
    let mut headers = Headers::new();
    let Some(limits) = &reads.limits else {
        return Ok(headers);
    };
    if let Some(io) = reads.io {
        io_headers(limits, io, reads.reset, &mut headers)?;
    }
    let (tpm, rpm) = reads.usage.unwrap_or((None, None));
    for (limit, used, remaining_name, limit_name) in [
        (
            &limits.tpm,
            tpm,
            "x-ratelimit-remaining-tokens",
            "x-ratelimit-limit-tokens",
        ),
        (
            &limits.rpm,
            rpm,
            "x-ratelimit-remaining-requests",
            "x-ratelimit-limit-requests",
        ),
    ] {
        let Some(limit) = limit else {
            continue;
        };
        headers.push((remaining_name, minus(limit, used)?));
        headers.push((limit_name, amount(limit)?));
    }
    Ok(headers)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const fn int(value: i128) -> Option<Limit> {
        Some(Limit::Amount(Amount::Int(value)))
    }

    fn limits(tpm: Option<Limit>, rpm: Option<Limit>) -> GroupLimits {
        GroupLimits {
            tpm,
            rpm,
            ..GroupLimits::default()
        }
    }

    fn reads(
        limits: Option<GroupLimits>,
        io: Option<Totals>,
        usage: Option<Totals>,
    ) -> RemainingReads {
        RemainingReads {
            limits,
            io,
            usage,
            reset: 42,
        }
    }

    fn headers(reads: &RemainingReads) -> Vec<(&'static str, Amount)> {
        remaining_usage(reads).unwrap_or_else(|error| panic!("{error}"))
    }

    #[rstest]
    #[case::ints(vec![Some(30), Some(4)], Some(34))]
    #[case::skips_what_is_not_counted(vec![None, Some(2), None], Some(2))]
    #[case::nothing_counted(vec![None, None], None)]
    #[case::empty(vec![], None)]
    #[case::zero_is_a_count(vec![Some(0)], Some(0))]
    fn a_total_sums_the_counters_held_and_is_none_when_none_are(
        #[case] values: Vec<Option<i128>>,
        #[case] expected: Option<i128>,
    ) {
        assert_eq!(usage_total(values), expected);
    }

    #[rstest]
    #[case::split(vec![Some(30), Some(4), Some(1), Some(2)], 2, (Some(34), Some(3)))]
    #[case::no_rpm_counted(vec![Some(30), None], 1, (Some(30), None))]
    #[case::past_the_end(vec![Some(5)], 3, (Some(5), None))]
    #[case::no_counters(vec![], 0, (None, None))]
    fn totals_split_the_tpm_counters_from_the_rpm_ones(
        #[case] values: Vec<Option<i128>>,
        #[case] tpm_count: usize,
        #[case] expected: Totals,
    ) {
        assert_eq!(usage_totals(&values, tpm_count), expected);
    }

    #[rstest]
    fn remaining_usage_is_each_limit_minus_this_minutes_usage() {
        let reads = reads(
            Some(limits(int(1100), int(15))),
            None,
            Some((Some(30), Some(3))),
        );
        assert_eq!(
            headers(&reads),
            vec![
                ("x-ratelimit-remaining-tokens", Amount::Int(1070)),
                ("x-ratelimit-limit-tokens", Amount::Int(1100)),
                ("x-ratelimit-remaining-requests", Amount::Int(12)),
                ("x-ratelimit-limit-requests", Amount::Int(15)),
            ]
        );
    }

    #[rstest]
    fn nothing_counted_leaves_the_whole_limit_remaining() {
        let reads = reads(Some(limits(int(100), int(5))), None, Some((None, None)));
        assert_eq!(
            headers(&reads),
            vec![
                ("x-ratelimit-remaining-tokens", Amount::Int(100)),
                ("x-ratelimit-limit-tokens", Amount::Int(100)),
                ("x-ratelimit-remaining-requests", Amount::Int(5)),
                ("x-ratelimit-limit-requests", Amount::Int(5)),
            ]
        );
    }

    #[rstest]
    fn an_io_limited_group_reports_input_and_output_token_headers() {
        let io = GroupLimits {
            itpm: int(300),
            otpm: int(40),
            ..GroupLimits::default()
        };
        let reads = reads(Some(io), Some((Some(7), None)), None);
        assert_eq!(
            headers(&reads),
            vec![
                ("x-ratelimit-limit-input-tokens", Amount::Int(300)),
                ("x-ratelimit-remaining-input-tokens", Amount::Int(293)),
                ("x-ratelimit-reset-input-tokens", Amount::Int(42)),
                ("x-ratelimit-limit-output-tokens", Amount::Int(40)),
                ("x-ratelimit-remaining-output-tokens", Amount::Int(40)),
                ("x-ratelimit-reset-output-tokens", Amount::Int(42)),
            ]
        );
    }

    #[rstest]
    fn io_and_tpm_limits_both_report_their_headers() {
        let both = GroupLimits {
            tpm: int(1000),
            rpm: int(100),
            itpm: int(900),
            otpm: int(450),
        };
        let reads = reads(Some(both), Some((None, None)), Some((None, None)));
        let names: Vec<_> = headers(&reads).into_iter().map(|(name, _)| name).collect();
        assert_eq!(
            names,
            vec![
                "x-ratelimit-limit-input-tokens",
                "x-ratelimit-remaining-input-tokens",
                "x-ratelimit-reset-input-tokens",
                "x-ratelimit-limit-output-tokens",
                "x-ratelimit-remaining-output-tokens",
                "x-ratelimit-reset-output-tokens",
                "x-ratelimit-remaining-tokens",
                "x-ratelimit-limit-tokens",
                "x-ratelimit-remaining-requests",
                "x-ratelimit-limit-requests",
            ]
        );
    }

    #[rstest]
    fn over_the_limit_clamps_only_the_io_headers_at_zero() {
        let all = GroupLimits {
            tpm: int(10),
            rpm: None,
            itpm: int(10),
            otpm: None,
        };
        let reads = reads(Some(all), Some((Some(25), None)), Some((Some(25), None)));
        assert_eq!(
            headers(&reads),
            vec![
                ("x-ratelimit-limit-input-tokens", Amount::Int(10)),
                ("x-ratelimit-remaining-input-tokens", Amount::Int(0)),
                ("x-ratelimit-reset-input-tokens", Amount::Int(42)),
                ("x-ratelimit-remaining-tokens", Amount::Int(-15)),
                ("x-ratelimit-limit-tokens", Amount::Int(10)),
            ]
        );
    }

    #[rstest]
    #[case::positive(2.5, Amount::Float(1.5))]
    #[case::exhausted(1.0, Amount::Int(0))]
    #[case::nan(f64::NAN, Amount::Int(0))]
    fn a_float_io_limit_keeps_python_max_with_an_int_zero(
        #[case] limit: f64,
        #[case] expected: Amount,
    ) {
        let io = GroupLimits {
            itpm: Some(Limit::Amount(Amount::Float(limit))),
            ..GroupLimits::default()
        };
        let reads = reads(Some(io), Some((Some(1), None)), None);
        assert_eq!(
            headers(&reads)[1],
            ("x-ratelimit-remaining-input-tokens", expected)
        );
    }

    #[rstest]
    fn a_float_tpm_limit_leaves_a_float_remaining() {
        let reads = reads(
            Some(limits(Some(Limit::Amount(Amount::Float(10.5))), None)),
            None,
            Some((Some(3), None)),
        );
        assert_eq!(
            headers(&reads),
            vec![
                ("x-ratelimit-remaining-tokens", Amount::Float(7.5)),
                ("x-ratelimit-limit-tokens", Amount::Float(10.5)),
            ]
        );
    }

    #[rstest]
    #[case::no_group(None, None, None)]
    #[case::no_limits(Some(GroupLimits::default()), None, None)]
    #[case::io_not_read(Some(GroupLimits { itpm: int(3), ..GroupLimits::default() }), None, None)]
    fn a_group_without_limits_or_reads_is_empty(
        #[case] limits: Option<GroupLimits>,
        #[case] io: Option<Totals>,
        #[case] usage: Option<Totals>,
    ) {
        assert_eq!(headers(&reads(limits, io, usage)), vec![]);
    }

    #[rstest]
    fn a_limit_python_cannot_subtract_raises_its_type_error() {
        let reads = reads(
            Some(limits(Some(Limit::Other("str".to_owned())), int(5))),
            None,
            Some((None, None)),
        );
        let error = remaining_usage(&reads).unwrap_err();
        assert_eq!(
            error.to_string(),
            "unsupported operand type(s) for -: 'str' and 'int'"
        );
    }
}
