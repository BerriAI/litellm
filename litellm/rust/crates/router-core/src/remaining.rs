use num_bigint::BigInt;

#[derive(Clone, Debug, PartialEq)]
pub enum Amount {
    Int(BigInt),
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

pub type Totals = (Option<BigInt>, Option<BigInt>);

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RemainingReads {
    pub limits: Option<GroupLimits>,
    pub io: Option<Totals>,
    pub usage: Option<Totals>,
    pub reset: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RemainingError {
    #[error("unsupported operand type(s) for -: '{0}' and 'int'")]
    Unsubtractable(String),
    #[error("int too large to convert to float")]
    IntTooLarge,
}

pub type Headers = Vec<(&'static str, Amount)>;

pub fn usage_total(values: impl IntoIterator<Item = Option<BigInt>>) -> Option<BigInt> {
    values
        .into_iter()
        .flatten()
        .fold(None, |total, value| Some(total.unwrap_or_default() + value))
}

pub fn usage_totals(values: &[Option<BigInt>], tpm_count: usize) -> Totals {
    let split = tpm_count.min(values.len());
    let (tpm, rpm) = values.split_at(split);
    (
        usage_total(tpm.iter().cloned()),
        usage_total(rpm.iter().cloned()),
    )
}

pub fn int_as_float(value: &BigInt) -> Result<f64, RemainingError> {
    value
        .to_string()
        .parse::<f64>()
        .ok()
        .filter(|float| float.is_finite())
        .ok_or(RemainingError::IntTooLarge)
}

fn minus(limit: &Limit, used: Option<&BigInt>) -> Result<Amount, RemainingError> {
    let zero = BigInt::default();
    let used = used.unwrap_or(&zero);
    match limit {
        Limit::Amount(Amount::Int(limit)) => Ok(Amount::Int(limit - used)),
        Limit::Amount(Amount::Float(limit)) => Ok(Amount::Float(limit - int_as_float(used)?)),
        Limit::Other(type_name) => Err(RemainingError::Unsubtractable(type_name.clone())),
    }
}

fn at_least_zero(amount: Amount) -> Amount {
    match amount {
        Amount::Int(ref value) if *value > BigInt::default() => amount,
        Amount::Float(value) if value > 0.0 => amount,
        _ => Amount::Int(BigInt::default()),
    }
}

fn amount(limit: &Limit) -> Result<Amount, RemainingError> {
    match limit {
        Limit::Amount(amount) => Ok(amount.clone()),
        Limit::Other(type_name) => Err(RemainingError::Unsubtractable(type_name.clone())),
    }
}

fn io_headers(
    limits: &GroupLimits,
    (itpm, otpm): &Totals,
    reset: i64,
    headers: &mut Headers,
) -> Result<(), RemainingError> {
    let reset = Amount::Int(BigInt::from(reset));
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
        let remaining = at_least_zero(minus(limit, used.as_ref())?);
        headers.push((names[0], amount(limit)?));
        headers.push((names[1], remaining));
        headers.push((names[2], reset.clone()));
    }
    Ok(())
}

pub fn remaining_usage(reads: &RemainingReads) -> Result<Headers, RemainingError> {
    let mut headers = Headers::new();
    let Some(limits) = &reads.limits else {
        return Ok(headers);
    };
    if let Some(io) = &reads.io {
        io_headers(limits, io, reads.reset, &mut headers)?;
    }
    let (tpm, rpm) = reads.usage.clone().unwrap_or((None, None));
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
        headers.push((remaining_name, minus(limit, used.as_ref())?));
        headers.push((limit_name, amount(limit)?));
    }
    Ok(headers)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn big(value: i64) -> BigInt {
        BigInt::from(value)
    }

    fn whole(value: i64) -> Amount {
        Amount::Int(big(value))
    }

    fn int(value: i64) -> Option<Limit> {
        Some(Limit::Amount(whole(value)))
    }

    fn counted(values: &[Option<i64>]) -> Vec<Option<BigInt>> {
        values.iter().map(|value| value.map(big)).collect()
    }

    fn totals(tpm: Option<i64>, rpm: Option<i64>) -> Totals {
        (tpm.map(big), rpm.map(big))
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
    #[case::ints(&[Some(30), Some(4)], Some(34))]
    #[case::skips_what_is_not_counted(&[None, Some(2), None], Some(2))]
    #[case::nothing_counted(&[None, None], None)]
    #[case::empty(&[], None)]
    #[case::zero_is_a_count(&[Some(0)], Some(0))]
    fn a_total_sums_the_counters_held_and_is_none_when_none_are(
        #[case] values: &[Option<i64>],
        #[case] expected: Option<i64>,
    ) {
        assert_eq!(usage_total(counted(values)), expected.map(big));
    }

    #[rstest]
    #[case::split(&[Some(30), Some(4), Some(1), Some(2)], 2, (Some(34), Some(3)))]
    #[case::no_rpm_counted(&[Some(30), None], 1, (Some(30), None))]
    #[case::past_the_end(&[Some(5)], 3, (Some(5), None))]
    #[case::no_counters(&[], 0, (None, None))]
    fn totals_split_the_tpm_counters_from_the_rpm_ones(
        #[case] values: &[Option<i64>],
        #[case] tpm_count: usize,
        #[case] expected: (Option<i64>, Option<i64>),
    ) {
        assert_eq!(
            usage_totals(&counted(values), tpm_count),
            totals(expected.0, expected.1)
        );
    }

    #[rstest]
    fn a_total_past_any_machine_int_stays_exact() {
        let huge: BigInt = "170141183460469231731687303715884105727".parse().unwrap();
        assert_eq!(
            usage_total([Some(huge.clone()), Some(big(1))]),
            Some(huge + 1)
        );
    }

    #[rstest]
    fn remaining_usage_is_each_limit_minus_this_minutes_usage() {
        let reads = reads(
            Some(limits(int(1100), int(15))),
            None,
            Some(totals(Some(30), Some(3))),
        );
        assert_eq!(
            headers(&reads),
            vec![
                ("x-ratelimit-remaining-tokens", whole(1070)),
                ("x-ratelimit-limit-tokens", whole(1100)),
                ("x-ratelimit-remaining-requests", whole(12)),
                ("x-ratelimit-limit-requests", whole(15)),
            ]
        );
    }

    #[rstest]
    fn nothing_counted_leaves_the_whole_limit_remaining() {
        let reads = reads(Some(limits(int(100), int(5))), None, Some((None, None)));
        assert_eq!(
            headers(&reads),
            vec![
                ("x-ratelimit-remaining-tokens", whole(100)),
                ("x-ratelimit-limit-tokens", whole(100)),
                ("x-ratelimit-remaining-requests", whole(5)),
                ("x-ratelimit-limit-requests", whole(5)),
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
        let reads = reads(Some(io), Some(totals(Some(7), None)), None);
        assert_eq!(
            headers(&reads),
            vec![
                ("x-ratelimit-limit-input-tokens", whole(300)),
                ("x-ratelimit-remaining-input-tokens", whole(293)),
                ("x-ratelimit-reset-input-tokens", whole(42)),
                ("x-ratelimit-limit-output-tokens", whole(40)),
                ("x-ratelimit-remaining-output-tokens", whole(40)),
                ("x-ratelimit-reset-output-tokens", whole(42)),
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
        let reads = reads(
            Some(all),
            Some(totals(Some(25), None)),
            Some(totals(Some(25), None)),
        );
        assert_eq!(
            headers(&reads),
            vec![
                ("x-ratelimit-limit-input-tokens", whole(10)),
                ("x-ratelimit-remaining-input-tokens", whole(0)),
                ("x-ratelimit-reset-input-tokens", whole(42)),
                ("x-ratelimit-remaining-tokens", whole(-15)),
                ("x-ratelimit-limit-tokens", whole(10)),
            ]
        );
    }

    #[rstest]
    #[case::positive(2.5, Amount::Float(1.5))]
    #[case::exhausted(1.0, whole(0))]
    #[case::nan(f64::NAN, whole(0))]
    fn a_float_io_limit_keeps_python_max_with_an_int_zero(
        #[case] limit: f64,
        #[case] expected: Amount,
    ) {
        let io = GroupLimits {
            itpm: Some(Limit::Amount(Amount::Float(limit))),
            ..GroupLimits::default()
        };
        let reads = reads(Some(io), Some(totals(Some(1), None)), None);
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
            Some(totals(Some(3), None)),
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
    fn an_int_limit_past_any_machine_int_subtracts_exactly() {
        let limit: BigInt = "10".parse::<BigInt>().unwrap().pow(40);
        let reads = reads(
            Some(limits(
                Some(Limit::Amount(Amount::Int(limit.clone()))),
                None,
            )),
            None,
            Some(totals(Some(1), None)),
        );
        assert_eq!(
            headers(&reads),
            vec![
                ("x-ratelimit-remaining-tokens", Amount::Int(&limit - 1)),
                ("x-ratelimit-limit-tokens", Amount::Int(limit)),
            ]
        );
    }

    #[rstest]
    #[case::exact("9007199254740993", 9_007_199_254_740_992.0)]
    #[case::rounds_half_to_even("9007199254740995", 9_007_199_254_740_996.0)]
    #[case::negative("-3", -3.0)]
    fn an_int_converts_to_the_float_python_rounds_it_to(
        #[case] value: &str,
        #[case] expected: f64,
    ) {
        assert_eq!(int_as_float(&value.parse().unwrap()), Ok(expected));
    }

    #[rstest]
    fn a_float_limit_minus_an_int_too_large_for_a_float_raises_overflow() {
        let reads = reads(
            Some(limits(Some(Limit::Amount(Amount::Float(10.5))), None)),
            None,
            Some((Some("10".parse::<BigInt>().unwrap().pow(400)), None)),
        );
        let error = remaining_usage(&reads).unwrap_err();
        assert_eq!(error, RemainingError::IntTooLarge);
        assert_eq!(error.to_string(), "int too large to convert to float");
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
