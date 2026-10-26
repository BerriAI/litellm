use std::{any::type_name, path::PathBuf, str::FromStr, time::Duration};

use crate::Invalid;

pub fn parse_str_bool(value: &str) -> Option<bool> {
    let token = value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    });
    if token.eq_ignore_ascii_case("true") {
        return Some(true);
    }
    token.eq_ignore_ascii_case("false").then_some(false)
}

/// `redis-py` string Booleans: only `1`, `true`, and `yes` (case-insensitive) are true.
pub fn parse_redis_bool(value: &str) -> bool {
    value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
}

pub trait FromEnv: Sized {
    fn from_env(value: &str) -> Option<Self>;
}

impl FromEnv for String {
    fn from_env(value: &str) -> Option<Self> {
        Some(value.to_owned())
    }
}

impl FromEnv for PathBuf {
    fn from_env(value: &str) -> Option<Self> {
        Some(value.into())
    }
}

impl FromEnv for bool {
    fn from_env(value: &str) -> Option<Self> {
        parse_str_bool(value)
    }
}

macro_rules! parsed_values {
    ($($ty:ty),* $(,)?) => {
        $(impl FromEnv for $ty {
            fn from_env(value: &str) -> Option<Self> {
                value.trim().parse().ok()
            }
        })*
    };
}

parsed_values!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64
);

pub fn value<T: FromEnv>(value: &str) -> Result<Option<T>, Invalid> {
    Ok(T::from_env(value))
}

pub fn truthy<T: for<'a> From<&'a str>>(value: &str) -> Result<Option<T>, Invalid> {
    Ok((!value.is_empty()).then(|| T::from(value)))
}

pub fn enabled(value: &str) -> Result<Option<bool>, Invalid> {
    Ok((parse_str_bool(value) == Some(true)).then_some(true))
}

pub fn parsed<T: FromStr>(value: &str) -> Result<Option<T>, Invalid> {
    Ok(value.trim().parse().ok())
}

pub fn required<T: FromStr>(value: &str) -> Result<Option<T>, Invalid> {
    value
        .trim()
        .parse()
        .map(Some)
        .map_err(|_| Invalid::expected(type_name::<T>()))
}

pub fn seconds(value: &str) -> Result<Option<Duration>, Invalid> {
    Ok(parsed::<u32>(value)?.map(|seconds| Duration::from_secs(u64::from(seconds))))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::lower("true", Some(true))]
    #[case::padded(" True ", Some(true))]
    #[case::off("false", None)]
    #[case::one("1", None)]
    fn enabled_only_switches_on_and_never_forces_off(
        #[case] value: &str,
        #[case] expected: Option<bool>,
    ) {
        assert_eq!(enabled(value), Ok(expected));
    }

    #[rstest]
    #[case::empty("", None)]
    #[case::kept_untrimmed(" value ", Some(" value "))]
    fn truthy_drops_only_empty_values(#[case] value: &str, #[case] expected: Option<&str>) {
        assert_eq!(truthy::<String>(value).unwrap().as_deref(), expected);
    }

    #[rstest]
    fn inferred_path_keeps_the_value_as_written() {
        assert_eq!(
            value::<PathBuf>(" /a b "),
            Ok(Some(PathBuf::from(" /a b ")))
        );
    }

    #[rstest]
    #[case::padded(" 45 ", Some(45))]
    #[case::zero("0", Some(0))]
    #[case::maximum("4294967295", Some(u32::MAX))]
    #[case::overflow("4294967296", None)]
    #[case::blank(" ", None)]
    #[case::word("soon", None)]
    #[case::fraction("0.5", None)]
    fn parsed_trims_and_treats_junk_as_unset(#[case] value: &str, #[case] expected: Option<u32>) {
        assert_eq!(parsed::<u32>(value), Ok(expected));
    }

    #[rstest]
    #[case::padded(" 8080 ", Ok(Some(8080)))]
    #[case::word("http", Err(Invalid::expected("u16")))]
    #[case::empty("", Err(Invalid::expected("u16")))]
    #[case::negative("-1", Err(Invalid::expected("u16")))]
    fn required_fails_on_junk(#[case] value: &str, #[case] expected: Result<Option<u16>, Invalid>) {
        assert_eq!(required::<u16>(value), expected);
    }

    #[rstest]
    #[case::whole("45", Some(Duration::from_secs(45)))]
    #[case::zero("0", Some(Duration::ZERO))]
    #[case::maximum("4294967295", Some(Duration::from_secs(u32::MAX as u64)))]
    #[case::overflow("4294967296", None)]
    #[case::junk("later", None)]
    fn seconds_reads_whole_seconds(#[case] value: &str, #[case] expected: Option<Duration>) {
        assert_eq!(seconds(value), Ok(expected));
    }

    #[rstest]
    #[case::trimmed_true(" True ", Some(true))]
    #[case::control_whitespace_true("\u{1c}TRUE\u{1f}", Some(true))]
    #[case::unicode_whitespace_false("\u{a0}False\u{2003}", Some(false))]
    #[case::zero_width_space("true\u{200b}", None)]
    #[case::yes("yes", None)]
    #[case::one("1", None)]
    #[case::empty("", None)]
    #[case::unknown("unknown", None)]
    fn boolean_tokens_follow_python_string_trimming_without_redis_tokens(
        #[case] input: &str,
        #[case] expected: Option<bool>,
    ) {
        assert_eq!(parse_str_bool(input), expected, "{input:?}");
    }
}
