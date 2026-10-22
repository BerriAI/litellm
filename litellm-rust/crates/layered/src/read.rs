use std::{any::type_name, str::FromStr, time::Duration};

use litellm_core_utils::serde_compat::parse_str_bool;

use crate::Invalid;

pub fn text<T: for<'a> From<&'a str>>(value: &str) -> Result<Option<T>, Invalid> {
    Ok(Some(T::from(value)))
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
    fn text_keeps_the_value_as_written() {
        assert_eq!(text::<PathBuf>(" /a b "), Ok(Some(PathBuf::from(" /a b "))));
    }

    #[rstest]
    #[case::padded(" 45 ", Some(45))]
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
    #[case::junk("later", None)]
    fn seconds_reads_whole_seconds(#[case] value: &str, #[case] expected: Option<Duration>) {
        assert_eq!(seconds(value), Ok(expected));
    }
}
