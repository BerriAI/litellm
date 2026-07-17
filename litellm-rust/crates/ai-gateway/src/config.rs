use crate::constants::ENV_REFERENCE_PREFIX;

pub(crate) fn resolve_env_reference(
    value: Option<&str>,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Option<String> {
    let value = value?;
    let Some(name) = value.strip_prefix(ENV_REFERENCE_PREFIX) else {
        return Some(value.to_string());
    };
    if name.trim().is_empty() {
        return None;
    }
    env_lookup(name).filter(|resolved| !resolved.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_lookup(name: &str) -> Option<String> {
        match name {
            "PRESENT" => Some("resolved".to_string()),
            "BLANK" => Some("   ".to_string()),
            _ => None,
        }
    }

    #[test]
    fn preserves_explicit_value() {
        assert_eq!(
            resolve_env_reference(Some("explicit"), &env_lookup),
            Some("explicit".to_string())
        );
    }

    #[test]
    fn preserves_value_that_only_contains_reference_prefix() {
        assert_eq!(
            resolve_env_reference(Some("prefix-os.environ/PRESENT"), &env_lookup),
            Some("prefix-os.environ/PRESENT".to_string())
        );
    }

    #[test]
    fn resolves_present_reference() {
        assert_eq!(
            resolve_env_reference(Some("os.environ/PRESENT"), &env_lookup),
            Some("resolved".to_string())
        );
    }

    #[test]
    fn missing_reference_is_absent() {
        assert_eq!(
            resolve_env_reference(Some("os.environ/MISSING"), &env_lookup),
            None
        );
    }

    #[test]
    fn blank_reference_value_is_absent() {
        assert_eq!(
            resolve_env_reference(Some("os.environ/BLANK"), &env_lookup),
            None
        );
    }

    #[test]
    fn malformed_reference_is_absent() {
        assert_eq!(
            resolve_env_reference(Some("os.environ/"), &env_lookup),
            None
        );
        assert_eq!(
            resolve_env_reference(Some("os.environ/   "), &env_lookup),
            None
        );
    }

    #[test]
    fn absent_input_stays_absent() {
        assert_eq!(resolve_env_reference(None, &env_lookup), None);
    }
}
