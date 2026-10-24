use crate::Object;

#[macro_rules_attribute::apply(crate::config_section)]
pub struct Credential {
    pub credential_name: Option<String>,
    pub credential_info: Option<Object>,
    pub credential_values: Option<Object>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::Config;

    fn parse(yaml: &str) -> Config {
        Config::from_yaml(yaml).unwrap()
    }

    #[rstest]
    fn credential_entries_keep_name_info_and_values() {
        let config = parse(
            r#"
credential_list:
  - credential_name: azure-prod
    credential_info: {description: prod}
    credential_values: {api_key: os.environ/AZURE_API_KEY}
"#,
        );
        let credential = &config.credential_list[0];

        assert_eq!(credential.credential_name.as_deref(), Some("azure-prod"));
        assert!(
            credential
                .credential_info
                .as_ref()
                .unwrap()
                .contains_key("description")
        );
        assert!(
            credential
                .credential_values
                .as_ref()
                .unwrap()
                .contains_key("api_key")
        );
    }

    #[rstest]
    #[case::credential_values(
        r#"
credential_list:
  - credential_values: {api_key: SENTINEL}
"#
    )]
    #[case::credential_info(
        r#"
credential_list:
  - credential_info: {owner: SENTINEL}
"#
    )]
    fn debug_never_prints_credential_values(#[case] yaml: &str) {
        assert!(!format!("{:?}", parse(yaml)).contains("SENTINEL"));
    }
}
