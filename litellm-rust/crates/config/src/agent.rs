use crate::{Object, Spelled};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct Agent {
    pub agent_name: Option<String>,
    pub agent_card_params: Option<Object>,
    pub litellm_params: Option<Object>,
    pub identity: Option<Object>,
    pub enabled: Option<Spelled<bool>>,
    pub execution_mode: Option<String>,
    pub object_permission: Option<Object>,
    pub tpm_limit: Option<Spelled<i64>>,
    pub rpm_limit: Option<Spelled<i64>>,
    pub session_tpm_limit: Option<Spelled<i64>>,
    pub session_rpm_limit: Option<Spelled<i64>>,
    pub static_headers: Option<Object>,
    pub extra_headers: Option<Box<[String]>>,
    pub access_group_ids: Option<Box<[String]>>,
    pub kill_switch: Option<Object>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{Config, Spelled};

    fn parse(yaml: &str) -> Config {
        Config::from_yaml(yaml).unwrap()
    }

    #[rstest]
    #[case::agents("agents")]
    #[case::agent_list("agent_list")]
    fn agent_entries_keep_typed_limits_and_flags(#[case] section: &str) {
        let config = parse(&format!(
            "{section}:\n  - {{agent_name: reviewer, agent_card_params: {{name: x}}, tpm_limit: 100, enabled: false}}\n"
        ));
        let agent = config.agents.first().or(config.agent_list.first()).unwrap();

        assert_eq!(agent.agent_name.as_deref(), Some("reviewer"));
        assert!(
            agent
                .agent_card_params
                .as_ref()
                .unwrap()
                .contains_key("name")
        );
        assert_eq!(agent.tpm_limit, Some(Spelled::Value(100)));
        assert_eq!(agent.enabled, Some(Spelled::Value(false)));
    }

    #[rstest]
    #[case::agent_headers(
        r#"
agents:
  - static_headers: {Authorization: SENTINEL}
"#
    )]
    #[case::agent_params(
        r#"
agents:
  - litellm_params: {api_key: SENTINEL}
"#
    )]
    #[case::agent_identity(
        r#"
agent_list:
  - identity: {token: SENTINEL}
"#
    )]
    fn debug_never_prints_agent_values(#[case] yaml: &str) {
        assert!(!format!("{:?}", parse(yaml)).contains("SENTINEL"));
    }
}
