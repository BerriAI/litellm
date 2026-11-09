use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Deserialize, PartialEq, Eq, strum::VariantArray)]
pub enum Kind {
    Proxy,
    Admin,
}

#[derive(Clone, Deserialize)]
pub struct Values {
    pub text: Option<String>,
    pub names: Vec<String>,
    pub failure: Option<String>,
}

#[derive(Clone, Deserialize)]
pub struct Facts {
    pub kind: Kind,
    pub route: String,
    pub present: bool,
    pub premium: bool,
    pub warned: bool,
    pub values: Option<Values>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind")]
pub enum Decision {
    Allow,
    Deny,
    Warn,
    ReadValues,
    Failure { message: String },
}

pub fn decide(facts: &Facts) -> Decision {
    if !facts.present {
        return Decision::Allow;
    }
    if !facts.premium {
        if !facts.warned {
            return Decision::Warn;
        }
        if facts.kind == Kind::Admin {
            return Decision::Allow;
        }
    }
    let Some(values) = &facts.values else {
        return Decision::ReadValues;
    };
    if let Some(message) = &values.failure {
        return Decision::Failure {
            message: message.clone(),
        };
    }
    let member = values.text.as_ref().map_or_else(
        || values.names.contains(&facts.route),
        |text| text.contains(&facts.route),
    );
    if member == (facts.kind == Kind::Proxy) {
        Decision::Allow
    } else {
        Decision::Deny
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::proxy_warn(Kind::Proxy, false, false, Decision::Warn)]
    #[case::admin_warn(Kind::Admin, false, false, Decision::Warn)]
    #[case::proxy_after_warn(Kind::Proxy, false, true, Decision::ReadValues)]
    #[case::admin_after_warn(Kind::Admin, false, true, Decision::Allow)]
    #[case::licensed_proxy(Kind::Proxy, true, false, Decision::ReadValues)]
    #[case::licensed_admin(Kind::Admin, true, false, Decision::ReadValues)]
    fn licensing_retains_distinct_warning_behavior(
        #[case] kind: Kind,
        #[case] premium: bool,
        #[case] warned: bool,
        #[case] expected: Decision,
    ) {
        assert_eq!(
            decide(&Facts {
                kind,
                route: "/responses".into(),
                present: true,
                premium,
                warned,
                values: None
            }),
            expected
        );
    }
}
