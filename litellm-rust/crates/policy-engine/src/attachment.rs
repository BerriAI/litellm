use crate::MatchContext;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Attachment {
    pub policy: String,
    pub global: bool,
    pub teams: Vec<String>,
    pub keys: Vec<String>,
    pub models: Vec<String>,
    pub tags: Vec<String>,
    pub priority: Option<i32>,
    pub default: bool,
}

pub(crate) fn matches_pattern(value: Option<&str>, patterns: &[String]) -> bool {
    let Some(value) = value else {
        return patterns.iter().any(|pattern| pattern == "*");
    };
    patterns
        .iter()
        .any(|pattern| match pattern.strip_suffix('*') {
            Some(prefix) => value.starts_with(prefix),
            None => value == pattern,
        })
}

fn matches_or_wildcard(value: Option<&str>, patterns: &[String]) -> bool {
    if patterns.is_empty() {
        return true;
    }
    matches_pattern(value, patterns)
}

impl Attachment {
    pub(crate) fn in_scope(&self, context: &MatchContext) -> bool {
        if self.global {
            return true;
        }
        matches_or_wildcard(context.team_alias.as_deref(), &self.teams)
            && matches_or_wildcard(context.key_alias.as_deref(), &self.keys)
            && matches_or_wildcard(context.model.as_deref(), &self.models)
            && (self.tags.is_empty()
                || context
                    .tags
                    .iter()
                    .any(|tag| matches_pattern(Some(tag), &self.tags)))
    }

    fn specificity(&self) -> (u8, u8) {
        if self.global {
            return (0, 0);
        }
        let dimensions = [
            (&self.teams, 1),
            (&self.keys, 2),
            (&self.tags, 3),
            (&self.models, 4),
        ]
        .into_iter()
        .filter(|(values, _)| !values.is_empty())
        .map(|(_, specificity)| specificity);
        dimensions.fold((0, 0), |(max, count), specificity| {
            (max.max(specificity), count + 1)
        })
    }

    pub(crate) fn order(&self) -> (bool, i32, u8, u8) {
        let (specificity, dimensions) = self.specificity();
        (
            self.priority.is_none(),
            self.priority.unwrap_or(0),
            specificity,
            dimensions,
        )
    }

    pub(crate) fn match_reason(&self, context: &MatchContext) -> String {
        let reason = self.scope_reason(context);
        if self.default {
            format!("default:{reason}")
        } else {
            reason
        }
    }

    fn scope_reason(&self, context: &MatchContext) -> String {
        if self.global {
            return "scope:*".to_owned();
        }
        let tag = (!self.tags.is_empty())
            .then(|| {
                context
                    .tags
                    .iter()
                    .find(|tag| matches_pattern(Some(tag), &self.tags))
            })
            .flatten()
            .map(|tag| format!("tag:{tag}"));
        let named = |values: &[String], label: &str, value: &Option<String>| {
            value
                .as_deref()
                .filter(|value| !values.is_empty() && !value.is_empty())
                .map(|value| format!("{label}:{value}"))
        };
        let reasons: Vec<String> = [
            tag,
            named(&self.teams, "team", &context.team_alias),
            named(&self.keys, "key", &context.key_alias),
            named(&self.models, "model", &context.model),
        ]
        .into_iter()
        .flatten()
        .collect();
        if reasons.is_empty() {
            "scope:default".to_owned()
        } else {
            reasons.join("+")
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[rstest]
    #[case::none_matches_only_star(None, &["*"], true)]
    #[case::none_misses_prefix(None, &["team-*"], false)]
    #[case::prefix(Some("team-a"), &["team-*"], true)]
    #[case::star_is_empty_prefix(Some(""), &["*"], true)]
    #[case::exact(Some("team-a"), &["team-a"], true)]
    #[case::star_only_at_end(Some("team-a"), &["*-a"], false)]
    #[case::any_of(Some("b"), &["a", "b"], true)]
    fn wildcard_patterns(
        #[case] value: Option<&str>,
        #[case] patterns: &[&str],
        #[case] expected: bool,
    ) {
        assert_eq!(matches_pattern(value, &strings(patterns)), expected);
    }

    fn context() -> MatchContext {
        MatchContext {
            team_alias: Some("team-a".to_owned()),
            key_alias: Some(String::new()),
            model: Some("gpt-4".to_owned()),
            tags: strings(&["env:dev", "health-x"]),
        }
    }

    #[rstest]
    #[case::global(Attachment { global: true, teams: strings(&["nope"]), ..Default::default() }, true, "scope:*")]
    #[case::unscoped_matches_everything(Attachment::default(), true, "scope:default")]
    #[case::team(Attachment { teams: strings(&["team-*"]), ..Default::default() }, true, "team:team-a")]
    #[case::empty_key_alias_matches_but_gives_no_reason(Attachment { keys: strings(&["*"]), ..Default::default() }, true, "scope:default")]
    #[case::first_matching_tag_then_dimensions(Attachment { tags: strings(&["health-*", "env:*"]), models: strings(&["gpt-4"]), ..Default::default() }, true, "tag:env:dev+model:gpt-4")]
    #[case::tag_miss(Attachment { tags: strings(&["prod"]), ..Default::default() }, false, "scope:default")]
    #[case::default_prefix(Attachment { teams: strings(&["team-a"]), default: true, ..Default::default() }, true, "default:team:team-a")]
    fn scope_and_reason(
        #[case] attachment: Attachment,
        #[case] in_scope: bool,
        #[case] reason: &str,
    ) {
        assert_eq!(attachment.in_scope(&context()), in_scope);
        assert_eq!(attachment.match_reason(&context()), reason);
    }

    #[test]
    fn tags_never_match_an_untagged_request() {
        let attachment = Attachment {
            tags: strings(&["*"]),
            ..Default::default()
        };
        assert!(!attachment.in_scope(&MatchContext::default()));
    }

    #[rstest]
    #[case::priority_before_none(Attachment { priority: Some(9), models: strings(&["m"]), ..Default::default() }, Attachment { global: true, ..Default::default() })]
    #[case::lower_priority_first(Attachment { priority: Some(-1), ..Default::default() }, Attachment { priority: Some(0), ..Default::default() })]
    #[case::global_before_scoped(Attachment { global: true, teams: strings(&["t"]), ..Default::default() }, Attachment { teams: strings(&["t"]), ..Default::default() })]
    #[case::broader_dimension_first(Attachment { teams: strings(&["t"]), ..Default::default() }, Attachment { keys: strings(&["k"]), ..Default::default() })]
    #[case::fewer_dimensions_first(Attachment { models: strings(&["m"]), ..Default::default() }, Attachment { models: strings(&["m"]), teams: strings(&["t"]), ..Default::default() })]
    #[case::most_specific_dimension_decides(Attachment { tags: strings(&["x"]), keys: strings(&["k"]), teams: strings(&["t"]), ..Default::default() }, Attachment { models: strings(&["m"]), ..Default::default() })]
    fn ordering(#[case] earlier: Attachment, #[case] later: Attachment) {
        assert!(earlier.order() < later.order());
    }
}
