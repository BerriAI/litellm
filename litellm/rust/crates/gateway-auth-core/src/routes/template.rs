use std::sync::OnceLock;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Part {
    Literal(String),
    Segment,
    Path,
    /// A `:path` value the template follows with a `:` of its own, as the Google verb routes do.
    PathBeforeColon,
}

/// A route template matched as `RouteChecks._route_matches_pattern` matches it: a `{name}` takes
/// one or more characters up to a `/`, and a `{name:path}` one or more of any character, or of
/// any but `:` when the template has a `:` after it. Literal text is compared exactly.
#[derive(Debug)]
pub(super) struct Template(Box<[Part]>);

impl Template {
    pub(super) fn compile(template: &str) -> Self {
        let is_host = !template.starts_with('/');
        let mut parts = Vec::new();
        let mut rest = template;
        while let Some((before, name, convertor, after)) = next_param(rest) {
            parts.push(Part::Literal(before.to_owned()));
            parts.push(match convertor {
                None | Some("str") => Part::Segment,
                Some("path") if after.contains(':') => Part::PathBeforeColon,
                Some("path") => Part::Path,
                Some(other) => {
                    panic!("route template {template} uses an unported convertor {other} on {name}")
                }
            });
            rest = after;
        }
        let tail = if is_host {
            rest.split(':').next().unwrap_or_default()
        } else {
            rest
        };
        parts.push(Part::Literal(tail.to_owned()));
        Self(
            parts
                .into_iter()
                .filter(|part| *part != Part::Literal(String::new()))
                .collect(),
        )
    }

    pub(super) fn matches(&self, route: &str) -> bool {
        matches_parts(&self.0, route)
    }
}

fn is_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    chars
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The next `{name}` or `{name:convertor}`, as starlette's `PARAM_REGEX` finds it.
fn next_param(text: &str) -> Option<(&str, &str, Option<&str>, &str)> {
    text.match_indices('{').find_map(|(start, _)| {
        let end = start + text[start..].find('}')?;
        let inner = &text[start + 1..end];
        let (name, convertor) = match inner.split_once(':') {
            Some((name, convertor)) => (name, Some(convertor)),
            None => (inner, None),
        };
        (is_identifier(name) && convertor.is_none_or(is_identifier))
            .then(|| (&text[..start], name, convertor, &text[end + 1..]))
    })
}

fn matches_parts(parts: &[Part], route: &str) -> bool {
    let Some((part, rest)) = parts.split_first() else {
        return route.is_empty() || route == "\n";
    };
    let candidates = |limit: usize, min: usize| {
        route
            .char_indices()
            .map(|(index, _)| index)
            .chain([route.len()])
            .filter(move |index| *index >= min && *index <= limit)
    };
    match part {
        Part::Literal(literal) => route
            .strip_prefix(literal.as_str())
            .is_some_and(|after| matches_parts(rest, after)),
        Part::Segment => candidates(route.find('/').unwrap_or(route.len()), 1)
            .any(|index| matches_parts(rest, &route[index..])),
        Part::Path => candidates(route.len(), 1).any(|index| matches_parts(rest, &route[index..])),
        Part::PathBeforeColon => candidates(route.find(':').unwrap_or(route.len()), 1)
            .any(|index| matches_parts(rest, &route[index..])),
    }
}

/// Route templates named in the core, compiled on first use.
#[derive(Debug)]
pub(crate) struct RouteSet {
    templates: &'static [&'static str],
    compiled: OnceLock<Box<[Template]>>,
}

impl RouteSet {
    pub(crate) const fn new(templates: &'static [&'static str]) -> Self {
        Self {
            templates,
            compiled: OnceLock::new(),
        }
    }

    pub(crate) fn matches(&self, route: &str) -> bool {
        self.compiled
            .get_or_init(|| {
                self.templates
                    .iter()
                    .map(|template| Template::compile(template))
                    .collect()
            })
            .iter()
            .any(|template| template.matches(route))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::exact("/models", "/models", true)]
    #[case::exact_is_whole_route("/models", "/models/x", false)]
    #[case::segment("/files/{file_id}", "/files/abc", true)]
    #[case::segment_needs_a_char("/files/{file_id}/content", "/files//content", false)]
    #[case::segment_stops_at_slash("/files/{file_id}", "/files/a/b", false)]
    #[case::path_spans_slashes(
        "/v1beta/models/{model_name:path}:generateContent",
        "/v1beta/models/a/b:generateContent",
        true
    )]
    #[case::path_needs_a_char("/engines/{model:path}/completions", "/engines//completions", false)]
    #[case::path_spans_newlines(
        "/engines/{model:path}/chat/completions",
        "/engines/a\nb/chat/completions",
        true
    )]
    #[case::path_takes_colons_without_one_after(
        "/v1/mcp/server/{path:path}",
        "/v1/mcp/server/a:b",
        true
    )]
    #[case::path_stops_before_the_templates_colon(
        "/models/{model_name:path}:generateContent",
        "/models/a:b:generateContent",
        false
    )]
    #[case::path_never_swallows_another_verb(
        "/models/{model_name:path}:countTokens",
        "/models/a:generateContent:countTokens",
        false
    )]
    #[case::path_before_colon_needs_a_char(
        "/models/{model_name:path}:generateContent",
        "/models/:generateContent",
        false
    )]
    #[case::literal_after_param(
        "/claude-code/plugins/{plugin_name}/archive/{digest}.zip",
        "/claude-code/plugins/p/archive/d.zip",
        true
    )]
    #[case::dot_is_literal(
        "/claude-code/plugins/{plugin_name}/archive/{digest}.zip",
        "/claude-code/plugins/p/archive/dxzip",
        false
    )]
    #[case::question_mark_is_literal("/realtime?{model}", "/realtime?gpt", true)]
    #[case::trailing_newline("/models", "/models\n", true)]
    #[case::star_is_literal_in_templates("/a/*", "/a/b", false)]
    #[case::host_template_drops_port("chat_completion:80", "chat_completion", true)]
    fn templates_match_like_route_checks(
        #[case] template: &str,
        #[case] route: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(Template::compile(template).matches(route), expected);
    }
}
