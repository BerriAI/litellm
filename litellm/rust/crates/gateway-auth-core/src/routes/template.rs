use std::sync::OnceLock;

use litellm_starlette::{Convertor, Param, RouteTemplate};

/// A route template matched as `RouteChecks._route_matches_pattern` matches it: every parameter
/// takes one or more characters up to a `/`, whatever its convertor, and a `{name:path}` one or
/// more of any character, or zero or more when the template has a `:` after it.
/// Literal text is compared exactly.
#[derive(Debug)]
pub(super) struct Template(RouteTemplate);

impl Template {
    pub(super) fn compile(template: &str) -> Self {
        Self(RouteTemplate::parse(template))
    }

    pub(super) fn matches(&self, route: &str) -> bool {
        self.0.matches_with(route, |param, rest| {
            let (shortest, limit) = match param {
                Param {
                    convertor: Convertor::Path,
                    colon_after: true,
                } => (0, rest.len()),
                Param {
                    convertor: Convertor::Path,
                    colon_after: false,
                } => (1, rest.len()),
                Param { .. } => (1, rest.find('/').unwrap_or(rest.len())),
            };
            Box::new(
                rest.char_indices()
                    .map(|(index, _)| index)
                    .chain([rest.len()])
                    .filter(move |index| *index >= shortest && *index <= limit),
            )
        })
    }

    /// The route matches the regex starlette's `compile_path` builds from the template, the
    /// matcher `_allowed_routes_check` reads route groups with.
    pub(super) fn matches_compiled(&self, route: &str) -> bool {
        self.0.matches(route)
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
    #[case::path_takes_colons_before_the_templates_colon(
        "/models/{model_name:path}:generateContent",
        "/models/a:b:generateContent",
        true
    )]
    #[case::path_takes_another_verb_before_its_own(
        "/models/{model_name:path}:countTokens",
        "/models/a:generateContent:countTokens",
        true
    )]
    #[case::path_still_ends_in_the_templates_verb(
        "/models/{model_name:path}:generateContent",
        "/models/a:generateContent:countTokens",
        false
    )]
    #[case::path_before_colon_may_be_empty(
        "/models/{model_name:path}:generateContent",
        "/models/:generateContent",
        true
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
    #[case::int_param_is_one_segment("/custom/{item_id:int}", "/custom/5", true)]
    #[case::int_param_takes_any_segment("/custom/{item_id:int}", "/custom/abc", true)]
    #[case::int_param_stops_at_slash("/custom/{item_id:int}", "/custom/5/x", false)]
    #[case::float_param_is_one_segment("/x/{value:float}/y", "/x/1.5/y", true)]
    #[case::uuid_param_is_one_segment(
        "/x/{id:uuid}",
        "/x/1b4e28ba-2fa1-11d2-883f-0016d3cca427",
        true
    )]
    #[case::one_trailing_newline_only("/models", "/models\n\n", false)]
    #[case::segment_takes_non_ascii("/files/{file_id}", "/files/\u{e9}", true)]
    #[case::two_segments("/a/{x}/b/{y}", "/a/1/b/2", true)]
    #[case::adjacent_params_need_a_char_each("/a/{x}{y}", "/a/b", false)]
    #[case::adjacent_params_split("/a/{x}{y}", "/a/bc", true)]
    #[case::int_param_backtracks_for_a_literal("/x/{id:int}5", "/x/125", true)]
    #[case::two_path_params_split("/{a:path}/{b:path}/end", "/x/y/z/end", true)]
    #[case::two_path_params_need_the_suffix("/{a:path}/{b:path}/end", "/x/y/z/ends", false)]
    #[case::two_path_params_need_a_char_each("/{a:path}/{b:path}/end", "//end", false)]
    fn templates_match_like_route_checks(
        #[case] template: &str,
        #[case] route: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(Template::compile(template).matches(route), expected);
    }

    /// Eight `path` parameters against a route that never ends in `/end`, which trying every
    /// split one at a time never finishes.
    #[test]
    fn many_path_params_answer_in_polynomial_time() {
        let params: Vec<String> = (0..8).map(|index| format!("{{p{index}:path}}")).collect();
        let template = Template::compile(&format!("/{}/end", params.join("/")));
        let route = format!("/{}", "x/".repeat(60));
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || sender.send(template.matches(&route)));
        assert_eq!(
            receiver
                .recv_timeout(std::time::Duration::from_secs(10))
                .ok(),
            Some(false)
        );
    }
}
