use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use litellm_starlette::{Convertor, Param, RouteTemplate, TemplatePart};
use rstest::rstest;

fn literal(text: &str) -> TemplatePart {
    TemplatePart::Literal(text.into())
}

fn param(convertor: Convertor, colon_after: bool) -> TemplatePart {
    TemplatePart::Param(Param {
        convertor,
        colon_after,
    })
}

#[rstest]
#[case::plain("/models", vec![literal("/models")])]
#[case::segment("/files/{file_id}", vec![literal("/files/"), param(Convertor::Str, false)])]
#[case::int("/custom/{item_id:int}", vec![literal("/custom/"), param(Convertor::Int, false)])]
#[case::float("/x/{v:float}/y", vec![literal("/x/"), param(Convertor::Float, false), literal("/y")])]
#[case::uuid("/x/{id:uuid}", vec![literal("/x/"), param(Convertor::Uuid, false)])]
#[case::unknown_convertor_is_str("/x/{id:slug}", vec![literal("/x/"), param(Convertor::Str, false)])]
#[case::path_before_colon(
    "/models/{m:path}:generateContent",
    vec![literal("/models/"), param(Convertor::Path, true), literal(":generateContent")]
)]
#[case::colon_after_counts_a_later_convertor(
    "/a/{x:path}{y:int}",
    vec![literal("/a/"), param(Convertor::Path, true), param(Convertor::Int, false)]
)]
#[case::not_a_param("/x/{1}", vec![literal("/x/{1}")])]
#[case::host_drops_port("chat_completion:80", vec![literal("chat_completion")])]
fn templates_parse(#[case] template: &str, #[case] expected: Vec<TemplatePart>) {
    assert_eq!(RouteTemplate::parse(template).parts(), expected.as_slice());
}

/// Expected values from starlette 1.3.1: `bool(compile_path(template)[0].match(route))`.
#[rstest]
#[case::exact("/models", "/models", true)]
#[case::trailing_newline("/models", "/models\n", true)]
#[case::one_trailing_newline_only("/models", "/models\n\n", false)]
#[case::str_needs_a_char("/files/{file_id}", "/files/", false)]
#[case::str_stops_at_slash("/files/{file_id}", "/files/a/b", false)]
#[case::str_takes_a_newline("/files/{file_id}", "/files/a\nb", true)]
#[case::str_takes_non_ascii("/files/{file_id}", "/files/\u{e9}", true)]
#[case::two_str_params("/a/{x}/b/{y}", "/a/1/b/2", true)]
#[case::adjacent_str_params_need_a_char_each("/a/{x}{y}", "/a/b", false)]
#[case::adjacent_str_params_split("/a/{x}{y}", "/a/bc", true)]
#[case::path_spans_slashes("/engines/{m:path}/completions", "/engines/a/b/completions", true)]
#[case::path_may_be_empty(
    "/v1beta/models/{m:path}:generateContent",
    "/v1beta/models/:generateContent",
    true
)]
#[case::path_swallows_colons(
    "/v1beta/models/{m:path}:generateContent",
    "/v1beta/models/a:b:generateContent",
    true
)]
#[case::path_stops_at_a_newline(
    "/engines/{m:path}/completions",
    "/engines/a\nb/completions",
    false
)]
#[case::two_path_params_split("/{a:path}/{b:path}/end", "/x/y/z/end", true)]
#[case::two_path_params_need_the_suffix("/{a:path}/{b:path}/end", "/x/y/z/ends", false)]
#[case::two_path_params_need_both_slashes("/{a:path}/{b:path}/end", "//end", false)]
#[case::int_takes_digits("/x/{id:int}", "/x/42", true)]
#[case::int_needs_a_digit("/x/{id:int}", "/x/", false)]
#[case::int_refuses_letters("/x/{id:int}", "/x/4a", false)]
#[case::int_refuses_non_ascii_digits("/x/{id:int}", "/x/\u{663}", false)]
#[case::int_backtracks_for_a_literal("/x/{id:int}5", "/x/125", true)]
#[case::float_with_fraction("/x/{v:float}", "/x/1.5", true)]
#[case::float_without_fraction("/x/{v:float}", "/x/1", true)]
#[case::float_needs_whole_digits("/x/{v:float}", "/x/.5", false)]
#[case::float_needs_fraction_digits("/x/{v:float}", "/x/1.", false)]
#[case::float_takes_one_fraction("/x/{v:float}/y", "/x/1.5.5/y", false)]
#[case::float_backtracks_inside_the_fraction("/x/{v:float}0", "/x/1.50", true)]
#[case::float_leaves_the_fraction_to_a_literal("/x/{v:float}.5", "/x/1.5", true)]
#[case::uuid_with_dashes("/x/{id:uuid}", "/x/1b4e28ba-2fa1-11d2-883f-0016D3CCA427", true)]
#[case::uuid_without_dashes("/x/{id:uuid}", "/x/1b4e28ba2fa111d2883f0016d3cca427", true)]
#[case::uuid_too_short("/x/{id:uuid}", "/x/1b4e28ba-2fa1-11d2-883f-0016d3cca42", false)]
#[case::uuid_too_long("/x/{id:uuid}", "/x/1b4e28ba-2fa1-11d2-883f-0016d3cca427a", false)]
#[case::uuid_then_literal("/x/{id:uuid}/y", "/x/1b4e28ba-2fa1-11d2-883f-0016d3cca427/y", true)]
#[case::dot_is_literal("/a/{x}.zip", "/a/bxzip", false)]
fn templates_match_like_compile_path(
    #[case] template: &str,
    #[case] route: &str,
    #[case] expected: bool,
) {
    assert_eq!(RouteTemplate::parse(template).matches(route), expected);
}

fn answer_within(budget: Duration, check: impl FnOnce() -> bool + Send + 'static) -> Option<bool> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || sender.send(check()));
    receiver.recv_timeout(budget).ok()
}

/// A pass-through `path` with eight `path` parameters against a route that never ends in `/end`.
/// Trying every split one at a time never finishes. Expected value from starlette 1.3.1.
#[test]
fn many_path_params_answer_in_polynomial_time() {
    let params: Vec<String> = (0..8).map(|index| format!("{{p{index}:path}}")).collect();
    let template = RouteTemplate::parse(&format!("/{}/end", params.join("/")));
    let route = format!("/{}", "x/".repeat(60));
    assert_eq!(
        answer_within(Duration::from_secs(10), move || template.matches(&route)),
        Some(false)
    );
}
