use crate::parse::{Single, Token, tokens};

#[derive(Clone, Debug)]
struct Segment(Box<[Single]>);

impl Segment {
    fn new(run: &[Token]) -> Self {
        Self(run.iter().filter_map(Token::single).cloned().collect())
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn matches(&self, window: &[char]) -> bool {
        self.len() == window.len()
            && self
                .0
                .iter()
                .zip(window)
                .all(|(single, c)| single.matches(*c))
    }

    /// What is left of `haystack` after the leftmost place this segment matches.
    fn skip_first_match<'a>(&self, haystack: &'a [char]) -> Option<&'a [char]> {
        let last_start = haystack.len().checked_sub(self.len())?;
        (0..=last_start)
            .find(|start| self.matches(&haystack[*start..start + self.len()]))
            .map(|start| &haystack[start + self.len()..])
    }
}

#[derive(Clone, Debug)]
enum Matcher {
    Exact(Segment),
    Wild {
        prefix: Segment,
        middles: Box<[Segment]>,
        suffix: Segment,
    },
}

/// A pattern in the dialect of Python's `fnmatch.fnmatchcase`, compiled once.
///
/// `*` matches any run of characters, `?` any one character, `[seq]` one character of a set that
/// may hold `lo-hi` ranges, and `[!seq]` one character outside it. Nothing escapes, so a class is
/// the only way to match a literal `*`, `?` or `[`. Every string is a pattern, and a `[` that no
/// `]` closes is a literal.
#[derive(Clone, Debug)]
pub struct Pattern(Matcher);

impl Pattern {
    pub fn new(pattern: &str) -> Self {
        let chars: Vec<char> = pattern.chars().collect();
        let tokens: Vec<Token> = tokens(&chars).collect();
        let runs: Vec<&[Token]> = tokens.split(Token::is_star).collect();
        Self(match runs.as_slice() {
            [prefix, middles @ .., suffix] => Matcher::Wild {
                prefix: Segment::new(prefix),
                middles: middles.iter().map(|run| Segment::new(run)).collect(),
                suffix: Segment::new(suffix),
            },
            _ => Matcher::Exact(Segment::new(&tokens)),
        })
    }

    pub fn matches(&self, name: &str) -> bool {
        let name: Vec<char> = name.chars().collect();
        match &self.0 {
            Matcher::Exact(segment) => segment.matches(&name),
            Matcher::Wild {
                prefix,
                middles,
                suffix,
            } => {
                let Some(free) = name.len().checked_sub(prefix.len() + suffix.len()) else {
                    return false;
                };
                let (head, rest) = name.split_at(prefix.len());
                let (between, tail) = rest.split_at(free);
                prefix.matches(head)
                    && suffix.matches(tail)
                    && middles
                        .iter()
                        .try_fold(between, |rest, middle| middle.skip_first_match(rest))
                        .is_some()
            }
        }
    }
}

/// Python's `fnmatch.fnmatchcase(name, pattern)`.
pub fn fnmatchcase(name: &str, pattern: &str) -> bool {
    Pattern::new(pattern).matches(name)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// Every expected value here and in the use-case tests below is what
    /// `fnmatch.fnmatchcase(name, pattern)` returned on CPython 3.12.13 and 3.14.7 on 2026-11-12.
    #[rstest]
    #[case::literal("admin", "admin", true)]
    #[case::literal_is_case_sensitive("Admin", "admin", false)]
    #[case::literal_needs_whole_name("admin-eu", "admin", false)]
    #[case::literal_needs_whole_pattern("adm", "admin", false)]
    #[case::empty_pattern_matches_empty_name("", "", true)]
    #[case::empty_pattern_rejects_a_name("a", "", false)]
    #[case::trailing_newline_is_not_ignored("a\n", "a", false)]
    #[case::dot_is_literal("axb", "a.b", false)]
    #[case::dot_matches_itself("a.b", "a.b", true)]
    #[case::plus_is_literal("aa", "a+", false)]
    #[case::parens_and_dollar_are_literal("(a)$", "(a)$", true)]
    #[case::braces_are_literal("{a,b}", "{a,b}", true)]
    #[case::braces_are_not_alternation("a", "{a,b}", false)]
    #[case::backslash_is_not_an_escape("a*", "a\\*", false)]
    #[case::backslash_matches_itself("a\\b", "a\\*", true)]
    #[case::question("ab", "a?", true)]
    #[case::question_needs_one("a", "a?", false)]
    #[case::question_takes_only_one("abc", "a?", false)]
    #[case::question_matches_newline("a\nb", "a?b", true)]
    #[case::question_matches_slash("a/b", "a?b", true)]
    #[case::question_counts_chars_not_bytes("aéb", "a?b", true)]
    #[case::star_alone_matches_empty("", "*", true)]
    #[case::star_alone_matches_anything("any thing/at\nall", "*", true)]
    #[case::star_suffix("admin-eu", "admin-*", true)]
    #[case::star_suffix_matches_empty("admin-", "admin-*", true)]
    #[case::star_suffix_needs_prefix("user-eu", "admin-*", false)]
    #[case::star_prefix("eu-admin", "*-admin", true)]
    #[case::star_prefix_needs_suffix("admin", "*x", false)]
    #[case::star_middle("a/b/c", "a*c", true)]
    #[case::star_middle_matches_empty("ac", "a*c", true)]
    #[case::star_spans_newline("a\nc", "a*c", true)]
    #[case::star_spans_space("a b c", "a*c", true)]
    #[case::double_star_is_one_star("abc", "a**c", true)]
    #[case::prefix_and_suffix_cannot_overlap("aba", "ab*ba", false)]
    #[case::prefix_and_suffix_may_touch("abba", "ab*ba", true)]
    #[case::star_reaches_the_last_occurrence("abcabc", "*abc", true)]
    #[case::two_stars_find_middle("mississippi", "*sip*", true)]
    #[case::three_stars_in_order("mississippi", "m*ss*ss*pi", true)]
    #[case::three_stars_out_of_order("mississippi", "m*pp*ss*i", false)]
    #[case::middle_must_fit_between_ends("mississippi", "mi*ssi*ssi*ssippi", false)]
    #[case::middle_with_question("a-xyz-b", "a*x?z*b", true)]
    #[case::star_then_class("team-7", "*-[0-9]", true)]
    #[case::url_wildcards("https://login.example.com/tenant", "https://*.example.com/*", true)]
    #[case::url_wildcards_need_host_suffix(
        "https://login.example.org/tenant",
        "https://*.example.com/*",
        false
    )]
    #[case::scope_wildcard("read:models", "read:*", true)]
    #[case::class("b1", "[abc]1", true)]
    #[case::class_rejects_non_member("d1", "[abc]1", false)]
    #[case::class_takes_one_char("ab", "[abc]", false)]
    #[case::class_needs_one_char("", "[abc]", false)]
    #[case::range("m", "[a-z]", true)]
    #[case::range_is_inclusive_low("a", "[a-z]", true)]
    #[case::range_is_inclusive_high("z", "[a-z]", true)]
    #[case::range_is_case_sensitive("M", "[a-z]", false)]
    #[case::two_ranges("7", "[a-z0-9]", true)]
    #[case::range_then_member("_", "[a-z_]", true)]
    #[case::unicode_range("é", "[à-ÿ]", true)]
    #[case::negated("x", "[!xyz]", false)]
    #[case::negated_other("q", "[!xyz]", true)]
    #[case::negated_range("m", "[!a-z]", false)]
    #[case::negated_range_other("5", "[!a-z]", true)]
    #[case::negated_needs_one_char("", "[!a]", false)]
    #[case::bang_after_first_is_literal("!", "[a!]", true)]
    #[case::double_bang_negates_bang("!", "[!!]", false)]
    #[case::double_bang_allows_others("a", "[!!]", true)]
    #[case::caret_is_literal("^", "[^a]", true)]
    #[case::caret_does_not_negate("b", "[^a]", false)]
    #[case::negated_caret("^", "[!^]", false)]
    #[case::leading_close_bracket("]", "[]]", true)]
    #[case::leading_close_bracket_then_member("a", "[]a]", true)]
    #[case::negated_leading_close_bracket("]", "[!]]", false)]
    #[case::negated_leading_close_bracket_other("a", "[!]]", true)]
    #[case::class_ends_at_first_close_bracket("a]", "[a]]", true)]
    #[case::trailing_hyphen_literal("-", "[a-]", true)]
    #[case::trailing_hyphen_keeps_member("a", "[a-]", true)]
    #[case::leading_hyphen_literal("-", "[-a]", true)]
    #[case::negated_leading_hyphen_literal("-", "[!-a]", false)]
    #[case::negated_leading_hyphen_is_no_range("#", "[!-a]", true)]
    #[case::range_from_hyphen("A", "[--a]", true)]
    #[case::range_to_hyphen("+", "[+--]", true)]
    #[case::hyphen_after_range_is_literal("-", "[a-c-e]", true)]
    #[case::hyphen_after_range_makes_no_second_range("d", "[a-c-e]", false)]
    #[case::member_after_range_and_hyphen("e", "[a-c-e]", true)]
    #[case::reversed_range_never_matches("m", "[z-a]", false)]
    #[case::reversed_range_rejects_its_ends("z", "[z-a]", false)]
    #[case::reversed_range_keeps_other_members("b", "[z-ab]", true)]
    #[case::negated_reversed_range_matches_any("m", "[!z-a]", true)]
    #[case::backslash_in_class_is_literal("\\", "[\\]", true)]
    #[case::backslash_is_a_range_bound("m", "[a\\-z]", true)]
    #[case::backslash_does_not_escape_hyphen("-", "[a\\-z]", false)]
    #[case::open_bracket_in_class("[", "[[]", true)]
    #[case::open_bracket_in_class_with_member("a", "[[a]", true)]
    #[case::star_in_class_is_literal("*", "[*]", true)]
    #[case::star_in_class_matches_no_other("a", "[*]", false)]
    #[case::question_in_class_is_literal("a", "[?]", false)]
    #[case::set_operators_are_literal("&", "[a&&b]", true)]
    #[case::tilde_and_pipe_are_literal("|", "[~|]", true)]
    #[case::posix_class_is_not_supported("x", "[[:alpha:]]", false)]
    #[case::posix_class_reads_as_members("a]", "[[:alpha:]]", true)]
    #[case::unclosed_bracket_literal("[a", "[a", true)]
    #[case::unclosed_bracket_matches_no_member("a", "[a", false)]
    #[case::empty_brackets_literal("[]", "[]", true)]
    #[case::unclosed_negation_literal("[!]", "[!]", true)]
    #[case::unclosed_bracket_then_star("[abc", "[a*", true)]
    fn matches_like_cpython(#[case] name: &str, #[case] pattern: &str, #[case] expected: bool) {
        assert_eq!(fnmatchcase(name, pattern), expected);
    }

    /// `jwt_litellm_role_map[].jwt_role` against one role from the token's roles claim.
    #[rstest]
    #[case::prefix_glob("AI_ADMIN_GLOBAL", "AI_ADMIN_*", true)]
    #[case::prefix_glob_matches_empty_tail("AI_ADMIN_", "AI_ADMIN_*", true)]
    #[case::prefix_glob_needs_the_separator("AI_ADMIN", "AI_ADMIN_*", false)]
    #[case::prefix_glob_is_anchored_at_start("XAI_ADMIN_EU", "AI_ADMIN_*", false)]
    #[case::prefix_glob_is_case_sensitive("ai_admin_global", "AI_ADMIN_*", false)]
    #[case::exact_role("AI_USER", "AI_USER", true)]
    #[case::exact_role_rejects_longer_role("AI_USER_EXTERNAL", "AI_USER", false)]
    #[case::catch_all("AI_VIEWER", "*", true)]
    #[case::catch_all_matches_empty_role("", "*", true)]
    #[case::prefix_glob_rejects_empty_role("", "AI_*", false)]
    #[case::role_with_spaces("AI Admin Team", "AI Admin*", true)]
    #[case::group_path("/litellm/admins/eu", "/litellm/admins/*", true)]
    #[case::group_path_other_group("/litellm/users/eu", "/litellm/admins/*", false)]
    #[case::group_path_star_spans_slashes("/litellm/admins/eu/west", "/litellm/*/west", true)]
    #[case::urn_role("urn:litellm:role:admin", "urn:litellm:role:*", true)]
    #[case::guid_role(
        "0f8fad5b-d9cb-469f-a165-70867728950e",
        "0f8fad5b-d9cb-469f-a165-70867728950e",
        true
    )]
    #[case::email_domain("admin@example.com", "*@example.com", true)]
    #[case::email_other_domain("admin@example.org", "*@example.com", false)]
    #[case::dotted_role("litellm.admin", "litellm.*", true)]
    #[case::dot_is_not_any_char("litellmXadmin", "litellm.admin", false)]
    #[case::one_char_suffix("AI_ADMIN_1", "AI_ADMIN_?", true)]
    #[case::one_char_suffix_rejects_two("AI_ADMIN_10", "AI_ADMIN_?", false)]
    #[case::class_of_teams("team-a-admin", "team-[ab]-admin", true)]
    #[case::class_of_teams_rejects_other("team-c-admin", "team-[ab]-admin", false)]
    #[case::bracketed_role_is_a_class_not_itself("[Admin]", "[Admin]", false)]
    #[case::bracketed_role_matches_one_member("A", "[Admin]", true)]
    #[case::glob_between_literals("AI_eu_ADMIN", "AI_*_ADMIN", true)]
    fn jwt_role_map_patterns(#[case] name: &str, #[case] pattern: &str, #[case] expected: bool) {
        assert_eq!(fnmatchcase(name, pattern), expected);
    }

    /// `routing_overrides` `iss`, `client_id`, `aud` and `scope` selectors against one claim value. The
    /// caller sends a selector here only when it holds `*` or `?`, and rejects whitespace in a
    /// non-scope claim before it does.
    #[rstest]
    #[case::client_id_suffix("svc-LITELLM", "*LITELLM", true)]
    #[case::client_id_suffix_alone("LITELLM", "*LITELLM", true)]
    #[case::client_id_suffix_is_case_sensitive("svc-litellm", "*LITELLM", false)]
    #[case::client_id_suffix_is_anchored_at_end("LITELLM-svc", "*LITELLM", false)]
    #[case::client_id_guid_prefix("0f8fad5b-d9cb-469f-a165-70867728950e", "0f8fad5b-*", true)]
    #[case::issuer_tenant_segment(
        "https://login.example.com/9188040d-6c67/v2.0",
        "https://login.example.com/*/v2.0",
        true
    )]
    #[case::issuer_tenant_other_version(
        "https://login.example.com/9188040d-6c67/v1.0",
        "https://login.example.com/*/v2.0",
        false
    )]
    #[case::issuer_dot_is_literal(
        "https://login.example.com/9188040d-6c67/v2x0",
        "https://login.example.com/*/v2.0",
        false
    )]
    #[case::issuer_subdomain("https://eu.idp.example.com", "https://*.idp.example.com", true)]
    #[case::issuer_subdomain_needs_a_label_dot(
        "https://idp.example.com",
        "https://*.idp.example.com",
        false
    )]
    #[case::issuer_subdomain_star_spans_slash(
        "https://evil.test/.idp.example.com",
        "https://*.idp.example.com",
        true
    )]
    #[case::issuer_trailing_star_admits_longer_host(
        "https://idp.example.com.evil.test",
        "https://idp.example.com*",
        true
    )]
    #[case::issuer_path_star(
        "https://idp.example.com/realms/prod",
        "https://idp.example.com/*",
        true
    )]
    #[case::issuer_path_star_matches_bare_slash(
        "https://idp.example.com/",
        "https://idp.example.com/*",
        true
    )]
    #[case::issuer_path_star_needs_the_slash(
        "https://idp.example.com",
        "https://idp.example.com/*",
        false
    )]
    #[case::issuer_port_and_realm(
        "https://idp.example.com:8443/realms/prod",
        "https://idp.example.com:*/realms/*",
        true
    )]
    #[case::issuer_scheme_is_literal(
        "http://idp.example.com/realms/prod",
        "https://idp.example.com/*",
        false
    )]
    #[case::issuer_one_char("https://idp-1.example.com", "https://idp-?.example.com", true)]
    #[case::issuer_one_char_rejects_two(
        "https://idp-10.example.com",
        "https://idp-?.example.com",
        false
    )]
    #[case::query_mark_matches_itself(
        "https://idp.example.com/auth?tenant=a",
        "https://idp.example.com/auth?tenant=a",
        true
    )]
    #[case::query_mark_matches_any_char(
        "https://idp.example.com/auth/tenant=a",
        "https://idp.example.com/auth?tenant=a",
        true
    )]
    #[case::ipv6_host_is_a_class("https://[::1]:8443/realms/prod", "https://[::1]:8443/*", false)]
    #[case::ipv6_host_class_takes_one_member(
        "https://::8443/realms/prod",
        "https://[::1]:8443/*",
        true
    )]
    #[case::star_spans_whitespace("trusted.example.com evil.test", "trusted.*", true)]
    #[case::scope_prefix("App:LiteLLM", "App:*", true)]
    #[case::scope_prefix_other_app("Api:LiteLLM", "App:*", false)]
    #[case::scope_suffix("read:models", "*:models", true)]
    #[case::scope_dotted_permission("App:LiteLLM.Read", "App:LiteLLM.*", true)]
    #[case::scope_dotted_permission_needs_dot("App:LiteLLM", "App:LiteLLM.*", false)]
    #[case::scope_uri_default("api://litellm/.default", "api://litellm/*", true)]
    #[case::audience_scheme("api://litellm", "api://*", true)]
    #[case::audience_env_suffix("api://litellm-prod", "api://litellm-*", true)]
    #[case::audience_env_suffix_other_app("api://other-prod", "api://litellm-*", false)]
    fn routing_override_selectors(
        #[case] name: &str,
        #[case] pattern: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(fnmatchcase(name, pattern), expected);
    }

    /// `cost_discount_config` model patterns and wildcard `model_name` routes against a model name.
    #[rstest]
    #[case::provider_wildcard("openai/gpt-5", "openai/*", true)]
    #[case::provider_wildcard_matches_empty_model("openai/", "openai/*", true)]
    #[case::provider_wildcard_needs_the_slash("openai", "openai/*", false)]
    #[case::provider_wildcard_other_provider("anthropic/claude-sonnet-4-5", "openai/*", false)]
    #[case::provider_wildcard_is_case_sensitive("OpenAI/gpt-5", "openai/*", false)]
    #[case::provider_wildcard_spans_nested_slashes(
        "or-team/openai/gpt-5.4-mini",
        "or-team/*",
        true
    )]
    #[case::catch_all("bedrock/us-east-1/anthropic.claude-sonnet-4-5", "*", true)]
    #[case::family_prefix("claude-sonnet-4-5@20250929", "claude-*", true)]
    #[case::family_prefix_other_family("gemini-2.5-pro", "claude-*", false)]
    #[case::family_prefix_without_hyphen("claude-opus-4-1", "claude-opus*", true)]
    #[case::narrower_family_prefix("claude-sonnet-4-5", "claude-sonnet-*", true)]
    #[case::narrower_family_prefix_other_tier("claude-opus-4-1", "claude-sonnet-*", false)]
    #[case::partial_name_prefix(
        "openai/gpt-partial-route-9-mini",
        "openai/gpt-partial-route-9*",
        true
    )]
    #[case::partial_name_prefix_other_number(
        "openai/gpt-partial-route-8",
        "openai/gpt-partial-route-9*",
        false
    )]
    #[case::bedrock_vendor_prefix("anthropic.claude-3-5-sonnet-20241022-v2:0", "anthropic.*", true)]
    #[case::bedrock_vendor_prefix_rejects_region_profile(
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "anthropic.*",
        false
    )]
    #[case::bedrock_any_region_profile(
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "*anthropic.claude-*",
        true
    )]
    #[case::bedrock_region_path(
        "us-east-1/anthropic.claude-3-haiku-20240307-v1:0",
        "us-east-1/anthropic.claude-*",
        true
    )]
    #[case::bedrock_other_region_path(
        "eu-west-1/anthropic.claude-3-haiku-20240307-v1:0",
        "us-east-1/anthropic.claude-*",
        false
    )]
    #[case::bedrock_dotted_family("meta.llama3-70b-instruct-v1:0", "meta.llama3-*", true)]
    #[case::vertex_version_suffix("claude-3-5-haiku@20241022", "claude-3-5-haiku@*", true)]
    #[case::any_vendor_fixed_model("openai/auto", "*/auto", true)]
    #[case::any_vendor_fixed_model_needs_vendor("auto", "*/auto", false)]
    #[case::fine_tune_id("ft:gpt-4o-mini:my-org::abc123", "ft:gpt-4o-mini:*", true)]
    #[case::double_colon_separators("openai/fo::a::static::b", "openai/fo::*::static::*", true)]
    #[case::double_colon_separators_need_both_colons(
        "openai/fo::a:static::b",
        "openai/fo::*::static::*",
        false
    )]
    #[case::at_least_one_char_after_prefix("gemini-2.5-flash", "gemini-?*", true)]
    #[case::at_least_one_char_after_prefix_rejects_none("gemini-", "gemini-?*", false)]
    #[case::minor_version_one_char("gemini-2.5-pro", "gemini-2.?-pro", true)]
    #[case::minor_version_one_char_rejects_two("gemini-2.10-pro", "gemini-2.?-pro", false)]
    #[case::major_version_class("gpt-4o-mini", "gpt-[45]*", true)]
    #[case::major_version_class_rejects_other("gpt-3.5-turbo", "gpt-[45]*", false)]
    #[case::letter_class_then_star("claude-sonnet", "claude-[abcdefghijklmnopqrstuvwxyz]*", true)]
    #[case::letter_class_rejects_digit(
        "claude-3-opus",
        "claude-[abcdefghijklmnopqrstuvwxyz]*",
        false
    )]
    #[case::digit_range_then_star("o3-mini", "o[0-9]*", true)]
    #[case::digit_range_rejects_letter("omni-moderation", "o[0-9]*", false)]
    #[case::unclosed_bracket_is_literal("weird[", "weird[", true)]
    #[case::unclosed_negation_is_literal("gemini-[!]-pro", "gemini-[!]*", true)]
    #[case::unclosed_negation_rejects_plain_name("gemini-pro", "gemini-[!]*", false)]
    #[case::stray_close_bracket_is_literal("a]x", "a]*", true)]
    #[case::class_holding_close_bracket("a]pro", "a[]x]*", true)]
    #[case::class_holding_close_bracket_other_member("axpro", "a[]x]*", true)]
    #[case::class_holding_close_bracket_non_member("abpro", "a[]x]*", false)]
    #[case::regex_group_is_literal("gpt-5", "(.*)", false)]
    #[case::regex_group_matches_only_its_own_shape("(.x)", "(.*)", true)]
    #[case::regex_route_is_literal("openai/gpt-5", "openai/(.*)", false)]
    fn model_name_patterns(#[case] name: &str, #[case] pattern: &str, #[case] expected: bool) {
        assert_eq!(fnmatchcase(name, pattern), expected);
    }

    /// Header allowlist globs against a header name the caller has lowercased.
    #[rstest]
    #[case::vendor_header("x-stainless-arch", "x-stainless-*", true)]
    #[case::vendor_header_bare_prefix("x-stainless-", "x-stainless-*", true)]
    #[case::vendor_header_needs_hyphen("x-stainless", "x-stainless-*", false)]
    #[case::litellm_header("x-litellm-trace-id", "x-litellm-*", true)]
    #[case::prefix_is_anchored_at_start("my-x-litellm-trace-id", "x-litellm-*", false)]
    #[case::other_header("authorization", "x-litellm-*", false)]
    #[case::caller_must_lowercase_the_name("X-Stainless-Arch", "x-stainless-*", false)]
    fn header_allowlist_patterns(
        #[case] name: &str,
        #[case] pattern: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(fnmatchcase(name, pattern), expected);
    }

    /// The one divergence. CPython looks for the negating `!` after it has dropped reversed ranges,
    /// so it answers the opposite for each of these.
    #[rstest]
    #[case::does_not_become_match_any("a", "[z-a!]", false)]
    #[case::does_not_negate_what_follows("b", "[z-a!b]", true)]
    #[case::does_not_admit_non_members("a", "[z-a!b]", false)]
    fn bang_after_a_reversed_range_is_a_member(
        #[case] name: &str,
        #[case] pattern: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(fnmatchcase(name, pattern), expected);
    }

    /// Retrying every start per `*` would need about 2000^20 steps here.
    #[rstest]
    #[case::no_match("", false)]
    #[case::match_at_the_end("b", true)]
    fn many_stars_on_a_long_name_finish(#[case] ending: &str, #[case] expected: bool) {
        let name = format!("{}{ending}", "a".repeat(2000));
        let pattern = format!("{}b", "*a".repeat(20));
        assert_eq!(fnmatchcase(&name, &pattern), expected);
    }
}
