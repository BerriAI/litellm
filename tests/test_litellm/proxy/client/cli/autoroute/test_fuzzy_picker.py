from litellm.proxy.client.cli.commands.autoroute.fuzzy_picker import _fuzzy_match


class TestFuzzyMatch:
    def test_empty_query_returns_all_choices_in_order(self):
        choices = ("gpt-4o", "claude-opus", "o1")
        assert _fuzzy_match("", choices) == choices

    def test_exact_substring_matches(self):
        choices = ("gpt-4o-mini", "gpt-4o", "claude-opus")
        assert _fuzzy_match("gpt-4o", choices) == ("gpt-4o-mini", "gpt-4o")

    def test_matches_as_a_subsequence_not_just_a_substring(self):
        choices = ("gpt-4o-mini", "claude-opus")
        assert _fuzzy_match("g4om", choices) == ("gpt-4o-mini",)

    def test_is_case_insensitive(self):
        choices = ("GPT-4o-Mini",)
        assert _fuzzy_match("gpt4o", choices) == ("GPT-4o-Mini",)

    def test_no_match_excludes_the_choice(self):
        choices = ("gpt-4o", "claude-opus")
        assert _fuzzy_match("xyz", choices) == ()

    def test_preserves_original_relative_order_among_matches(self):
        choices = ("z-model", "a-model", "m-model")
        assert _fuzzy_match("model", choices) == ("z-model", "a-model", "m-model")
