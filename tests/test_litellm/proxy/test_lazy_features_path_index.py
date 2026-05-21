from litellm.proxy._lazy_features import (
    LAZY_FEATURES,
    LazyFeature,
    _build_prefix_trie,
    _build_suffix_rules,
    _find_unloaded_features_for_path,
)


def _feature_by_name(name: str) -> LazyFeature:
    return next(f for f in LAZY_FEATURES if f.name == name)


class TestLazyFeaturesPathIndex:
    def setup_method(self) -> None:
        self.trie = _build_prefix_trie(LAZY_FEATURES)
        self.suffix_rules = _build_suffix_rules(LAZY_FEATURES)
        self.loaded: set = set()

    def test_should_not_match_chat_completions_hot_path(self) -> None:
        matches = _find_unloaded_features_for_path(
            "/v1/chat/completions",
            self.trie,
            self.suffix_rules,
            self.loaded,
        )
        assert matches == []

    def test_should_match_guardrails_prefix(self) -> None:
        matches = _find_unloaded_features_for_path(
            "/guardrails/list",
            self.trie,
            self.suffix_rules,
            self.loaded,
        )
        assert len(matches) == 1
        assert matches[0].name == "guardrails"

    def test_should_match_mcp_prefix(self) -> None:
        matches = _find_unloaded_features_for_path(
            "/mcp",
            self.trie,
            self.suffix_rules,
            self.loaded,
        )
        assert len(matches) == 1
        assert matches[0].name == "mcp_app"

    def test_should_match_suffix_authorize_route(self) -> None:
        matches = _find_unloaded_features_for_path(
            "/my-mcp-server/authorize",
            self.trie,
            self.suffix_rules,
            self.loaded,
        )
        assert _feature_by_name("mcp_discoverable") in matches

    def test_should_skip_already_loaded_features(self) -> None:
        guardrails = _feature_by_name("guardrails")
        self.loaded.add(guardrails.module_path)
        matches = _find_unloaded_features_for_path(
            "/guardrails/list",
            self.trie,
            self.suffix_rules,
            self.loaded,
        )
        assert matches == []
