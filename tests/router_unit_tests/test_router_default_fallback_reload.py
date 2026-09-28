"""Offline coverage for default fallback chains during router settings reloads."""

from litellm import Router


def test_default_fallbacks_survive_repeated_settings_updates():
    configured = [{"audio-a": []}, {"audio-b": []}]
    defaults = ["local-chat-model"]
    router = Router(model_list=[], fallbacks=configured, default_fallbacks=defaults)
    expected = [*configured, {"*": defaults}]
    assert router.fallbacks == expected
    assert configured == [{"audio-a": []}, {"audio-b": []}]

    for _ in range(2):
        router.update_settings(fallbacks=configured, default_fallbacks=defaults)
        assert router.fallbacks == expected
        assert router._get_first_default_fallback() == "local-chat-model"
        assert configured == [{"audio-a": []}, {"audio-b": []}]
        assert defaults == ["local-chat-model"]


def test_explicit_wildcard_fallback_wins_on_reload():
    configured = [{"audio-a": []}, {"*": ["explicit-model"]}]
    router = Router(model_list=[], fallbacks=configured, default_fallbacks=["default-model"])
    assert router.fallbacks == configured
    router.update_settings(fallbacks=configured, default_fallbacks=["default-model"])
    assert router.fallbacks == configured
    assert router._get_first_default_fallback() == "explicit-model"
    assert len([entry for entry in router.fallbacks if "*" in entry]) == 1
    router.update_settings(default_fallbacks=["changed-default"])
    assert router.fallbacks == configured
    assert configured == [{"audio-a": []}, {"*": ["explicit-model"]}]


def test_default_only_update_rebuilds_materialized_fallback():
    configured = [{"audio-a": []}]
    router = Router(model_list=[], fallbacks=configured, default_fallbacks=["old-model"])
    router.update_settings(default_fallbacks=["new-model"])
    assert router.fallbacks == [{"audio-a": []}, {"*": ["new-model"]}]
    router.update_settings(default_fallbacks=None)
    assert router.fallbacks == configured


def test_reloading_materialized_settings_does_not_duplicate_wildcard():
    router = Router(model_list=[], fallbacks=[{"audio-a": []}], default_fallbacks=["old-model"])
    for _ in range(2):
        router.update_settings(fallbacks=router.get_settings()["fallbacks"])
        assert router.fallbacks == [{"audio-a": []}, {"*": ["old-model"]}]
    # Both keyword orders should replace only the wildcard the router added.
    router.update_settings(fallbacks=router.fallbacks, default_fallbacks=["new-model"])
    assert router.fallbacks == [{"audio-a": []}, {"*": ["new-model"]}]
    router.update_settings(default_fallbacks=["newer-model"], fallbacks=router.fallbacks)
    assert router.fallbacks == [{"audio-a": []}, {"*": ["newer-model"]}]


def test_default_only_router_regenerates_after_reload():
    router = Router(model_list=[], default_fallbacks=["local-chat-model"])
    router.update_settings(fallbacks=None, default_fallbacks=["local-chat-model"])
    assert router.fallbacks == [{"*": ["local-chat-model"]}]
