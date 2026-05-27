"""Tests for custom_technical_keywords on ComplexityRouter (LIT-3237)."""
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.router_strategy.complexity_router.complexity_router import (
    ComplexityRouter,
)
from litellm.router_strategy.complexity_router.config import (
    DEFAULT_TECHNICAL_KEYWORDS,
)


BASE_CONFIG = {
    "tiers": {
        "SIMPLE": "gpt-4o-mini",
        "MEDIUM": "gpt-4o",
        "COMPLEX": "claude-sonnet-4-20250514",
        "REASONING": "o1-preview",
    },
}


def _router(extra_cfg=None):
    cfg = dict(BASE_CONFIG)
    if extra_cfg:
        cfg.update(extra_cfg)
    return ComplexityRouter(
        model_name="test",
        litellm_router_instance=MagicMock(),
        complexity_router_config=cfg,
    )


class TestCustomTechnicalKeywords:
    """Behavioural tests for ComplexityRouterConfig.custom_technical_keywords."""

    def test_defaults_unchanged_when_field_absent(self):
        r = _router()
        assert list(r.technical_keywords) == list(DEFAULT_TECHNICAL_KEYWORDS)

    def test_extends_defaults_not_replaces(self):
        custom = ["udp", "dns", "kafka", "postgresql"]
        r = _router({"custom_technical_keywords": custom})
        for kw in DEFAULT_TECHNICAL_KEYWORDS:
            assert kw in r.technical_keywords, f"default '{kw}' was dropped"
        for kw in custom:
            assert kw in r.technical_keywords, f"custom '{kw}' missing"
        unique_new = [k for k in custom if k.lower() not in {d.lower() for d in DEFAULT_TECHNICAL_KEYWORDS}]
        assert len(r.technical_keywords) == len(DEFAULT_TECHNICAL_KEYWORDS) + len(unique_new)

    def test_overrides_then_extends_with_technical_keywords_set(self):
        override = ["protocol", "encryption"]
        custom = ["udp", "dns"]
        r = _router({"technical_keywords": override, "custom_technical_keywords": custom})
        assert r.technical_keywords == ["protocol", "encryption", "udp", "dns"]
        assert "architecture" not in r.technical_keywords
        assert "microservice" not in r.technical_keywords

    def test_deduplicates_against_base_case_insensitive(self):
        custom = ["Architecture", "udp", "ARCHITECTURE", "udp"]
        r = _router({"custom_technical_keywords": custom})
        assert "architecture" in r.technical_keywords
        assert r.technical_keywords.count("udp") == 1
        assert sum(1 for k in r.technical_keywords if k.lower() == "architecture") == 1

    def test_preserves_order_of_custom_terms(self):
        custom = ["zeta", "alpha", "mu"]
        r = _router({"custom_technical_keywords": custom})
        tech = r.technical_keywords
        z_idx = tech.index("zeta")
        a_idx = tech.index("alpha")
        m_idx = tech.index("mu")
        assert z_idx < a_idx < m_idx
        last_default_idx = max(tech.index(d) for d in DEFAULT_TECHNICAL_KEYWORDS)
        assert last_default_idx < z_idx

    def test_empty_list_is_noop(self):
        r = _router({"custom_technical_keywords": []})
        assert list(r.technical_keywords) == list(DEFAULT_TECHNICAL_KEYWORDS)

    def test_runtime_classify_changes_with_custom_terms(self):
        prompt = (
            "Diagnose a UDP packet loss issue between two services over DNS, "
            "with TLS termination and SSH tunneling."
        )
        r_defaults = _router()
        _, _, signals_before = r_defaults.classify(prompt)
        assert not any(s.startswith("technical (") for s in signals_before)
        r_custom = _router({"custom_technical_keywords": ["udp", "dns", "tls", "ssh"]})
        _, _, signals_after = r_custom.classify(prompt)
        assert any(s.startswith("technical (") for s in signals_after)

    def test_arch_prompt_still_triggers_when_custom_added(self):
        prompt = (
            "Design a scalable distributed microservice architecture with "
            "strong authentication and authorization."
        )
        r = _router({"custom_technical_keywords": ["udp", "dns"]})
        _, _, signals = r.classify(prompt)
        assert any(s.startswith("technical (") for s in signals)
