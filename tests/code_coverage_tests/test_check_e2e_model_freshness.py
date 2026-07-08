"""Unit tests for check_e2e_model_freshness.py.

Each test feeds the check synthetic inputs and asserts it flags exactly the rot
it exists to catch: a pin vanishing from the pricing map, a pin nearing its
deprecation_date, the compose gateway config drifting from GATEWAY_MODELS, and
a hardcoded model literal sneaking into an e2e test.
"""

import datetime
import importlib.util
from pathlib import Path
from types import SimpleNamespace

CHECK_PATH = Path(__file__).resolve().parent / "check_e2e_model_freshness.py"


def _load_check():
    spec = importlib.util.spec_from_file_location("check_e2e_model_freshness", CHECK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = _load_check()
matrix = check.load_matrix()
ModelPin = matrix.ModelPin
TODAY = datetime.date(2026, 7, 8)


def _matrix_of(*pins, gateway=()):
    return SimpleNamespace(ALL_PINS=tuple(pins), GATEWAY_MODELS=tuple(gateway))


def _pricing(**entries):
    return {
        key: check.PricingEntry(deprecation_date=deprecation)
        for key, deprecation in entries.items()
    }


class TestExistence:
    def test_pin_missing_from_pricing_map_is_flagged(self):
        pin = ModelPin("gemini", "gemini-99-flash")
        violations = check.existence_violations(_matrix_of(pin), _pricing())
        assert len(violations) == 1
        assert "gemini/gemini-99-flash" in violations[0]

    def test_pin_resolving_via_prefixed_key_passes(self):
        pin = ModelPin("gemini", "gemini-99-flash")
        pricing = _pricing(**{"gemini/gemini-99-flash": None})
        assert check.existence_violations(_matrix_of(pin), pricing) == ()

    def test_pin_resolving_via_bare_model_id_passes(self):
        pin = ModelPin("openai", "gpt-99")
        pricing = _pricing(**{"gpt-99": None})
        assert check.existence_violations(_matrix_of(pin), pricing) == ()

    def test_pricing_key_override_is_used(self):
        pin = ModelPin("azure", "my-deployment-name", pricing_key="azure/gpt-99")
        pricing = _pricing(**{"azure/gpt-99": None})
        assert check.existence_violations(_matrix_of(pin), pricing) == ()


class TestDeprecation:
    def test_deprecation_within_window_is_flagged(self):
        pin = ModelPin("openai", "gpt-99")
        pricing = _pricing(**{"gpt-99": "2026-07-20"})
        violations = check.deprecation_violations(_matrix_of(pin), pricing, TODAY)
        assert len(violations) == 1
        assert "2026-07-20" in violations[0]

    def test_already_deprecated_is_flagged(self):
        pin = ModelPin("openai", "gpt-99")
        pricing = _pricing(**{"gpt-99": "2026-01-01"})
        assert len(check.deprecation_violations(_matrix_of(pin), pricing, TODAY)) == 1

    def test_deprecation_beyond_window_passes(self):
        pin = ModelPin("openai", "gpt-99")
        pricing = _pricing(**{"gpt-99": "2027-01-01"})
        assert check.deprecation_violations(_matrix_of(pin), pricing, TODAY) == ()

    def test_no_deprecation_date_passes(self):
        pin = ModelPin("openai", "gpt-99")
        pricing = _pricing(**{"gpt-99": None})
        assert check.deprecation_violations(_matrix_of(pin), pricing, TODAY) == ()


class TestGatewaySync:
    def _config(self, models, fallbacks=()):
        return check.GatewayConfig(
            model_list=[
                check.GatewayModelEntry(
                    model_name=name,
                    litellm_params=check.GatewayLiteLLMParams(model=backend),
                )
                for name, backend in models
            ],
            router_settings=check.GatewayRouterSettings(fallbacks=list(fallbacks)),
        )

    def test_matching_config_passes(self):
        pin = ModelPin("gemini", "gemini-99-flash")
        config = self._config([(pin.alias, pin.backend)])
        assert check.gateway_sync_violations(_matrix_of(gateway=(pin,)), config) == ()

    def test_stale_compose_model_is_flagged_both_ways(self):
        pin = ModelPin("gemini", "gemini-99-flash")
        config = self._config([("gemini-98-flash", "gemini/gemini-98-flash")])
        violations = check.gateway_sync_violations(_matrix_of(gateway=(pin,)), config)
        assert any("missing model_name 'gemini-99-flash'" in v for v in violations)
        assert any("has model_name 'gemini-98-flash'" in v for v in violations)

    def test_fallback_referencing_unknown_alias_is_flagged(self):
        pin = ModelPin("gemini", "gemini-99-flash")
        config = self._config(
            [(pin.alias, pin.backend)],
            fallbacks=[{pin.alias: ["gpt-99"]}],
        )
        violations = check.gateway_sync_violations(_matrix_of(gateway=(pin,)), config)
        assert len(violations) == 1
        assert "gpt-99" in violations[0]


class TestLiteralScan:
    def test_hardcoded_model_literal_is_flagged(self):
        source = 'MODEL = "gemini-2.5-flash"\n'
        violations = check.literal_violations_in_source(source, "suite/test_x.py")
        assert len(violations) == 1
        assert "suite/test_x.py:1" in violations[0]

    def test_pin_usage_is_not_flagged(self):
        source = (
            "from model_matrix import GEMINI_CHAT\n"
            "MODEL = GEMINI_CHAT.alias\n"
            'MESSAGE = f"model {GEMINI_CHAT.alias} denied"\n'
        )
        assert check.literal_violations_in_source(source, "suite/test_x.py") == ()

    def test_docstrings_and_comments_are_exempt(self):
        source = (
            '"""Drives gemini-2.5-flash against the proxy."""\n'
            "MODEL = None  # was gemini-2.5-flash\n"
        )
        assert check.literal_violations_in_source(source, "suite/test_x.py") == ()

    def test_rerank_literal_is_flagged(self):
        source = 'MODEL = "cohere/rerank-v3.5"\n'
        assert len(check.literal_violations_in_source(source, "suite/test_x.py")) == 1

    def test_syntax_newer_than_interpreter_is_still_scanned(self):
        source = (
            "def unwrap[R](result: Result[R]) -> R: ...\n"
            'MODEL = "gpt-4o-mini"\n'
        )
        violations = check.literal_violations_in_source(source, "suite/test_x.py")
        assert len(violations) == 1
        assert "suite/test_x.py:2" in violations[0]
