import pytest
from pydantic import ValidationError

from litellm.proxy._types import ProxyException
from litellm.proxy.common_utils.request_tag_mappings import resolve_baggage_tag_mappings
from litellm.types.proxy.request_tag_mappings import RequestTagMapping


@pytest.mark.parametrize(
    "header,value",
    [
        ("entrypoint=a", "a"),
        ("entrypoint = a", "a"),
        ("region=sg, entrypoint=customer%20support;version=1", "customer support"),
    ],
)
def test_variable_mapper_uses_decoded_baggage_value(header, value):
    rule = RequestTagMapping(key="entrypoint", tag_prefix="usecase:")
    assert resolve_baggage_tag_mappings(header, [rule]) == ((f"usecase:{value}", False),)


@pytest.mark.parametrize(
    "header,expected",
    [("entrypoint=a", "department:support"), ("entrypoint=b", "department:sales"), ("entrypoint=x", "usecase:x")],
)
def test_exact_rules_override_variable_fallback_independent_of_order(header, expected):
    rules = [
        RequestTagMapping(key="entrypoint", tag_prefix="usecase:"),
        RequestTagMapping(key="entrypoint", value="a", tag="department:support"),
        RequestTagMapping(key="entrypoint", value="b", tag="department:sales"),
    ]
    assert resolve_baggage_tag_mappings(header, rules) == ((expected, False),)
    assert resolve_baggage_tag_mappings(header, list(reversed(rules))) == ((expected, False),)


@pytest.mark.parametrize("header", ["entrypoint=%00", "entrypoint=a%0Ab", "entrypoint=b,entry%70oint=a"])
def test_variable_mapper_skips_control_characters_and_percent_encoded_duplicate_keys(header):
    rule = RequestTagMapping(key="entrypoint", tag_prefix="usecase:")
    assert resolve_baggage_tag_mappings(header, [rule]) == ()


def test_exact_rules_work_without_a_variable_rule():
    rule = RequestTagMapping(key="entrypoint", value="a", tag="usecase:manual")
    assert resolve_baggage_tag_mappings("entrypoint=a", [rule]) == (("usecase:manual", False),)
    assert resolve_baggage_tag_mappings("entrypoint=x", [rule]) == ()


@pytest.mark.parametrize(
    "header", ["", "region=sg", "entrypoint=", "entrypoint=a,entrypoint=b", "entrypoint=b,entry%70oint=a", "entrypoint=x"]
)
def test_required_mapping_rejects_missing_ambiguous_or_unmapped_values(header):
    rule = RequestTagMapping(key="entrypoint", value="a", tag="usecase:a", required=True)
    with pytest.raises(ProxyException) as error:
        resolve_baggage_tag_mappings(header, [rule])
    assert error.value.code == "400"



@pytest.mark.parametrize(
    "rule",
    [
        {"key": "entrypoint", "tag": "usecase:a"},
        {"key": "entrypoint", "value": "a", "tag_prefix": "usecase:"},
        {"key": "entrypoint", "value": "a", "tag": "usecase:a", "tag_prefix": "usecase:"},
        {"key": "", "tag_prefix": "usecase:"},
        {"key": "entry point", "tag_prefix": "usecase:"},
    ],
)
def test_invalid_rule_is_rejected(rule):
    with pytest.raises(ValidationError):
        RequestTagMapping.model_validate(rule)
