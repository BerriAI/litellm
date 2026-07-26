"""Tests for the ``litellm cost-estimate`` CLI subcommand."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from litellm.cli.cost_estimate import (
    CostEstimate,
    cli,
    estimate_cost,
    _compute_costs,
    _resolve_input_tokens,
)


def test_estimate_cost_uses_local_cost_map(monkeypatch):
    """End-to-end: pass --input-tokens and get a non-negative USD total."""
    monkeypatch.setattr(
        "litellm.cost_per_token",
        lambda **kw: (0.0025, 0.0050),
    )
    estimate = estimate_cost(model="gpt-4o", input_tokens=1000, output_tokens=500)
    assert isinstance(estimate, CostEstimate)
    assert estimate.model == "gpt-4o"
    assert estimate.input_tokens == 1000
    assert estimate.output_tokens == 500
    assert estimate.input_cost == pytest.approx(0.0025)
    assert estimate.output_cost == pytest.approx(0.0050)
    assert estimate.total_cost == pytest.approx(0.0075)


def test_estimate_cost_counts_tokens_from_text(monkeypatch):
    """When --input-text is used, token_counter is invoked for the given model."""
    monkeypatch.setattr(
        "litellm.token_counter",
        lambda **kw: 42,
    )
    monkeypatch.setattr(
        "litellm.cost_per_token",
        lambda **kw: (0.0001, 0.0),
    )
    estimate = estimate_cost(
        model="gpt-4o",
        input_text="hello world",
        output_tokens=0,
    )
    assert estimate.input_tokens == 42
    assert estimate.output_tokens == 0
    assert estimate.total_cost == pytest.approx(0.0001)


def test_estimate_cost_counts_tokens_from_messages(monkeypatch):
    """When --messages is used, token_counter receives the parsed JSON list."""
    captured: dict = {}

    def fake_token_counter(**kw):
        captured.update(kw)
        return 17

    monkeypatch.setattr("litellm.token_counter", fake_token_counter)
    monkeypatch.setattr("litellm.cost_per_token", lambda **kw: (0.001, 0.0))
    messages = '[{"role": "user", "content": "hi"}]'
    estimate = estimate_cost(
        model="claude-sonnet-4-6",
        messages_json=messages,
        output_tokens=10,
    )
    assert captured["model"] == "claude-sonnet-4-6"
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    assert estimate.input_tokens == 17
    assert estimate.output_tokens == 10


def test_estimate_cost_rejects_mutually_exclusive_input_options(monkeypatch):
    """Passing two input sources at once is a usage error."""
    monkeypatch.setattr("litellm.cost_per_token", lambda **kw: (0.0, 0.0))
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 0)
    with pytest.raises(ValueError) as exc:
        estimate_cost(
            model="gpt-4o",
            input_tokens=10,
            input_text="also this",
        )
    assert "mutually exclusive" in str(exc.value).lower()


def test_estimate_cost_requires_some_input():
    """No input source at all is a usage error."""
    with pytest.raises(ValueError) as exc:
        estimate_cost(model="gpt-4o")
    assert "input-tokens" in str(exc.value)


def test_estimate_cost_rejects_negative_tokens():
    with pytest.raises(ValueError):
        estimate_cost(model="gpt-4o", input_tokens=-1)
    with pytest.raises(ValueError):
        estimate_cost(model="gpt-4o", input_tokens=0, output_tokens=-1)


def test_estimate_cost_raises_on_unknown_model(monkeypatch):
    monkeypatch.setattr(
        "litellm.cli.cost_estimate._load_local_cost_map",
        lambda: {"gpt-4o": {}},
    )
    monkeypatch.setattr("litellm.cost_per_token", lambda **kw: (0.0, 0.0))
    from click import ClickException

    with pytest.raises(ClickException) as exc:
        estimate_cost(model="totally-unknown-model-9000", input_tokens=10)
    assert "not in the local cost map" in str(exc.value)


def test_estimate_cost_propagates_cost_per_token_failure(monkeypatch):
    monkeypatch.setattr(
        "litellm.cli.cost_estimate._load_local_cost_map",
        lambda: {"gpt-4o": {}},
    )

    def _raise(**kw):
        raise RuntimeError("simulated cost_per_token failure")

    monkeypatch.setattr("litellm.cost_per_token", _raise)
    from click import ClickException

    with pytest.raises(ClickException) as exc:
        estimate_cost(model="gpt-4o", input_tokens=10)
    assert "simulated cost_per_token failure" in str(exc.value)


def test_to_jsonable_round_trip():
    estimate = CostEstimate(
        model="gpt-4o",
        input_tokens=10,
        output_tokens=5,
        input_cost=0.001,
        output_cost=0.002,
        total_cost=0.003,
    )
    assert json.loads(json.dumps(estimate.to_jsonable())) == {
        "model": "gpt-4o",
        "input_tokens": 10,
        "output_tokens": 5,
        "input_cost": 0.001,
        "output_cost": 0.002,
        "total_cost": 0.003,
    }


def test_cli_table_output_for_input_tokens():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--model", "gpt-4o", "--input-tokens", "1000", "--output-tokens", "500"],
    )
    assert result.exit_code == 0, result.output
    assert "cost-estimate: gpt-4o" in result.output
    assert "input_tokens" in result.output
    assert "output_tokens" in result.output
    assert "total_cost" in result.output


def test_cli_json_output_is_valid_json():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--model",
            "gpt-4o",
            "--input-tokens",
            "1000",
            "--output-tokens",
            "500",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["model"] == "gpt-4o"
    assert payload["input_tokens"] == 1000
    assert payload["output_tokens"] == 500
    assert payload["total_cost"] > 0


def test_cli_input_text_counts_tokens_and_estimates(monkeypatch):
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 9)
    monkeypatch.setattr("litellm.cost_per_token", lambda **kw: (0.00009, 0.0))
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--model",
            "gpt-4o",
            "--input-text",
            "hi",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["input_tokens"] == 9
    assert payload["output_tokens"] == 0
    assert payload["total_cost"] == pytest.approx(0.00009)


def test_cli_messages_json_input(monkeypatch):
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 25)
    monkeypatch.setattr("litellm.cost_per_token", lambda **kw: (0.00025, 0.0))
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--model",
            "claude-sonnet-4-6",
            "--messages",
            '[{"role": "user", "content": "hello there"}]',
            "--output-tokens",
            "100",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["input_tokens"] == 25
    assert payload["output_tokens"] == 100


def test_cli_rejects_mutually_exclusive_input_flags():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--model",
            "gpt-4o",
            "--input-tokens",
            "10",
            "--input-text",
            "also this",
        ],
    )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output.lower()


def test_cli_rejects_missing_input():
    runner = CliRunner()
    result = runner.invoke(cli, ["--model", "gpt-4o"])
    assert result.exit_code == 2


def test_cli_rejects_invalid_messages_json():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--model", "gpt-4o", "--messages", "not-json"],
    )
    assert result.exit_code == 2
    assert "valid json" in result.output.lower()


def test_cli_rejects_non_list_messages():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--model", "gpt-4o", "--messages", '{"role": "user", "content": "hi"}'],
    )
    assert result.exit_code == 2
    assert "list" in result.output.lower()


def test_cli_exits_1_on_unknown_model(monkeypatch):
    """The ClickException path surfaces as exit code 1 in the CLI runner."""
    monkeypatch.setattr(
        "litellm.cli.cost_estimate._load_local_cost_map",
        lambda: {"gpt-4o": {}},
    )
    monkeypatch.setattr("litellm.cost_per_token", lambda **kw: (0.0, 0.0))
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--model", "totally-unknown-xyz", "--input-tokens", "10"],
    )
    assert result.exit_code == 1
    assert "not in the local cost map" in result.output


def test_resolve_input_tokens_uses_token_counter_for_text(monkeypatch):
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 7)
    assert _resolve_input_tokens("gpt-4o", input_tokens=None, input_text="hi", messages_json=None) == 7


def test_resolve_input_tokens_uses_token_counter_for_messages(monkeypatch):
    monkeypatch.setattr("litellm.token_counter", lambda **kw: 11)
    assert (
        _resolve_input_tokens(
            "gpt-4o",
            input_tokens=None,
            input_text=None,
            messages_json='[{"role":"user","content":"hi"}]',
        )
        == 11
    )


def test_resolve_input_tokens_returns_raw_int_when_provided():
    assert _resolve_input_tokens("gpt-4o", input_tokens=42, input_text=None, messages_json=None) == 42


def test_compute_costs_returns_tuple_from_cost_per_token(monkeypatch):
    monkeypatch.setattr(
        "litellm.cli.cost_estimate._load_local_cost_map",
        lambda: {"gpt-4o": {}},
    )
    monkeypatch.setattr("litellm.cost_per_token", lambda **kw: (0.01, 0.02))
    assert _compute_costs("gpt-4o", 100, 50) == (0.01, 0.02)
