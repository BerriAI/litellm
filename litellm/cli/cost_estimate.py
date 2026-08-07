"""``litellm cost-estimate`` — pre-deployment cost estimation for a single prompt.

Computes the dollar cost of a given model + token count (or model + text)
using the local ``model_prices_and_context_window.json``. Reuses the
public ``litellm.cost_per_token`` and ``litellm.token_counter`` helpers,
so the calculation matches what a real ``litellm.completion`` call
would produce. No network calls are made.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from typing import Any, Optional

import click
from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class CostEstimate:
    """Result of a cost-estimate run, suitable for both table and JSON output."""

    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def _load_local_cost_map() -> dict[str, Any]:
    """Read the local cost map. Raises if the file is missing or malformed."""
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    return GetModelCostMap.load_local_model_cost_map()


def _resolve_input_tokens(
    model: str,
    *,
    input_tokens: Optional[int],
    input_text: Optional[str],
    messages_json: Optional[str],
) -> int:
    """Pick the right input-source option and return the input token count.

    Exactly one of ``input_tokens``, ``input_text``, or ``messages_json``
    must be provided. Returns the input token count. Raises ``click.UsageError``
    on invalid combinations.
    """
    provided = [
        name
        for name, val in (("input-tokens", input_tokens), ("input-text", input_text), ("messages", messages_json))
        if val is not None
    ]
    if len(provided) == 0:
        raise click.UsageError("Provide one of --input-tokens, --input-text, or --messages (as a JSON list).")
    if len(provided) > 1:
        raise click.UsageError(f"--{provided[0]} and --{provided[1]} are mutually exclusive; pass only one.")
    if input_tokens is not None:
        if input_tokens < 0:
            raise click.UsageError("--input-tokens must be >= 0")
        return int(input_tokens)
    if input_text is not None:
        from litellm import token_counter

        return int(
            token_counter(
                model=model,
                text=input_text,
            )
        )
    try:
        messages = json.loads(messages_json)  # type: ignore[arg-type]
    except json.JSONDecodeError as exc:
        raise click.UsageError(f"--messages must be valid JSON: {exc}") from exc
    if not isinstance(messages, list):
        raise click.UsageError("--messages must be a JSON list of message objects.")
    from litellm import token_counter

    return int(token_counter(model=model, messages=messages))


def _compute_costs(model: str, input_tokens: int, output_tokens: int) -> tuple[float, float]:
    """Return (input_cost, output_cost) in USD for the given token counts.

    Raises ``click.ClickException`` if the model is not in the local
    cost map.
    """
    cost_map = _load_local_cost_map()
    if model not in cost_map:
        raise click.ClickException(
            f"model {model!r} is not in the local cost map; register it with custom pricing or update the cost map."
        )
    try:
        from litellm import cost_per_token

        input_cost, output_cost = cost_per_token(
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
    except Exception as exc:
        raise click.ClickException(f"cost_per_token raised {type(exc).__name__}: {exc}") from exc
    return float(input_cost), float(output_cost)


def estimate_cost(
    model: str,
    *,
    input_tokens: Optional[int] = None,
    input_text: Optional[str] = None,
    messages_json: Optional[str] = None,
    output_tokens: int = 0,
) -> CostEstimate:
    """Public helper for programmatic use. Returns a ``CostEstimate``.

    Raises :class:`ValueError` for invalid input (no input source, two
    input sources, negative token counts, malformed messages JSON).
    Raises :class:`click.ClickException` only when the model is not
    in the local cost map, so callers can catch the missing-model
    case separately from input-validation errors.
    """
    if output_tokens < 0:
        raise ValueError("output_tokens must be >= 0")
    try:
        resolved_input = _resolve_input_tokens(
            model,
            input_tokens=input_tokens,
            input_text=input_text,
            messages_json=messages_json,
        )
    except click.UsageError as exc:
        raise ValueError(str(exc)) from exc
    input_cost, output_cost = _compute_costs(model, resolved_input, output_tokens)
    return CostEstimate(
        model=model,
        input_tokens=resolved_input,
        output_tokens=output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
    )


def _render_table(estimate: CostEstimate, console: Console) -> None:
    table = Table(title=f"cost-estimate: {estimate.model}", show_lines=False)
    table.add_column("field", style="cyan", no_wrap=True)
    table.add_column("value")
    table.add_row("model", estimate.model)
    table.add_row("input_tokens", f"{estimate.input_tokens}")
    table.add_row("output_tokens", f"{estimate.output_tokens}")
    table.add_row("input_cost", f"${estimate.input_cost:.6f}")
    table.add_row("output_cost", f"${estimate.output_cost:.6f}")
    table.add_row("total_cost", f"[bold]${estimate.total_cost:.6f}[/bold]")
    console.print(table)


@click.command()
@click.option(
    "--model",
    "model",
    required=True,
    help="Model name as it appears in the local cost map (e.g. gpt-4o, claude-sonnet-4-6).",
)
@click.option(
    "--input-tokens",
    "input_tokens",
    type=int,
    default=None,
    help="Pre-counted input token count. Mutually exclusive with --input-text and --messages.",
)
@click.option(
    "--input-text",
    "input_text",
    default=None,
    help="Input text to count tokens for. Mutually exclusive with --input-tokens and --messages.",
)
@click.option(
    "--messages",
    "messages_json",
    default=None,
    help="JSON list of OpenAI-style message objects to count input tokens for. Mutually exclusive with --input-tokens and --input-text.",
)
@click.option(
    "--output-tokens",
    "output_tokens",
    type=int,
    default=0,
    show_default=True,
    help="Expected output token count (default 0).",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Emit a JSON object instead of a table.",
)
def cli(
    model: str,
    input_tokens: Optional[int],
    input_text: Optional[str],
    messages_json: Optional[str],
    output_tokens: int,
    output_json: bool,
) -> None:
    """Estimate the cost of a single litellm completion call."""
    try:
        estimate = estimate_cost(
            model,
            input_tokens=input_tokens,
            input_text=input_text,
            messages_json=messages_json,
            output_tokens=output_tokens,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    except click.UsageError:
        raise
    except click.ClickException:
        raise
    except Exception as exc:
        click.echo(f"cost-estimate failed: {type(exc).__name__}: {exc}", err=True)
        sys.exit(1)

    if output_json:
        click.echo(json.dumps(estimate.to_jsonable()))
    else:
        _render_table(estimate, Console())
    sys.exit(0)


if __name__ == "__main__":
    cli()
