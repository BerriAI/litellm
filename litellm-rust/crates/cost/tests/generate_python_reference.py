import subprocess
from dataclasses import dataclass
from pathlib import Path

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import Usage


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    prompt: int
    completion: int
    cache_read: int
    cache_write: int
    input_rate: float
    output_rate: float
    cache_read_rate: float | None = None
    cache_write_rate: float | None = None
    threshold: int | None = None
    threshold_input_rate: float | None = None
    threshold_output_rate: float | None = None


CASES = (
    Case("ordinary", 100, 20, 0, 0, 2.0, 4.0),
    Case("cache_fallback", 100, 20, 25, 10, 2.0, 4.0),
    Case("cache_specific", 100, 20, 25, 10, 2.0, 4.0, 0.5, 3.0),
    Case("free_cache", 100, 20, 25, 10, 2.0, 4.0, 0.0, 0.0),
    Case("threshold_below", 99, 20, 0, 0, 2.0, 4.0, threshold=100, threshold_input_rate=5.0, threshold_output_rate=8.0),
    Case("threshold_at", 100, 20, 0, 0, 2.0, 4.0, threshold=100, threshold_input_rate=5.0, threshold_output_rate=8.0),
    Case(
        "threshold_above", 101, 20, 0, 0, 2.0, 4.0, threshold=100, threshold_input_rate=5.0, threshold_output_rate=8.0
    ),
    Case(
        "cache_threshold_above",
        101,
        20,
        25,
        10,
        2.0,
        4.0,
        threshold=100,
        threshold_input_rate=5.0,
        threshold_output_rate=8.0,
    ),
)


def reference(case: Case) -> tuple[float, float]:
    info = {"input_cost_per_token": case.input_rate, "output_cost_per_token": case.output_rate}
    if case.cache_read_rate is not None:
        info["cache_read_input_token_cost"] = case.cache_read_rate
    if case.cache_write_rate is not None:
        info["cache_creation_input_token_cost"] = case.cache_write_rate
    if case.threshold is not None:
        info[f"input_cost_per_token_above_{case.threshold}_tokens"] = case.threshold_input_rate
        info[f"output_cost_per_token_above_{case.threshold}_tokens"] = case.threshold_output_rate
    details = {"cached_tokens": case.cache_read, "cache_write_tokens": case.cache_write}
    usage = Usage(prompt_tokens=case.prompt, completion_tokens=case.completion, prompt_tokens_details=details)
    return generic_cost_per_token(
        model="synthetic",
        usage=usage,
        custom_llm_provider="openai",
        model_info=info,
    )


def main() -> None:
    revision = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    rows = ("# Python reference commit: " + revision,) + tuple(
        "\t".join(
            str(value) if value is not None else ""
            for value in (
                case.name,
                case.prompt,
                case.completion,
                case.cache_read,
                case.cache_write,
                case.input_rate,
                case.output_rate,
                case.cache_read_rate,
                case.cache_write_rate,
                case.threshold,
                case.threshold_input_rate,
                case.threshold_output_rate,
                *reference(case),
            )
        )
        for case in CASES
    )
    Path(__file__).with_name("python_reference.tsv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
