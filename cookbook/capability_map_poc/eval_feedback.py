import argparse
import json
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import openai
from datasets import load_dataset  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]  # no stubs
from pydantic import TypeAdapter

from .corrections import correction_stats, turn_key
from .evaluate import FeedbackEval, Mismatch, evaluate, mismatches
from .labeler import PROMPT_VERSION, GatewayCompleter, label_conversations
from .schemas import CorrectionStats, FeedbackLabel, LabelFailure, LabelOutcome
from .sources import EmnlpParse, EmnlpSplit, parse_emnlp

DATASET: Final = "yuhan-nlp/multiturn-feedback"
OUTCOME_ADAPTER: Final = TypeAdapter[FeedbackLabel | LabelFailure](FeedbackLabel | LabelFailure)  # pyright: ignore[reportArgumentType]  # pydantic accepts a runtime union
PAPER_DENSE_GPT4O_MINI: Final = "binary 41.6%, three-way 55.4%, fine-grained 49.0% (Table 2)"


def load_cache(path: Path) -> dict[tuple[str, int], LabelOutcome]:
    if not path.exists():
        return {}
    outcomes: Final = (OUTCOME_ADAPTER.validate_json(line) for line in path.read_text().splitlines() if line)
    return {turn_key(o.session_id, o.turn_idx): o for o in outcomes}


def _load_rows(split: EmnlpSplit) -> tuple[object, ...]:
    return tuple(load_dataset(DATASET, split=split))  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]  # datasets ships no stubs, parse_emnlp validates rows


def run_split(split: EmnlpSplit, completer: GatewayCompleter, out_dir: Path, workers: int, limit: int | None) -> str:
    parse: Final = parse_emnlp(_load_rows(split), split)
    conversations: Final = parse.conversations[:limit] if limit else parse.conversations
    sessions: Final = {c.session_id for c in conversations}
    examples: Final = tuple(e for e in parse.examples if e.turn.session_id in sessions)
    cache_path: Final = out_dir / f"labels_{split}.jsonl"
    lock: Final = threading.Lock()

    def append(outcomes: tuple[LabelOutcome, ...]) -> None:
        with lock, cache_path.open("a") as handle:
            handle.writelines(o.model_dump_json() + "\n" for o in outcomes)

    outcomes: Final = label_conversations(
        conversations, completer, load_cache(cache_path), workers=workers, on_labels=append
    )
    wanted: Final = {turn_key(e.turn.session_id, e.turn.turn_idx) for e in examples}
    current: Final = tuple(o for o in outcomes if turn_key(o.session_id, o.turn_idx) in wanted)
    gold: Final = tuple(e.gold for e in examples)
    report: Final = evaluate(gold, current)
    misses: Final = mismatches(gold, current)
    stats: Final = correction_stats((e.turn for e in examples), current)
    cost: Final = sum(o.cost_usd or 0.0 for o in outcomes if o.session_id in sessions)
    failures: Final = [o.model_dump() for o in current if not isinstance(o, FeedbackLabel)]
    (out_dir / f"metrics_{split}.json").write_text(report.model_dump_json(indent=2))
    (out_dir / f"correction_stats_{split}.json").write_text(json.dumps([s.model_dump() for s in stats], indent=2))
    (out_dir / f"failures_{split}.json").write_text(json.dumps(failures, indent=2))
    return _render(split, parse, report, misses, stats, cost, len(sessions))


def _render(
    split: EmnlpSplit,
    parse: EmnlpParse,
    report: FeedbackEval,
    misses: Sequence[Mismatch],
    stats: Sequence[CorrectionStats],
    cost: float,
    conversation_count: int,
) -> str:
    labels: Final = [label.value for label in report.confusion_labels]
    confusion: Final = "\n".join(
        f"| {gold} | " + " | ".join(str(v) for v in row) + " |" for gold, row in zip(labels, report.confusion)
    )
    per_label: Final = "\n".join(
        f"| {s.label.value} | {s.precision:.2f} | {s.recall:.2f} | {s.f1:.2f} | {s.support} |" for s in report.per_label
    )
    examples: Final = "\n".join(_render_miss(m, parse) for m in misses[:10])
    return f"""## Split `{split}`

Parsing: {len(parse.examples)} labeled turns, {len(parse.skipped_conversations)} conversations skipped
(label count did not match user turns), {parse.unaligned_feedback} gold labels not aligned to a turn, {parse.duplicate_feedback} duplicate gold labels
for an already labeled turn dropped (first kept),
{parse.text_mismatches} aligned labels whose feedback text was not found in the turn

Labeled {report.labeled_turns}/{report.gold_turns} gold turns, {report.failed_turns} failures, \
labeler cost ${cost:.4f} across {conversation_count} conversations

| Metric | Value |
|---|---|
| Correction (NEG_1-3) precision | {report.corrective.precision:.3f} |
| Correction recall | {report.corrective.recall:.3f} |
| Correction F1 | {report.corrective.f1:.3f} |
| Gold corrections | {report.corrective.support} |
| Fine-grained accuracy | {report.fine_grained_accuracy:.3f} |
| Three-way (POS/NEG/NEU) accuracy | {report.three_way_accuracy:.3f} |
| Feedback vs no-feedback accuracy | {report.any_feedback_accuracy:.3f} |

### Per label

| Label | P | R | F1 | Gold count |
|---|---|---|---|---|
{per_label}

### Confusion (rows gold, columns predicted)

| gold \\ pred | {" | ".join(labels)} |
|---|{"---|" * len(labels)}
{confusion}

### Correction stats via the shared interface (grouped by dataset source)

```json
{json.dumps([s.model_dump() for s in stats], indent=2)}
```

### First 10 disagreements (sorted by conversation id)

{examples or "none"}
"""


def _render_miss(miss: Mismatch, parse: EmnlpParse) -> str:
    turn: Final = next(
        e.turn for e in parse.examples if (e.turn.session_id, e.turn.turn_idx) == (miss.session_id, miss.turn_idx)
    )
    return (
        f"- `{miss.session_id}` turn {miss.turn_idx}: gold **{miss.gold.value}**, predicted **{miss.predicted.value}**"
        f"\n  - next user turn: {_clip(turn.next_user_turn or '')}\n"
        f"  - labeler: {_clip(miss.rationale)}"
    )


def _clip(text: str) -> str:
    flat: Final = " ".join(text.split())
    return flat if len(flat) <= 240 else flat[:237] + "..."


def main() -> None:
    parser: Final = argparse.ArgumentParser(description="Stage 4a: check the correction labeler against EMNLP labels")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="claude-sonnet-5-5")
    parser.add_argument("--splits", nargs="+", default=["dense", "sparse"], choices=["dense", "sparse"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="label only the first N conversations per split")
    args: Final = parser.parse_args()
    client: Final = openai.OpenAI(
        base_url=os.environ["LITELLM_PROXY_BASE_URL"], api_key=os.environ["LITELLM_PROXY_KEY"]
    )
    completer: Final = GatewayCompleter(client, args.model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sections: Final = [run_split(split, completer, args.out_dir, args.workers, args.limit) for split in args.splits]
    header: Final = (
        f"# Stage 4a: correction labeler vs EMNLP 2025 human labels\n\nLabeler `{args.model}` via the LiteLLM gateway, "
        f"prompt `{PROMPT_VERSION}`, temperature 0, one call per conversation labeling every user turn after the first\n\n"
        f"Paper reference on dense, GPT-4o-mini with full conversation: {PAPER_DENSE_GPT4O_MINI}. "
        "Human inter-annotator Cohen's kappa 0.70 binary, 0.74 three-way, 0.60 fine-grained\n"
    )
    (args.out_dir / "feedback_eval_report.md").write_text("\n".join([header, *sections]))


if __name__ == "__main__":
    main()
