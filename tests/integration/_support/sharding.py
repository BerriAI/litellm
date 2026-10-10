from __future__ import annotations

import argparse
import sys
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ShardingOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    directory: Path
    shards: int = Field(gt=0)


class ShardExecution(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    inventory: tuple[str, ...]
    collected: tuple[str, ...]
    passed: tuple[str, ...]
    skipped: tuple[str, ...]
    shard_count: int
    shard_index: int
    complete: bool
    exitstatus: int


def shard_errors(executions: tuple[ShardExecution, ...], count: int) -> tuple[str, ...]:
    if not executions:
        return ("No integration shard execution records",)
    inventory: Final = frozenset(executions[0].inventory)
    selected: Final = Counter(chain.from_iterable(execution.collected for execution in executions))
    indices: Final = tuple(execution.shard_index for execution in executions)
    return (
        *(("Missing or duplicate integration shards",) if sorted(indices) != list(range(count)) else ()),
        *(
            ("Integration shard counts disagree",)
            if any(execution.shard_count != count for execution in executions)
            else ()
        ),
        *(
            ("Integration collection inventories disagree",)
            if any(frozenset(execution.inventory) != inventory for execution in executions)
            else ()
        ),
        *(
            ("Integration inventory is empty or contains duplicates",)
            if not inventory or any(len(execution.inventory) != len(inventory) for execution in executions)
            else ()
        ),
        *(
            ("Integration selection omitted, duplicated, or added test cases",)
            if frozenset(selected) != inventory or any(value != 1 for value in selected.values())
            else ()
        ),
        *(
            ("An integration shard failed or did not execute its complete selection",)
            if any(
                not execution.complete
                or execution.exitstatus != 0
                or Counter(execution.collected) != Counter(execution.passed + execution.skipped)
                for execution in executions
            )
            else ()
        ),
    )


def main() -> int:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--shards", required=True, type=int)
    options: Final = ShardingOptions.model_validate(vars(parser.parse_args()))
    paths: Final = tuple(sorted(options.directory.glob("*/execution.json")))
    try:
        executions: Final = tuple(ShardExecution.model_validate_json(path.read_text()) for path in paths)
    except (OSError, ValidationError) as error:
        sys.stderr.write(f"Invalid integration shard execution record: {error}\n")
        return 1
    errors: Final = shard_errors(executions, options.shards)
    if errors:
        sys.stderr.write("\n".join(errors) + "\n")
        return 1
    sys.stdout.write(
        f"Verified {len(executions[0].inventory)} integration cases across {options.shards} isolated shards\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
