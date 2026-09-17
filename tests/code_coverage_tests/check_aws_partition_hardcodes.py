#!/usr/bin/env python3
"""Ban hardcoded commercial-partition AWS hosts and ARN prefixes under `litellm/`.

An endpoint or ARN built with a literal `amazonaws.com` or `arn:aws:` works in every
commercial region and breaks only for GovCloud (`us-gov-*`, `arn:aws-us-gov:`) and
China (`amazonaws.com.cn`, `arn:aws-cn:`) deployments, so the failure never shows up
in CI or on a developer laptop. `litellm/litellm_core_utils/aws_partition.py` derives
both from the region and is the only place those literals belong. Build hosts with
`get_aws_dns_suffix(region)` and ARNs with `get_aws_arn_prefix(region)`.

Every string constant in every `litellm/**/*.py` file is scanned, including the
literal parts of f-strings and the strings inside `.format()` calls and
concatenations. Docstrings and comments are not, since they never reach a request.
`amazonaws.com.cn` passes because it is already the China partition.

`ALLOWED` holds the (file, token, count) triples that are text rather than a request
target: a hosted logo, an IAM service principal, and hostnames quoted as examples
inside error messages and field descriptions. An entry only covers that many
occurrences of that exact token in that exact file, so a second copy of an allowed
literal is still caught, and an entry whose token is gone or whose count has changed
fails the check so the set only shrinks.
"""

from __future__ import annotations

import ast
import re
import sys
from collections import Counter
from pathlib import Path
from types import MappingProxyType
from typing import Final, NamedTuple

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
SCAN_ROOT: Final = REPO_ROOT / "litellm"
PARTITION_HELPER: Final = "litellm/litellm_core_utils/aws_partition.py"

COMMERCIAL_TOKEN: Final = re.compile(r"[A-Za-z0-9.-]*amazonaws\.com(?!\.cn)|arn:aws:[A-Za-z0-9:/_.*-]*")


class Allowance(NamedTuple):
    file: str
    token: str
    occurrences: int


ALLOWED: Final = frozenset(
    {
        Allowance("litellm/integrations/email_alerting.py", "litellm-listing.s3.amazonaws.com", 1),
        Allowance("litellm/types/integrations/slack_alerting.py", "litellm-listing.s3.amazonaws.com", 1),
        Allowance("litellm/rag/ingestion/bedrock_ingestion.py", "bedrock.amazonaws.com", 1),
        Allowance(
            "litellm/llms/bedrock/chat/agentcore/transformation.py",
            "arn:aws:bedrock-agentcore:region:account:runtime/runtime_id",
            1,
        ),
        Allowance("litellm/llms/bedrock/search/transformation.py", ".amazonaws.com", 1),
        Allowance(
            "litellm/proxy/anthropic_endpoints/claude_code_endpoints/claude_code_marketplace.py",
            "bucket.s3.amazonaws.com",
            1,
        ),
        Allowance("litellm/types/proxy/claude_code_endpoints.py", "bucket.s3.amazonaws.com", 1),
    }
)
ALLOWED_COUNTS: Final = MappingProxyType({(entry.file, entry.token): entry.occurrences for entry in ALLOWED})


class Hit(NamedTuple):
    file: str
    line: int
    token: str


def _docstring_ids(tree: ast.Module) -> frozenset[int]:
    return frozenset(
        id(statement.value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        for statement in node.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _hits_in_file(path: Path) -> tuple[Hit, ...]:
    tree: Final = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstrings: Final = _docstring_ids(tree)
    relative: Final = path.relative_to(REPO_ROOT).as_posix()
    return tuple(
        Hit(relative, node.lineno, match.group(0))
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
        for match in COMMERCIAL_TOKEN.finditer(node.value)
    )


def find_hits(scan_root: Path) -> tuple[Hit, ...]:
    return tuple(
        hit
        for path in sorted(scan_root.rglob("*.py"))
        if path.relative_to(REPO_ROOT).as_posix() != PARTITION_HELPER
        for hit in _hits_in_file(path)
    )


def _violation_message(hit: Hit, found: int) -> str:
    allowed: Final = ALLOWED_COUNTS.get((hit.file, hit.token))
    if allowed is None:
        return f"{hit.file}:{hit.line}: hardcoded commercial AWS partition literal {hit.token!r}"
    return (
        f"{hit.file}:{hit.line}: {hit.token!r} appears {found} times but ALLOWED covers {allowed}; "
        "build it from the region helper or update the count"
    )


def main() -> int:
    hits: Final = find_hits(SCAN_ROOT)
    counts: Final = MappingProxyType(Counter((hit.file, hit.token) for hit in hits))
    violations: Final = tuple(
        sorted(hit for hit in hits if Allowance(hit.file, hit.token, counts[hit.file, hit.token]) not in ALLOWED)
    )
    stale: Final = tuple(entry for entry in sorted(ALLOWED) if (entry.file, entry.token) not in counts)
    for hit in violations:
        print(_violation_message(hit, counts[hit.file, hit.token]))
    for allowance in stale:
        print(f"{allowance.file}: ALLOWED entry {allowance.token!r} no longer matches anything, remove it")
    if violations or stale:
        print(
            "\nBuild AWS hosts with get_aws_dns_suffix(region) and ARNs with get_aws_arn_prefix(region) "
            "from litellm/litellm_core_utils/aws_partition.py so GovCloud and China regions resolve."
        )
        return 1
    print(f"No hardcoded commercial AWS partition literals outside {PARTITION_HELPER}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
