#!/usr/bin/env python3
"""
Vector-based semantic duplicate detection for GitHub Issues and PRs.

Uses embedding models (OpenAI or Voyage via LiteLLM) to build a vector index
of all historical closed/merged PRs and issues, then queries it to find
semantic duplicates of new items.

Usage:
    # Build/update the index
    python vector_duplicate_detection.py index --repo owner/repo

    # Query for duplicates of a specific issue or PR
    python vector_duplicate_detection.py query --repo owner/repo --type issue --number 123

    # Batch scan: check all open issues/PRs against the index
    python vector_duplicate_detection.py scan --repo owner/repo

Environment variables:
    LITELLM_EMBEDDING_MODEL: Model to use (default: text-embedding-3-small)
        Examples: text-embedding-3-small, voyage-3, voyage-code-3
    OPENAI_API_KEY: Required for OpenAI models
    VOYAGE_API_KEY: Required for Voyage models
    GITHUB_TOKEN: Required for GitHub API access
    VECTOR_INDEX_PATH: Path to store the index (default: .github/vector_index.json)
    SIMILARITY_THRESHOLD: Minimum cosine similarity to flag (default: 0.82)
    TOP_K: Number of similar items to return (default: 5)
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = os.getenv("LITELLM_EMBEDDING_MODEL", "text-embedding-3-small")
INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", ".github/vector_index.json")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.82"))
TOP_K = int(os.getenv("TOP_K", "5"))
BATCH_SIZE = 50  # Items per embedding API call
GH_PAGE_SIZE = 100  # Items per GitHub API page


# ---------------------------------------------------------------------------
# Embedding via LiteLLM (supports OpenAI, Voyage, Cohere, etc.)
# ---------------------------------------------------------------------------


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts using litellm."""
    import litellm

    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = litellm.embedding(model=EMBEDDING_MODEL, input=batch)
        batch_embeddings = [item["embedding"] for item in response.data]
        all_embeddings.extend(batch_embeddings)
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.5)  # Rate limit courtesy
    return all_embeddings


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# GitHub data fetching via gh CLI
# ---------------------------------------------------------------------------


def _run_gh(args: List[str]) -> str:
    """Run a gh CLI command and return stdout."""
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def fetch_closed_issues(repo: str) -> List[Dict[str, Any]]:
    """Fetch all closed issues from the repo."""
    print(f"Fetching closed issues from {repo}...")
    raw = _run_gh(
        [
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "closed",
            "--limit",
            "5000",
            "--json",
            "number,title,body,labels,closedAt,stateReason",
        ]
    )
    issues = json.loads(raw) if raw else []
    print(f"  Found {len(issues)} closed issues")
    return issues


def fetch_merged_prs(repo: str) -> List[Dict[str, Any]]:
    """Fetch all merged PRs from the repo."""
    print(f"Fetching merged PRs from {repo}...")
    raw = _run_gh(
        [
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "merged",
            "--limit",
            "5000",
            "--json",
            "number,title,body,labels,mergedAt",
        ]
    )
    prs = json.loads(raw) if raw else []
    print(f"  Found {len(prs)} merged PRs")
    return prs


def fetch_closed_prs(repo: str) -> List[Dict[str, Any]]:
    """Fetch all closed (not merged) PRs from the repo."""
    print(f"Fetching closed PRs from {repo}...")
    raw = _run_gh(
        [
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "closed",
            "--limit",
            "5000",
            "--json",
            "number,title,body,labels,closedAt",
        ]
    )
    prs = json.loads(raw) if raw else []
    print(f"  Found {len(prs)} closed PRs")
    return prs


def fetch_open_issues(repo: str) -> List[Dict[str, Any]]:
    """Fetch all open issues."""
    raw = _run_gh(
        [
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            "5000",
            "--json",
            "number,title,body,labels",
        ]
    )
    return json.loads(raw) if raw else []


def fetch_open_prs(repo: str) -> List[Dict[str, Any]]:
    """Fetch all open PRs."""
    raw = _run_gh(
        [
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            "5000",
            "--json",
            "number,title,body,labels",
        ]
    )
    return json.loads(raw) if raw else []


def fetch_single_item(
    repo: str, item_type: str, number: int
) -> Dict[str, Any]:
    """Fetch a single issue or PR."""
    cmd = "issue" if item_type == "issue" else "pr"
    raw = _run_gh(
        [
            cmd,
            "view",
            str(number),
            "--repo",
            repo,
            "--json",
            "number,title,body,labels",
        ]
    )
    return json.loads(raw)


def post_comment(repo: str, item_type: str, number: int, body: str):
    """Post a comment on an issue or PR."""
    cmd = "issue" if item_type == "issue" else "pr"
    _run_gh([cmd, "comment", str(number), "--repo", repo, "--body", body])


# ---------------------------------------------------------------------------
# Vector index management
# ---------------------------------------------------------------------------


def _item_to_text(item: Dict[str, Any]) -> str:
    """Convert an issue/PR to embeddable text."""
    title = item.get("title", "")
    body = item.get("body", "") or ""
    labels = " ".join(
        label.get("name", "") for label in item.get("labels", [])
    )
    # Truncate body to ~2000 chars to stay within embedding token limits
    if len(body) > 2000:
        body = body[:2000]
    return f"{title}\n{labels}\n{body}".strip()


class VectorIndex:
    """Simple vector index backed by a JSON file."""

    def __init__(self, path: str = INDEX_PATH):
        self.path = path
        self.items: List[Dict[str, Any]] = []  # metadata
        self.vectors: List[List[float]] = []  # embeddings
        self.model: str = EMBEDDING_MODEL
        self._indexed_keys: set = set()

    def load(self) -> bool:
        """Load index from disk. Returns True if loaded successfully."""
        if not os.path.exists(self.path):
            return False
        with open(self.path, "r") as f:
            data = json.load(f)

        stored_model = data.get("model", "")
        if stored_model != EMBEDDING_MODEL:
            print(
                f"Warning: Index was built with {stored_model}, "
                f"current model is {EMBEDDING_MODEL}. Rebuilding required."
            )
            return False

        self.items = data.get("items", [])
        self.vectors = data.get("vectors", [])
        self.model = stored_model
        self._indexed_keys = {
            f"{item['type']}:{item['number']}" for item in self.items
        }
        print(f"Loaded index with {len(self.items)} items")
        return True

    def save(self):
        """Save index to disk."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        data = {
            "model": self.model,
            "item_count": len(self.items),
            "items": self.items,
            "vectors": self.vectors,
        }
        with open(self.path, "w") as f:
            json.dump(data, f)
        print(f"Saved index with {len(self.items)} items to {self.path}")

    def add_items(
        self, items: List[Dict[str, Any]], item_type: str
    ) -> int:
        """Add items to the index, skipping already-indexed ones. Returns count added."""
        new_items = []
        for item in items:
            key = f"{item_type}:{item['number']}"
            if key not in self._indexed_keys:
                new_items.append(item)

        if not new_items:
            print(f"  No new {item_type}s to index")
            return 0

        print(f"  Embedding {len(new_items)} new {item_type}s...")
        texts = [_item_to_text(item) for item in new_items]
        embeddings = get_embeddings(texts)

        for item, embedding in zip(new_items, embeddings):
            meta = {
                "type": item_type,
                "number": item["number"],
                "title": item.get("title", ""),
                "labels": [
                    label.get("name", "")
                    for label in item.get("labels", [])
                ],
            }
            self.items.append(meta)
            self.vectors.append(embedding)
            self._indexed_keys.add(f"{item_type}:{item['number']}")

        return len(new_items)

    def query(
        self,
        text: str,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
        exclude_type: Optional[str] = None,
        exclude_number: Optional[int] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Find the most similar items to the given text."""
        if not self.vectors:
            return []

        query_embedding = get_embeddings([text])[0]

        scores: List[Tuple[int, float]] = []
        for i, vec in enumerate(self.vectors):
            # Skip self-matches
            if (
                exclude_number is not None
                and self.items[i]["number"] == exclude_number
                and (
                    exclude_type is None
                    or self.items[i]["type"] == exclude_type
                )
            ):
                continue
            score = cosine_similarity(query_embedding, vec)
            if score >= threshold:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            results.append((self.items[idx], round(score, 4)))
        return results


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_index(repo: str):
    """Build or update the vector index from closed/merged items."""
    index = VectorIndex()
    index.load()  # Load existing if available (incremental update)

    # Fetch historical data
    closed_issues = fetch_closed_issues(repo)
    merged_prs = fetch_merged_prs(repo)
    closed_prs = fetch_closed_prs(repo)

    # Add to index
    added = 0
    added += index.add_items(closed_issues, "issue")
    added += index.add_items(merged_prs, "pr")
    added += index.add_items(closed_prs, "pr")

    if added > 0:
        index.save()
        print(f"\nIndex updated: {added} new items added, {len(index.items)} total")
    else:
        print("\nIndex is already up to date")


def cmd_query(
    repo: str,
    item_type: str,
    number: int,
    comment: bool = False,
):
    """Query the index for duplicates of a specific issue or PR."""
    index = VectorIndex()
    if not index.load():
        print("Error: No index found. Run 'index' command first.", file=sys.stderr)
        sys.exit(1)

    # Fetch the item
    item = fetch_single_item(repo, item_type, number)
    text = _item_to_text(item)

    print(f"\nSearching for duplicates of {item_type} #{number}: {item['title']}")
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  Threshold: {SIMILARITY_THRESHOLD}")
    print(f"  Index size: {len(index.items)} items")
    print()

    results = index.query(
        text,
        exclude_type=item_type,
        exclude_number=number,
    )

    if not results:
        print("No duplicates found above threshold.")
        return

    print(f"Found {len(results)} potential duplicate(s):\n")
    lines = []
    for meta, score in results:
        prefix = "Issue" if meta["type"] == "issue" else "PR"
        line = f"- #{meta['number']} ({prefix}): {meta['title']} — **{score:.1%}** similarity"
        lines.append(line)
        print(f"  [{score:.1%}] {prefix} #{meta['number']}: {meta['title']}")

    if comment and lines:
        comment_body = (
            "_This comment was generated by semantic duplicate detection and may be inaccurate._\n\n"
            f"This {item_type} appears similar to existing items. Please check:\n"
            + "\n".join(lines)
        )
        post_comment(repo, item_type, number, comment_body)
        print(f"\nComment posted on {item_type} #{number}")


def cmd_scan(repo: str):
    """Batch scan all open issues and PRs against the index."""
    index = VectorIndex()
    if not index.load():
        print("Error: No index found. Run 'index' command first.", file=sys.stderr)
        sys.exit(1)

    open_issues = fetch_open_issues(repo)
    open_prs = fetch_open_prs(repo)

    print(f"\nScanning {len(open_issues)} open issues and {len(open_prs)} open PRs...")
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  Threshold: {SIMILARITY_THRESHOLD}")
    print(f"  Index size: {len(index.items)} items")
    print()

    flagged = []

    for item in open_issues:
        text = _item_to_text(item)
        results = index.query(text, exclude_type="issue", exclude_number=item["number"])
        if results:
            flagged.append(("issue", item, results))

    for item in open_prs:
        text = _item_to_text(item)
        results = index.query(text, exclude_type="pr", exclude_number=item["number"])
        if results:
            flagged.append(("pr", item, results))

    if not flagged:
        print("No duplicates found.")
        return

    print(f"\n{'='*70}")
    print(f"DUPLICATE DETECTION REPORT")
    print(f"{'='*70}\n")

    for item_type, item, results in flagged:
        prefix = "Issue" if item_type == "issue" else "PR"
        print(f"{prefix} #{item['number']}: {item.get('title', '')}")
        for meta, score in results:
            dup_prefix = "Issue" if meta["type"] == "issue" else "PR"
            print(f"  -> [{score:.1%}] {dup_prefix} #{meta['number']}: {meta['title']}")
        print()

    print(f"Total: {len(flagged)} items with potential duplicates")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Vector-based semantic duplicate detection for GitHub"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index
    idx_parser = subparsers.add_parser("index", help="Build/update the vector index")
    idx_parser.add_argument("--repo", required=True, help="GitHub repo (owner/repo)")

    # query
    q_parser = subparsers.add_parser("query", help="Check a single item for duplicates")
    q_parser.add_argument("--repo", required=True, help="GitHub repo (owner/repo)")
    q_parser.add_argument(
        "--type", required=True, choices=["issue", "pr"], dest="item_type"
    )
    q_parser.add_argument("--number", required=True, type=int)
    q_parser.add_argument(
        "--comment", action="store_true", help="Post results as a comment"
    )

    # scan
    s_parser = subparsers.add_parser("scan", help="Batch scan all open items")
    s_parser.add_argument("--repo", required=True, help="GitHub repo (owner/repo)")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args.repo)
    elif args.command == "query":
        cmd_query(args.repo, args.item_type, args.number, args.comment)
    elif args.command == "scan":
        cmd_scan(args.repo)


if __name__ == "__main__":
    main()
