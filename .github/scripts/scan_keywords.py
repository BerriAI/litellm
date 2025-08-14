#!/usr/bin/env python3
import json
import os
import sys
import urllib.request
import urllib.error


def read_event_payload() -> dict:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path or not os.path.exists(event_path):
        return {}
    with open(event_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_issue_text(event: dict) -> tuple[str, str, int, str, str]:
    issue = event.get("issue") or {}
    title = (issue.get("title") or "").strip()
    body = (issue.get("body") or "").strip()
    number = issue.get("number") or 0
    html_url = issue.get("html_url") or ""
    author = ((issue.get("user") or {}).get("login") or "").strip()
    return title, body, number, html_url, author


def detect_keywords(text: str, keywords: list[str]) -> list[str]:
    lowered = text.lower()
    matches = []
    for keyword in keywords:
        k = keyword.strip().lower()
        if not k:
            continue
        if k in lowered:
            matches.append(keyword.strip())
    # Deduplicate while preserving order
    seen = set()
    unique_matches = []
    for m in matches:
        if m not in seen:
            unique_matches.append(m)
            seen.add(m)
    return unique_matches


def send_webhook(webhook_url: str, payload: dict) -> None:
    if not webhook_url:
        return
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        print(f"Webhook HTTP error: {e.code} {e.reason}", file=sys.stderr)
    except urllib.error.URLError as e:
        print(f"Webhook URL error: {e.reason}", file=sys.stderr)
    except Exception as e:
        print(f"Webhook unexpected error: {e}", file=sys.stderr)


def _excerpt(text: str, max_len: int = 400) -> str:
    if not text:
        return ""
    
    # Keep original formatting
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"



def main() -> int:
    event = read_event_payload()
    if not event:
        print("::warning::No event payload found; exiting without labeling.")
        return 0

    # Read issue details
    title, body, number, html_url, author = get_issue_text(event)
    combined_text = f"{title}\n\n{body}".strip()

    # Keywords from env or defaults
    keywords_env = os.environ.get("KEYWORDS", "")
    default_keywords = ["azure", "openai", "bedrock", "vertexai", "vertex ai", "anthropic"]
    keywords = [k.strip() for k in keywords_env.split(",")] if keywords_env else default_keywords

    matches = detect_keywords(combined_text, keywords)
    found = bool(matches)

    # Emit outputs
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as fh:
            fh.write(f"found={'true' if found else 'false'}\n")
            fh.write(f"matches={','.join(matches)}\n")

    # Optional webhook notification
    webhook_url = os.environ.get("PROVIDER_ISSUE_WEBHOOK_URL", "").strip()
    if found and webhook_url:
        repo_full = (event.get("repository") or {}).get("full_name", "")
        title_part = f"*{title}*" if title else "New issue"
        author_part = f" by @{author}" if author else ""
        body_preview = _excerpt(body)
        preview_block = f"\n{body_preview}" if body_preview else ""
        payload = {
            "text": (
                f"New issue 🚨\n"
                f"{title_part}\n\n{preview_block}\n"
                f"<{html_url}|View issue>\n"
                f"Author: {author}"
            )
        }
        send_webhook(webhook_url, payload)

    # Print a short log line for Actions UI
    if found:
        print(f"Detected provider keywords: {', '.join(matches)}")
    else:
        print("No provider keywords detected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


