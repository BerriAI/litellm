"""
Constants and helpers for ChatGPT subscription OAuth.
"""
import os
import platform
from typing import Any, Optional, Union
from uuid import uuid4

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException

# OAuth + API constants (derived from openai/codex)
CHATGPT_AUTH_BASE = "https://auth.openai.com"
CHATGPT_DEVICE_CODE_URL = f"{CHATGPT_AUTH_BASE}/api/accounts/deviceauth/usercode"
CHATGPT_DEVICE_TOKEN_URL = f"{CHATGPT_AUTH_BASE}/api/accounts/deviceauth/token"
CHATGPT_OAUTH_TOKEN_URL = f"{CHATGPT_AUTH_BASE}/oauth/token"
CHATGPT_DEVICE_VERIFY_URL = f"{CHATGPT_AUTH_BASE}/codex/device"
CHATGPT_API_BASE = "https://chatgpt.com/backend-api/codex"
CHATGPT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

DEFAULT_ORIGINATOR = "codex_cli_rs"
DEFAULT_USER_AGENT = "codex_cli_rs/0.0.0 (Unknown 0; unknown) unknown"
CHATGPT_DEFAULT_INSTRUCTIONS = """You are Codex, based on GPT-5. You are running as a coding agent in the Codex CLI on a user's computer.

## General

- When searching for text or files, prefer using `rg` or `rg --files` respectively because `rg` is much faster than alternatives like `grep`. (If the `rg` command is not found, then use alternatives.)

## Editing constraints

- Default to ASCII when editing or creating files. Only introduce non-ASCII or other Unicode characters when there is a clear justification and the file already uses them.
- Add succinct code comments that explain what is going on if code is not self-explanatory. You should not add comments like "Assigns the value to the variable", but a brief comment might be useful ahead of a complex code block that the user would otherwise have to spend time parsing out. Usage of these comments should be rare.
- Try to use apply_patch for single file edits, but it is fine to explore other options to make the edit if it does not work well. Do not use apply_patch for changes that are auto-generated (i.e. generating package.json or running a lint or format command like gofmt) or when scripting is more efficient (such as search and replacing a string across a codebase).
- You may be in a dirty git worktree.
    * NEVER revert existing changes you did not make unless explicitly requested, since these changes were made by the user.
    * If asked to make a commit or code edits and there are unrelated changes to your work or changes that you didn't make in those files, don't revert those changes.
    * If the changes are in files you've touched recently, you should read carefully and understand how you can work with the changes rather than reverting them.
    * If the changes are in unrelated files, just ignore them and don't revert them.
- Do not amend a commit unless explicitly requested to do so.
- While you are working, you might notice unexpected changes that you didn't make. If this happens, STOP IMMEDIATELY and ask the user how they would like to proceed.
- **NEVER** use destructive commands like `git reset --hard` or `git checkout --` unless specifically requested or approved by the user.

## Plan tool

When using the planning tool:
- Skip using the planning tool for straightforward tasks (roughly the easiest 25%).
- Do not make single-step plans.
- When you made a plan, update it after having performed one of the sub-tasks that you shared on the plan.

## Special user requests

- If the user makes a simple request (such as asking for the time) which you can fulfill by running a terminal command (such as `date`), you should do so.
- If the user asks for a "review", default to a code review mindset: prioritise identifying bugs, risks, behavioural regressions, and missing tests. Findings must be the primary focus of the response - keep summaries or overviews brief and only after enumerating the issues. Present findings first (ordered by severity with file/line references), follow with open questions or assumptions, and offer a change-summary only as a secondary detail. If no findings are discovered, state that explicitly and mention any residual risks or testing gaps.

## Frontend tasks
When doing frontend design tasks, avoid collapsing into "AI slop" or safe, average-looking layouts.
Aim for interfaces that feel intentional, bold, and a bit surprising.
- Typography: Use expressive, purposeful fonts and avoid default stacks (Inter, Roboto, Arial, system).
- Color & Look: Choose a clear visual direction; define CSS variables; avoid purple-on-white defaults. No purple bias or dark mode bias.
- Motion: Use a few meaningful animations (page-load, staggered reveals) instead of generic micro-motions.
- Background: Don't rely on flat, single-color backgrounds; use gradients, shapes, or subtle patterns to build atmosphere.
- Overall: Avoid boilerplate layouts and interchangeable UI patterns. Vary themes, type families, and visual languages across outputs.
- Ensure the page loads properly on both desktop and mobile

Exception: If working within an existing website or design system, preserve the established patterns, structure, and visual language.

## Presenting your work and final message

You are producing plain text that will later be styled by the CLI. Follow these rules exactly. Formatting should make results easy to scan, but not feel mechanical. Use judgment to decide how much structure adds value.

- Default: be very concise; friendly coding teammate tone.
- Ask only when needed; suggest ideas; mirror the user's style.
- For substantial work, summarize clearly; follow final-answer formatting.
- Skip heavy formatting for simple confirmations.
- Don't dump large files you've written; reference paths only.
- No "save/copy this file" - User is on the same machine.
- Offer logical next steps (tests, commits, build) briefly; add verify steps if you couldn't do something.
- For code changes:
  * Lead with a quick explanation of the change, and then give more details on the context covering where and why a change was made. Do not start this explanation with "summary", just jump right in.
  * If there are natural next steps the user may want to take, suggest them at the end of your response. Do not make suggestions if there are no natural next steps.
  * When suggesting multiple options, use numeric lists for the suggestions so the user can quickly respond with a single number.
- The user does not command execution outputs. When asked to show the output of a command (e.g. `git show`), relay the important details in your answer or summarize the key lines so the user understands the result.

### Final answer structure and style guidelines

- Plain text; CLI handles styling. Use structure only when it helps scanability.
- Headers: optional; short Title Case (1-3 words) wrapped in **...**; no blank line before the first bullet; add only if they truly help.
- Bullets: use - ; merge related points; keep to one line when possible; 4-6 per list ordered by importance; keep phrasing consistent.
- Monospace: backticks for commands/paths/env vars/code ids and inline examples; use for literal keyword bullets; never combine with **.
- Code samples or multi-line snippets should be wrapped in fenced code blocks; include an info string as often as possible.
- Structure: group related bullets; order sections general -> specific -> supporting; for subsections, start with a bolded keyword bullet, then items; match complexity to the task.
- Tone: collaborative, concise, factual; present tense, active voice; self-contained; no "above/below"; parallel wording.
- Don'ts: no nested bullets/hierarchies; no ANSI codes; don't cram unrelated keywords; keep keyword lists short--wrap/reformat if long; avoid naming formatting styles in answers.
- Adaptation: code explanations -> precise, structured with code refs; simple tasks -> lead with outcome; big changes -> logical walkthrough + rationale + next actions; casual one-offs -> plain sentences, no headers/bullets.
- File References: When referencing files in your response follow the below rules:
  * Use inline code to make file paths clickable.
  * Each reference should have a stand alone path. Even if it's the same file.
  * Accepted: absolute, workspace-relative, a/ or b/ diff prefixes, or bare filename/suffix.
  * Optionally include line/column (1-based): :line[:column] or #Lline[Ccolumn] (column defaults to 1).
  * Do not use URIs like file://, vscode://, or https://.
  * Do not provide range of lines
  * Examples: src/app.ts, src/app.ts:42, b/server/index.js#L10, C:\\repo\\project\\main.rs:12:5
"""


class ChatGPTAuthError(BaseLLMException):
    def __init__(
        self,
        status_code,
        message,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
        headers: Optional[Union[httpx.Headers, dict]] = None,
        body: Optional[dict] = None,
    ):
        super().__init__(
            status_code=status_code,
            message=message,
            request=request,
            response=response,
            headers=headers,
            body=body,
        )


class GetDeviceCodeError(ChatGPTAuthError):
    pass


class GetAccessTokenError(ChatGPTAuthError):
    pass


class RefreshAccessTokenError(ChatGPTAuthError):
    pass


def _safe_header_value(value: str) -> str:
    if not value:
        return ""
    return "".join(ch if 32 <= ord(ch) <= 126 else "_" for ch in value)


def _sanitize_user_agent_token(value: str) -> str:
    if not value:
        return ""
    return "".join(
        ch if (ch.isalnum() or ch in "-_./") else "_" for ch in value
    )


def _terminal_user_agent() -> str:
    term_program = os.getenv("TERM_PROGRAM")
    if term_program:
        version = os.getenv("TERM_PROGRAM_VERSION")
        token = f"{term_program}/{version}" if version else term_program
        return _sanitize_user_agent_token(token) or "unknown"

    wezterm_version = os.getenv("WEZTERM_VERSION")
    if wezterm_version is not None:
        token = (
            f"WezTerm/{wezterm_version}" if wezterm_version else "WezTerm"
        )
        return _sanitize_user_agent_token(token) or "WezTerm"

    if (
        os.getenv("ITERM_SESSION_ID")
        or os.getenv("ITERM_PROFILE")
        or os.getenv("ITERM_PROFILE_NAME")
    ):
        return "iTerm.app"

    if os.getenv("TERM_SESSION_ID"):
        return "Apple_Terminal"

    if os.getenv("KITTY_WINDOW_ID") or "kitty" in (os.getenv("TERM") or ""):
        return "kitty"

    if os.getenv("ALACRITTY_SOCKET") or os.getenv("TERM") == "alacritty":
        return "Alacritty"

    konsole_version = os.getenv("KONSOLE_VERSION")
    if konsole_version is not None:
        token = (
            f"Konsole/{konsole_version}" if konsole_version else "Konsole"
        )
        return _sanitize_user_agent_token(token) or "Konsole"

    if os.getenv("GNOME_TERMINAL_SCREEN"):
        return "gnome-terminal"

    vte_version = os.getenv("VTE_VERSION")
    if vte_version is not None:
        token = f"VTE/{vte_version}" if vte_version else "VTE"
        return _sanitize_user_agent_token(token) or "VTE"

    if os.getenv("WT_SESSION"):
        return "WindowsTerminal"

    term = os.getenv("TERM")
    if term:
        return _sanitize_user_agent_token(term) or "unknown"

    return "unknown"


def _get_litellm_version() -> str:
    try:
        from importlib.metadata import version

        return version("litellm")
    except Exception:
        return "0.0.0"


def get_chatgpt_originator() -> str:
    originator = os.getenv("CHATGPT_ORIGINATOR") or DEFAULT_ORIGINATOR
    return _safe_header_value(originator) or DEFAULT_ORIGINATOR


def get_chatgpt_user_agent(originator: str) -> str:
    override = os.getenv("CHATGPT_USER_AGENT")
    if override:
        return _safe_header_value(override) or DEFAULT_USER_AGENT
    version = _get_litellm_version()
    os_type = platform.system() or "Unknown"
    os_version = platform.release() or "0"
    arch = platform.machine() or "unknown"
    terminal_ua = _terminal_user_agent()
    suffix = os.getenv("CHATGPT_USER_AGENT_SUFFIX", "").strip()
    suffix = f" ({suffix})" if suffix else ""
    candidate = (
        f"{originator}/{version} ({os_type} {os_version}; {arch}) {terminal_ua}{suffix}"
    )
    return _safe_header_value(candidate) or DEFAULT_USER_AGENT


def get_chatgpt_default_headers(
    access_token: str,
    account_id: Optional[str],
    session_id: Optional[str] = None,
) -> dict:
    originator = get_chatgpt_originator()
    user_agent = get_chatgpt_user_agent(originator)
    headers = {
        "Authorization": f"Bearer {access_token}",
        "content-type": "application/json",
        "accept": "text/event-stream",
        "originator": originator,
        "user-agent": user_agent,
    }
    if session_id:
        headers["session_id"] = session_id
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    return headers


def get_chatgpt_default_instructions() -> str:
    return os.getenv("CHATGPT_DEFAULT_INSTRUCTIONS") or CHATGPT_DEFAULT_INSTRUCTIONS


def _normalize_litellm_params(litellm_params: Optional[Any]) -> dict:
    if litellm_params is None:
        return {}
    if isinstance(litellm_params, dict):
        return litellm_params
    if hasattr(litellm_params, "model_dump"):
        try:
            return litellm_params.model_dump()
        except Exception:
            return {}
    if hasattr(litellm_params, "dict"):
        try:
            return litellm_params.dict()
        except Exception:
            return {}
    return {}


def get_chatgpt_session_id(litellm_params: Optional[Any]) -> Optional[str]:
    params = _normalize_litellm_params(litellm_params)
    for key in ("litellm_session_id", "session_id"):
        value = params.get(key)
        if value:
            return str(value)
    metadata = params.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("session_id")
        if value:
            return str(value)
    for key in ("litellm_trace_id", "litellm_call_id"):
        value = params.get(key)
        if value:
            return str(value)
    return None


def ensure_chatgpt_session_id(litellm_params: Optional[Any]) -> str:
    return get_chatgpt_session_id(litellm_params) or str(uuid4())
