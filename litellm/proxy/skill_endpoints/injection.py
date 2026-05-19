"""
Skill injection into chat completion requests (S2-05).

Lets callers pass a ``skills: ["fact-check@v3", "summarize"]`` array in the
chat completion body. For each skill:
  - the ``system_prompt_template`` is rendered (Jinja2 if available, else
    str.format-style ``{name}`` substitution) using values from the optional
    ``skill_inputs`` dict in the request body, and prepended as a system
    message.
  - ``tool_schema`` entries (an OpenAI-shaped ``tools`` array) are merged
    into the request's ``tools`` array, deduplicated by function name.

Errors that do NOT abort the request (best-effort): missing template
variables (substituted as empty string), unknown skill version (falls
through to ``latest`` if available).

Errors that DO abort: completely unknown ``skill_id`` (HTTP 400), explicit
``@version`` requested but no row matches (HTTP 404).
"""

from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger


def _split_skill_ref(ref: str) -> Tuple[str, Optional[str]]:
    """`"fact-check@v3"` → ``("fact-check", "v3")``; `"summarize"` → ``("summarize", None)``."""
    if "@" in ref:
        skill_id, version = ref.split("@", 1)
        return skill_id.strip(), version.strip() or None
    return ref.strip(), None


def _render_prompt(template: Optional[str], inputs: Dict[str, Any]) -> str:
    """Render a system prompt template.

    Prefers Jinja2 if available (matches our existing prompt-template
    stack); falls back to safe-substitution ``str.format_map(...)`` with
    missing keys replaced by an empty string.
    """
    if not template:
        return ""
    try:
        from jinja2 import Environment, StrictUndefined

        env = Environment(undefined=StrictUndefined, autoescape=False)
        return env.from_string(template).render(**(inputs or {}))
    except Exception:
        try:

            class _SafeDict(dict):
                def __missing__(self, key: str) -> str:
                    return ""

            return template.format_map(_SafeDict(**(inputs or {})))
        except Exception as e:
            verbose_proxy_logger.debug("skill prompt render failed: %s", e)
            return template


def _merge_tools(
    existing: Optional[List[Dict[str, Any]]],
    additions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge tool definitions; dedup by ``function.name``."""
    merged: List[Dict[str, Any]] = list(existing or [])
    seen = {
        (t.get("function") or {}).get("name") for t in merged if isinstance(t, dict)
    }
    for tool in additions:
        if not isinstance(tool, dict):
            continue
        name = (tool.get("function") or {}).get("name")
        if name in seen:
            continue
        merged.append(tool)
        seen.add(name)
    return merged


async def _fetch_skill_row(skill_id: str, version: Optional[str]):
    """Return the skill row matching (skill_id, version), or None."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return None
    where: Dict[str, Any] = {"skill_id": skill_id, "source": "custom"}
    if version is not None:
        where["version"] = version
    try:
        if version is None:
            # Latest by skill_id alone — caller didn't pin a version.
            return await prisma_client.db.litellm_skillstable.find_unique(
                where={"skill_id": skill_id}
            )
        # Versioned: find_many to allow filtering by source+version, return first.
        rows = await prisma_client.db.litellm_skillstable.find_many(where=where, take=1)
        return rows[0] if rows else None
    except Exception as e:
        verbose_proxy_logger.debug(
            "skill fetch failed for %s@%s: %s", skill_id, version, e
        )
        return None


async def inject_skills_into_chat_request(data: Dict[str, Any]) -> None:
    """Mutate ``data`` in place: resolve skills, inject prompt + tools.

    Looks for ``data["skills"]`` (list of ``skill_id`` or ``skill_id@version``)
    and ``data["skill_inputs"]`` (dict of variables for prompt rendering).
    No-ops cleanly when no skills field is present.
    """
    skill_refs = data.get("skills")
    if not skill_refs or not isinstance(skill_refs, list):
        return
    inputs = data.get("skill_inputs") or {}

    prompt_parts: List[str] = []
    tool_additions: List[Dict[str, Any]] = []

    for ref in skill_refs:
        if not isinstance(ref, str):
            continue
        skill_id, version = _split_skill_ref(ref)
        row = await _fetch_skill_row(skill_id, version)
        if row is None:
            if version is not None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Skill version not found: {ref}",
                )
            raise HTTPException(status_code=400, detail=f"Unknown skill: {ref}")
        if getattr(row, "source", None) != "custom":
            raise HTTPException(
                status_code=400,
                detail=f"Skill '{skill_id}' is not an xct skill (source != 'custom').",
            )
        rendered = _render_prompt(getattr(row, "system_prompt_template", None), inputs)
        if rendered:
            prompt_parts.append(rendered)
        tool_schema = getattr(row, "tool_schema", None) or []
        if isinstance(tool_schema, list):
            tool_additions.extend(t for t in tool_schema if isinstance(t, dict))

    if prompt_parts:
        messages: List[Dict[str, Any]] = data.setdefault("messages", [])
        # Single combined system message at index 0; preserves any existing
        # leading system message by prepending instead of replacing.
        combined = "\n\n".join(prompt_parts)
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = (
                combined + "\n\n" + str(messages[0].get("content") or "")
            )
        else:
            messages.insert(0, {"role": "system", "content": combined})

    if tool_additions:
        data["tools"] = _merge_tools(data.get("tools"), tool_additions)

    # Strip the meta fields so they don't ride downstream to the provider.
    data.pop("skills", None)
    data.pop("skill_inputs", None)
