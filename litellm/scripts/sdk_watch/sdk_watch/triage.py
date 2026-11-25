from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Final

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from sdk_watch.changelog import ChangelogEntry
from sdk_watch.sources import Candidate

IGNORED_SECTIONS: Final = frozenset(
    {"Build System", "Chores", "Continuous Integration", "Documentation", "Refactors", "Styles", "Tests"}
)

TRIAGE_PROMPT: Final = """You review provider SDK release notes for LiteLLM. LiteLLM is a Python SDK and proxy that \
exposes OpenAI-compatible and Anthropic-compatible APIs and translates each request to the provider's own API.

Pick the release note items that need a change in LiteLLM itself. An item needs one when it:
- adds a model LiteLLM should know about (model cost map entry, capability flags)
- adds or changes a request parameter LiteLLM must pass through or translate
- adds a response or usage field LiteLLM must keep or map
- adds an API endpoint LiteLLM should route
- changes API behavior LiteLLM's translation depends on, such as a message role newly allowed in a new position

Items about the SDK client itself need no change: retries, typing, auth helpers, examples, docs, CLI tools, \
internal refactors and SDK-only utilities. One release note item can name several API changes, so return one \
entry per API change. Prefer a short list of changes you are sure about. Return an empty list when nothing needs \
a change in LiteLLM."""


class ChangeKind(StrEnum):
    NEW_MODEL = "new_model"
    NEW_REQUEST_PARAM = "new_request_param"
    NEW_RESPONSE_FIELD = "new_response_field"
    NEW_ENDPOINT = "new_endpoint"
    BEHAVIOR_CHANGE = "behavior_change"


class NeededChange(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: ChangeKind
    slug: Annotated[
        str, Field(pattern=r"^[a-z0-9_]{3,48}$", description="Short snake_case name, e.g. mid_conversation_system")
    ]
    sdk_change: str = Field(description="The release note item this comes from, copied verbatim with its links")
    litellm_change: str = Field(description="What LiteLLM must change, in one or two sentences")
    search_terms: tuple[str, ...] = Field(
        min_length=1,
        max_length=4,
        description="Exact identifiers as they appear in code or PR titles: a parameter, field, or model id",
    )


class TriageVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    changes: tuple[NeededChange, ...]


@dataclass(frozen=True, slots=True)
class Finding:
    candidate: Candidate
    change: NeededChange

    @property
    def branch(self) -> str:
        sdk: Final = self.candidate.sdk.name.replace("-", "_")
        version: Final = self.candidate.release.version.replace(".", "_").replace("-", "_")
        return f"litellm_sdk_watch_{sdk}_{version}_{self.change.slug}"


def relevant_entries(candidate: Candidate) -> tuple[ChangelogEntry, ...]:
    return tuple(entry for entry in candidate.release.entries if entry.section not in IGNORED_SECTIONS)


def llm_triage(model: BaseChatModel) -> Callable[[Candidate], TriageVerdict]:
    structured: Final = model.with_structured_output(TriageVerdict, method="json_schema")

    def triage(candidate: Candidate) -> TriageVerdict:
        entries: Final = relevant_entries(candidate)
        if not entries:
            return TriageVerdict(changes=())
        notes: Final = "\n".join(f"- [{entry.section}] {entry.text}" for entry in entries)
        release: Final = candidate.release
        question: Final = (
            f"SDK: {candidate.sdk.name} ({candidate.sdk.repo}), release {release.version} on {release.released}\n"
            f"Release notes:\n{notes}"
        )
        return TriageVerdict.model_validate(structured.invoke([SystemMessage(TRIAGE_PROMPT), HumanMessage(question)]))

    return triage
