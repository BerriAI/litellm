from dataclasses import dataclass, field
from datetime import date

from langchain_core.messages import BaseMessage
from sdk_watch.changelog import ChangelogEntry, Release
from sdk_watch.sources import Candidate, Sdk
from sdk_watch.triage import ChangeKind, NeededChange, TriageVerdict, llm_triage

VERDICT = TriageVerdict(
    changes=(
        NeededChange(
            kind=ChangeKind.BEHAVIOR_CHANGE,
            slug="mid_conversation_system",
            sdk_change="**api:** mid-conversation system blocks",
            litellm_change="Keep system messages in place",
            search_terms=("mid-conversation system",),
        ),
    )
)


@dataclass
class RecordingModel:
    prompts: list[list[BaseMessage]] = field(default_factory=list)

    def with_structured_output(self, schema: type, method: str) -> "RecordingModel":
        assert schema is TriageVerdict
        return self

    def invoke(self, messages: list[BaseMessage]) -> TriageVerdict:
        self.prompts.append(messages)
        return VERDICT


def candidate(*entries: ChangelogEntry) -> Candidate:
    return Candidate(
        sdk=Sdk(name="anthropic", repo="anthropics/anthropic-sdk-python"),
        release=Release(version="0.105.0", released=date(2026, 5, 28), entries=entries),
    )


def test_a_release_of_only_chores_and_docs_costs_no_model_call() -> None:
    model = RecordingModel()

    verdict = llm_triage(model)(
        candidate(ChangelogEntry("Chores", "bump deps"), ChangelogEntry("Documentation", "fix typo"))
    )

    assert verdict == TriageVerdict(changes=())
    assert model.prompts == []


def test_the_model_sees_api_entries_but_not_chores_and_its_verdict_is_returned() -> None:
    model = RecordingModel()

    verdict = llm_triage(model)(
        candidate(
            ChangelogEntry("Features", "**api:** mid-conversation system blocks"),
            ChangelogEntry("Chores", "**internal:** bump deps"),
        )
    )

    assert verdict == VERDICT
    [[_, question]] = model.prompts
    assert "- [Features] **api:** mid-conversation system blocks" in question.content
    assert "bump deps" not in question.content
    assert "release 0.105.0 on 2026-05-28" in question.content
