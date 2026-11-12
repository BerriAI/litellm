"""Bucketing with Jev, TypeSafe's decision model, through the gateway's own `/v1/systemone` route.

https://docs.typesafe.ai/primitives/choice. A Choice always picks an option even when nothing fits, so the same call
asks a companion Noul whether the request fits any definition, and both read the definitions from the shared state.
Calling through the gateway reuses LiteLLM's TypeSafe provider, so each decision is priced, billed and logged like any
other request, and the analysis tag keeps it out of the analyzed traffic. Jev's probabilities are its own estimate,
not a validated accuracy, so the thresholds are checked against labeled requests before they are trusted.
"""

from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import ValidationError

from litellm.models.task_insights import TASK_INSIGHTS_ANALYSIS_TAG, Decision, PendingRequest, TaskDefinition
from litellm.types.decisions import ChoiceAnswer, ChoiceQuestion, DecisionsResponse, NoulAnswer, NoulQuestion

MAX_JEV_OPTIONS: Final = 255
_TASK_QUESTION: Final = "task"
_FIT_QUESTION: Final = "fits"


def jev_state(tasks: tuple[TaskDefinition, ...], request: PendingRequest) -> Mapping[str, object]:
    return {
        "request": request.request_text,
        "session_context": request.context_text or "",
        "task_definitions": [
            {
                "id": task.task_id,
                "name": task.name,
                "description": task.description,
                "includes": task.inclusion_criteria,
                "excludes": task.exclusion_criteria,
            }
            for task in tasks
        ],
    }


def jev_questions(tasks: tuple[TaskDefinition, ...]) -> Mapping[str, ChoiceQuestion | NoulQuestion]:
    return {
        _TASK_QUESTION: ChoiceQuestion(
            type="choice",
            instructions="Which entry in task_definitions best describes what the request asks for?",
            criteria={task.task_id: f"{task.name}: {task.description}" for task in tasks},
        ),
        _FIT_QUESTION: NoulQuestion(
            type="noul",
            instructions="Does the request belong to at least one entry in task_definitions?",
            criteria={
                "true": "It meets that entry's includes and none of its excludes",
                "false": "It fits no entry, or only matches an entry's excludes",
            },
        ),
    }


def jev_decision(
    response: DecisionsResponse,
    *,
    task_ids: frozenset[str],
    requested_model: str,
    min_probability: float,
    min_fit: float,
) -> Decision:
    """Assigns the chosen task only when its own probability and the fit both clear their thresholds."""
    choice: Final = response.answers.get(_TASK_QUESTION)
    fits: Final = response.answers.get(_FIT_QUESTION)
    served_by: Final = response.model or requested_model
    if not isinstance(choice, ChoiceAnswer) or choice.choice not in task_ids:
        return Decision(outcome="failed", task_id=None, decision_model=served_by, error="InvalidTaskChoice")
    selected: Final = choice.probabilities.get(choice.choice)
    fit: Final = fits.noul if isinstance(fits, NoulAnswer) else None
    accepted: Final = selected is not None and selected >= min_probability and fit is not None and fit >= min_fit
    return Decision(
        outcome="assigned" if accepted else "unknown",
        task_id=choice.choice if accepted else None,
        probabilities={task_id: value for task_id, value in choice.probabilities.items() if task_id in task_ids},
        top_probability=selected,
        fit_probability=fit,
        decision_model=served_by,
        provenance={
            "requested_model": requested_model,
            "confidence": choice.confidence,
            "input_tokens": response.usage.input_tokens if response.usage is not None else None,
            "min_probability": min_probability,
            "min_fit": min_fit,
        },
    )


class GatewayJevDecider:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        url: str,
        api_key: str,
        model: str,
        min_probability: float = 0.5,
        min_fit: float = 0.5,
    ) -> None:
        self._client = client
        self._url = url
        self._api_key = api_key
        self._model = model
        self._min_probability = min_probability
        self._min_fit = min_fit

    @property
    def decision_model(self) -> str:
        return self._model

    def _failed(self, error: str) -> Decision:
        return Decision(outcome="failed", task_id=None, decision_model=self._model, error=error)

    async def decide(self, tasks: tuple[TaskDefinition, ...], request: PendingRequest) -> Decision:
        if len(tasks) > MAX_JEV_OPTIONS:
            return self._failed("TooManyTasks")
        questions: Final = jev_questions(tasks)
        response: Final = await self._client.post(
            self._url,
            json={
                "model": self._model,
                "state": jev_state(tasks, request),
                "questions": {key: question.model_dump(exclude_none=True) for key, question in questions.items()},
            },
            headers={"Authorization": f"Bearer {self._api_key}", "x-litellm-tags": TASK_INSIGHTS_ANALYSIS_TAG},
        )
        if response.is_error:
            return self._failed(f"HTTP{response.status_code}")
        try:
            parsed: Final = DecisionsResponse.model_validate_json(response.content)
        except ValidationError:
            return self._failed("UnreadableDecision")
        return jev_decision(
            parsed,
            task_ids=frozenset(task.task_id for task in tasks),
            requested_model=self._model,
            min_probability=self._min_probability,
            min_fit=self._min_fit,
        )
