"""Embedding and synthesis calls the analysis worker sends through the LiteLLM gateway, so they are billed like any
other traffic. The client carries `ANALYSIS_HEADERS`, whose tag keeps these calls out of the analyzed requests."""

import json
from collections.abc import Iterable
from itertools import chain
from typing import Final

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.models.task_insights import TASK_INSIGHTS_ANALYSIS_TAG, TaskCapability
from litellm.task_insights.discovery import SynthesizedTask

# DashScope's compatible endpoint rejects more than 20 inputs per call (400 InvalidParameter, 2026-11-12).
EMBED_BATCH_SIZE: Final = 20
ANALYSIS_HEADERS: Final = {"x-litellm-tags": TASK_INSIGHTS_ANALYSIS_TAG}


class _SynthesisReply(BaseModel):
    """What the synthesis model sends back. Models sometimes list criteria as bullets, which are folded into one
    sentence each rather than discarding an otherwise usable definition."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str
    description: str = ""
    inclusion_criteria: str | tuple[str, ...] = ""
    exclusion_criteria: str | tuple[str, ...] = ""
    required_capabilities: tuple[TaskCapability, ...] = ()
    example_indices: tuple[int, ...] = ()


_SYNTHESIS: Final = TypeAdapter(_SynthesisReply)


def _sentence(value: str | tuple[str, ...]) -> str:
    return (
        value if isinstance(value, str) else " ".join(item.strip().rstrip(".") + "." for item in value if item.strip())
    )


def _task(reply: _SynthesisReply) -> SynthesizedTask:
    return SynthesizedTask(
        name=reply.name,
        description=reply.description,
        inclusion_criteria=_sentence(reply.inclusion_criteria),
        exclusion_criteria=_sentence(reply.exclusion_criteria),
        required_capabilities=reply.required_capabilities,
        example_indices=reply.example_indices,
    )


_INTENTS: Final = TypeAdapter(tuple[str, ...])
_DISTILL_BATCH: Final = 25
_DISTILL_PROMPT: Final = """Each numbered item is a request a developer sent to a coding agent.
For every item write one short sentence, at most 15 words, saying what kind of work it asks for.
Leave out repository, product, company, person, file and variable names.

Reply with only a JSON array of strings, one per item, in the same order.

Items:
{requests}"""
_SYNTHESIS_PROMPT: Final = """You are naming one group of requests that developers sent to coding agents.
Read the numbered requests and describe the kind of work they share.

Rules:
- Name the task type, such as "Debug failing tests", in at most 6 words.
- Never use repository, product, company, person or file names. Describe the work, not the codebase.
- inclusion_criteria says what a request must ask for to belong. exclusion_criteria names the nearest kinds of work
  that do not belong.
- required_capabilities lists only what every request needs from a model, chosen from function_calling, vision,
  reasoning and structured_output. Leave it empty when unsure.
- example_indices lists up to 3 request numbers that best represent the group.

Reply with only a JSON object. name, description, inclusion_criteria and exclusion_criteria are strings of one or
two sentences, required_capabilities is an array of strings, and example_indices is an array of integers.

Requests:
{requests}"""


class GatewayEmbedder:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        batches: Final = tuple(
            texts[offset : offset + EMBED_BATCH_SIZE] for offset in range(0, len(texts), EMBED_BATCH_SIZE)
        )
        responses: Final = tuple(
            [await self._client.embeddings.create(model=self._model, input=list(batch)) for batch in batches]
        )
        ordered: Final = chain.from_iterable(
            sorted(response.data, key=lambda item: item.index) for response in responses
        )
        return tuple(tuple(item.embedding) for item in ordered)


def _numbered(texts: Iterable[str]) -> str:
    return "\n".join(f"{index}. {json.dumps(text)}" for index, text in enumerate(texts))


def _json_array(text: str) -> str:
    start: Final = text.find("[")
    end: Final = text.rfind("]")
    return text[start : end + 1] if start != -1 and end > start else text


def _json_object(text: str) -> str:
    start: Final = text.find("{")
    end: Final = text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


class GatewaySynthesizer:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def distill(self, texts: tuple[str, ...]) -> tuple[str, ...]:
        batches: Final = tuple(
            texts[offset : offset + _DISTILL_BATCH] for offset in range(0, len(texts), _DISTILL_BATCH)
        )
        distilled: Final = tuple([await self._distill_batch(batch) for batch in batches])
        return tuple(chain.from_iterable(distilled))

    async def _distill_batch(self, texts: tuple[str, ...]) -> tuple[str, ...]:
        """A batch whose reply does not parse falls back to its own excerpts, so one bad reply costs only that
        batch's distillation, never the run."""
        content: Final = await self._complete(
            _DISTILL_PROMPT.format(requests=_numbered(t[:1500] for t in texts)), 60 * len(texts)
        )
        try:
            intents: Final = _INTENTS.validate_json(_json_array(content))
        except ValidationError:
            return tuple(text[:300] for text in texts)
        return intents if len(intents) == len(texts) else tuple(text[:300] for text in texts)

    async def _complete(self, prompt: str, max_tokens: int) -> str:
        response: Final = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def synthesize(self, examples: tuple[str, ...]) -> SynthesizedTask:
        content: Final = await self._complete(_SYNTHESIS_PROMPT.format(requests=_numbered(examples)), 800)
        try:
            return _task(_SYNTHESIS.validate_json(_json_object(content)))
        except ValidationError:
            return SynthesizedTask(
                name="Unnamed group",
                description="The synthesis model returned no usable definition. Review and rename before freezing.",
                inclusion_criteria="",
                exclusion_criteria="",
            )
