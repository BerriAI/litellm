"""Embedding and synthesis calls the analysis worker sends through the LiteLLM gateway, so they are billed like any
other traffic. The client carries `ANALYSIS_HEADERS`, whose tag keeps these calls out of the analyzed requests."""

import json
from collections.abc import Iterable
from itertools import chain
from typing import Final

from openai import AsyncOpenAI
from pydantic import TypeAdapter, ValidationError

from litellm.models.task_insights import TASK_INSIGHTS_ANALYSIS_TAG
from litellm.task_insights.discovery import SynthesizedTask

_EMBED_BATCH: Final = 32
ANALYSIS_HEADERS: Final = {"x-litellm-tags": TASK_INSIGHTS_ANALYSIS_TAG}
_SYNTHESIS: Final = TypeAdapter(SynthesizedTask)
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

Reply with only a JSON object with the keys name, description, inclusion_criteria, exclusion_criteria,
required_capabilities and example_indices.

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
        batches: Final = tuple(texts[offset : offset + _EMBED_BATCH] for offset in range(0, len(texts), _EMBED_BATCH))
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
            return _SYNTHESIS.validate_json(_json_object(content))
        except ValidationError:
            return SynthesizedTask(
                name="Unnamed group",
                description="The synthesis model returned no usable definition. Review and rename before freezing.",
                inclusion_criteria="",
                exclusion_criteria="",
            )
