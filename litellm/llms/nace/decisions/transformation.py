from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig

_MAPPING: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_LABELS: Final[TypeAdapter[tuple[str, ...]]] = TypeAdapter(tuple[str, ...])


def _normalize_answer(value: object) -> Mapping[str, object]:
    answer: Final = _MAPPING.validate_python(value)
    legend: Final = answer.get("legend")
    if answer.get("type") != "score" or not isinstance(legend, list):
        return answer
    labels: Final = _LABELS.validate_python(legend)
    return {**answer, "legend": {str(index): label for index, label in enumerate(labels)}}


class NaceDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("NACE_API_KEY",)
    api_base_env = ("NACE_API_BASE", "NACE_BASE_URL")

    def get_default_api_base(self) -> str | None:
        return "https://console.nace.ai"

    def unwrap_response(self, payload: object) -> object:
        response: Final = _MAPPING.validate_python(payload)
        answers: Final = _MAPPING.validate_python(response.get("answers"))
        return {**response, "answers": {name: _normalize_answer(answer) for name, answer in answers.items()}}
