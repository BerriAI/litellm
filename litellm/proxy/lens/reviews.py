import hashlib
import json
from collections.abc import Callable
from types import MappingProxyType
from typing import Final

from .models import Extraction, LensSettings, Review


def criteria_key(settings: LensSettings) -> str:
    payload: Final = (
        settings.context.strip(),
        tuple(sorted((check.id, check.instruction.strip()) for check in settings.analysis_checks)),
        settings.model,
    )
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def map_extraction(extraction: Extraction, identity: Callable[[str], str]) -> Extraction:
    return extraction.model_copy(
        update=MappingProxyType(
            {
                "observations": tuple(
                    observation.model_copy(
                        update=MappingProxyType(
                            {
                                "evidence": tuple(
                                    quote.model_copy(
                                        update=MappingProxyType({"execution_id": identity(quote.execution_id)})
                                    )
                                    for quote in observation.evidence
                                )
                            }
                        )
                    )
                    for observation in extraction.observations
                )
            }
        )
    )


def map_review(review: Review, identity: Callable[[str], str]) -> Review:
    return review.model_copy(
        update=MappingProxyType(
            {
                "execution_id": identity(review.execution_id),
                "extraction": map_extraction(review.extraction, identity) if review.extraction else None,
            }
        )
    )
