"""Shared helpers for the integration presets."""

from collections.abc import Iterable
from typing import Final

from litellm.integrations.otel.model.config import ExporterOwner, ExporterSpec

#: What ``OpenTelemetryV2Config._normalize`` folds in when no destination is configured.
_DEFAULT_SHORTHAND_EXPORTER: Final = ExporterSpec()


def ensure_mappers(mapper_names: Iterable[str], *names: str) -> list[str]:
    """Return ``mapper_names`` with each of ``names`` appended if not already present.

    Order is preserved and duplicates are skipped, so composing several presets
    (or re-applying one) never double-adds a vocabulary.
    """
    result: Final = list(mapper_names)
    for name in names:
        if name not in result:
            result.append(name)
    return result


def credential_gated_exporters(
    exporters: "Iterable[ExporterSpec]", owner: "ExporterOwner"
) -> "tuple[ExporterSpec, ...]":
    """``exporters`` with the operator's destination replaced by a header-gated one.

    Used when a credential-mandatory backend is asked to build without the operator's
    own credentials, so only key/team destinations receive spans. Two things have to
    happen for that to mean "export nowhere": the placeholder console spec that
    ``OpenTelemetryV2Config`` folds in for an empty exporter list is dropped, or every
    span would be printed to stdout, and the gated spec keeps the owner so the
    override filter still recognises which backend this provider speaks for.
    """
    return (
        *(spec for spec in exporters if spec != _DEFAULT_SHORTHAND_EXPORTER),
        ExporterSpec(owner=owner, requires_headers=True),
    )
