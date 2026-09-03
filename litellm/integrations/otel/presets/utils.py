"""Shared helpers for the integration presets."""

from collections.abc import Iterable
from typing import Final

from litellm.integrations.otel.model.config import ExporterOwner, ExporterSpec


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
        *(spec for spec in exporters if not _prints_to_stdout(spec)),
        ExporterSpec(owner=owner, requires_headers=True),
    )


def _prints_to_stdout(spec: "ExporterSpec") -> bool:
    """Whether ``spec`` is the placeholder ``_normalize`` folds in for an empty list.

    Identified by what it does rather than by equality with a default instance:
    ``OpenTelemetryV2Config`` reads the standard ``OTEL_EXPORTER_OTLP_*`` env vars, so
    the shorthand it synthesizes is a real operator destination whenever any of them
    is set, and only a console exporter with no endpoint prints every span.
    """
    return spec.kind == "console" and spec.endpoint is None
