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
        *(spec for spec in exporters if not _is_stdout_placeholder(spec)),
        ExporterSpec(owner=owner, requires_headers=True),
    )


#: The fields ``OpenTelemetryV2Config._normalize`` fills the synthesized spec from.
_SHORTHAND_FIELDS: Final = frozenset({"kind", "endpoint", "headers"})


def _is_stdout_placeholder(spec: "ExporterSpec") -> bool:
    """Whether ``spec`` is the placeholder ``_normalize`` folds in for an empty list.

    Two conditions. It must have nowhere to send a span, which is what
    ``exporter_transport`` answers: an unrecognized or misspelled kind falls back to the
    console exporter, so comparing against the literal ``"console"`` would miss it. And
    every non-shorthand field must still be at its default, which is what says the
    operator did not ask for it: an exporter they configured survives, and so does the
    gated spec this module appends, which would otherwise eat itself when one preset
    layers onto another.
    """
    from litellm.integrations.otel.plumbing.providers import exporter_transport

    return (
        exporter_transport(spec.kind) == "headerless"
        and spec.endpoint is None
        and spec.model_dump(exclude_defaults=True).keys() <= _SHORTHAND_FIELDS
    )
