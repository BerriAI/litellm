from collections.abc import Iterator


def deployment_pricing_models(base_model: object, model: object) -> tuple[str, ...]:
    """Cost-map keys to price a deployment under, ``model_info.base_model`` first, then the
    deployment's own model, blanks and duplicates dropped. Every spend and routing surface that
    prices a deployment walks this order, so a ``base_model`` the cost map does not know falls
    through to the deployment model instead of billing or ranking at the unmapped default."""
    return tuple(dict.fromkeys(_named_models(base_model, model)))


def _named_models(*candidates: object) -> Iterator[str]:
    return (candidate for candidate in candidates if isinstance(candidate, str) and candidate)
