from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CostDiscountKey:
    provider: str
    model_pattern: str | None


def parse_cost_discount_key(key: str) -> CostDiscountKey:
    provider, separator, pattern = key.partition("/")
    return CostDiscountKey(provider=provider, model_pattern=pattern if separator else None)
