from typing_extensions import TypedDict


class AutorouterSavingsMetadata(TypedDict):
    """
    Per-request autorouter cost-savings context recorded into the spend-log
    metadata JSON so daily spend aggregates can estimate dollars saved by
    complexity routing versus the most expensive configured candidate model.

    The baseline per-token prices are captured at routing time (where the
    autorouter config and the router's resolved deployment pricing are known)
    so the daily spend writer can price the counterfactual without needing the
    router instance. ``escalated`` records whether the caller asked to escalate
    to a stronger model on this turn, a quality signal surfaced alongside the
    savings.
    """

    autorouter_name: str
    baseline_model: str
    baseline_input_cost_per_token: float
    baseline_output_cost_per_token: float
    escalated: bool
