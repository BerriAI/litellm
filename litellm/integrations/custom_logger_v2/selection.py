from dataclasses import dataclass

from litellm.integrations.custom_logger_v2.enums import Activation, OptOutBy


@dataclass(frozen=True, slots=True)
class Selection:
    name: str
    aliases: frozenset[str] = frozenset()
    activation: Activation = Activation.default_on
    opt_out_by: frozenset[OptOutBy] = frozenset({OptOutBy.admin})
    honors_no_log: bool = False
