from dataclasses import dataclass
from typing import Final

from litellm.integrations.custom_logger_v2.enums import Activation, OptOutBy


@dataclass(frozen=True, slots=True)
class Selection:
    name: str
    aliases: frozenset[str] = frozenset()
    activation: Activation = Activation.default_on
    opt_out_by: frozenset[OptOutBy] = frozenset({OptOutBy.admin})
    honors_no_log: bool = False


@dataclass(frozen=True, slots=True)
class RequestSelection:
    requested: frozenset[str] = frozenset()
    admin_opt_outs: frozenset[str] = frozenset()
    caller_opt_outs: frozenset[str] = frozenset()
    no_log: bool = False


def _opted_out(selection: Selection, names: frozenset[str], request: RequestSelection) -> bool:
    by_admin: Final = OptOutBy.admin in selection.opt_out_by and not names.isdisjoint(request.admin_opt_outs)
    by_caller: Final = OptOutBy.caller in selection.opt_out_by and not names.isdisjoint(request.caller_opt_outs)
    return by_admin or by_caller or (request.no_log and selection.honors_no_log)


def is_selected(selection: Selection, activation: Activation, request: RequestSelection) -> bool:
    if activation is Activation.required:
        return True
    names: Final = selection.aliases | {selection.name}
    if _opted_out(selection, names, request):
        return False
    return activation is Activation.default_on or not names.isdisjoint(request.requested)
