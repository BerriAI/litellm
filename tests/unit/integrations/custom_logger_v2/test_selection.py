import pytest

from litellm.integrations.custom_logger_v2 import Activation, OptOutBy, RequestSelection, Selection, is_selected

ADMIN = frozenset({OptOutBy.admin})
BOTH = frozenset({OptOutBy.admin, OptOutBy.caller})


@pytest.mark.parametrize(
    ("selection", "activation", "request_selection", "expected"),
    [
        (Selection("limiter"), Activation.required, RequestSelection(admin_opt_outs=frozenset({"limiter"})), True),
        (
            Selection("limiter", honors_no_log=True, opt_out_by=BOTH),
            Activation.required,
            RequestSelection(caller_opt_outs=frozenset({"limiter"}), no_log=True),
            True,
        ),
        (Selection("langfuse"), Activation.default_on, RequestSelection(), True),
        (Selection("langfuse"), Activation.default_on, RequestSelection(admin_opt_outs=frozenset({"langfuse"})), False),
        (Selection("langfuse"), Activation.default_on, RequestSelection(caller_opt_outs=frozenset({"langfuse"})), True),
        (
            Selection("langfuse", opt_out_by=BOTH),
            Activation.default_on,
            RequestSelection(caller_opt_outs=frozenset({"langfuse"})),
            False,
        ),
        (
            Selection("v2_langfuse", aliases=frozenset({"langfuse"})),
            Activation.default_on,
            RequestSelection(admin_opt_outs=frozenset({"langfuse"})),
            False,
        ),
        (Selection("guard"), Activation.opt_in, RequestSelection(), False),
        (Selection("guard"), Activation.opt_in, RequestSelection(requested=frozenset({"guard"})), True),
        (
            Selection("guard", opt_out_by=ADMIN),
            Activation.opt_in,
            RequestSelection(requested=frozenset({"guard"}), admin_opt_outs=frozenset({"guard"})),
            False,
        ),
        (Selection("langfuse", honors_no_log=True), Activation.default_on, RequestSelection(no_log=True), False),
        (Selection("langfuse"), Activation.default_on, RequestSelection(no_log=True), True),
        (
            Selection("langfuse", honors_no_log=True, opt_out_by=frozenset()),
            Activation.default_on,
            RequestSelection(no_log=True),
            False,
        ),
    ],
)
def test_is_selected(
    selection: Selection, activation: Activation, request_selection: RequestSelection, expected: bool
) -> None:
    assert is_selected(selection, activation, request_selection) is expected
