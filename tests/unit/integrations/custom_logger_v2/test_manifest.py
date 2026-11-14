import pytest

from litellm.integrations.custom_logger_v2 import (
    Abort,
    Activation,
    CallContext,
    CustomLoggerV2,
    Field,
    Hook,
    InvalidDeclarationError,
    OptOutBy,
    Patch,
    Selection,
    build_manifest,
    on,
)

F = Field


class ContentGuardrail(CustomLoggerV2):
    def __init__(
        self,
        selection: Selection,
        enabled_hooks: frozenset[Hook] | None = None,
        parallel: bool = False,
    ) -> None:
        self.selection = selection
        self.enabled_hooks = enabled_hooks
        self.parallel = parallel

    @on(Hook.pre_call, reads={F.messages}, may_abort=True)
    async def check_input(self, view: object, ctx: CallContext[None]) -> Abort | None: ...

    @on(Hook.post_call_success, reads={F.response}, writes={F.response, F.guardrail_info}, may_abort=True)
    async def check_output(self, view: object, ctx: CallContext[None]) -> Patch | Abort | None: ...

    @on(Hook.log_prepare, reads={F.standard_logging_payload}, writes={F.standard_logging_payload})
    async def mask_logs(self, view: object, ctx: CallContext[None]) -> Patch | None: ...


class PureLogger(CustomLoggerV2):
    @on(Hook.attempt_ended, reads={F.standard_logging_payload})
    async def export(self, view: object, ctx: CallContext[None]) -> None: ...

    @on(Hook.call_ended, reads={F.key_identity}, awaited=True)
    async def settle(self, view: object, ctx: CallContext[None]) -> None: ...


def test_manifest_keeps_only_the_hooks_the_instance_enables() -> None:
    manifest = build_manifest(ContentGuardrail(Selection("input-only"), enabled_hooks=frozenset({Hook.pre_call})))
    assert dict(manifest.subscriptions) == {Hook.pre_call: ContentGuardrail.subscriptions[Hook.pre_call]}


def test_manifest_defaults_to_every_declared_hook_and_the_class_name() -> None:
    manifest = build_manifest(PureLogger())
    assert set(manifest.subscriptions) == {Hook.attempt_ended, Hook.call_ended}
    assert manifest.selection == Selection("PureLogger")


def test_two_instances_of_one_class_get_independent_manifests() -> None:
    first = build_manifest(ContentGuardrail(Selection("a", activation=Activation.opt_in)))
    second = build_manifest(ContentGuardrail(Selection("b"), enabled_hooks=frozenset({Hook.log_prepare})))
    assert (first.selection.name, first.selection.activation) == ("a", Activation.opt_in)
    assert (second.selection.name, set(second.subscriptions)) == ("b", {Hook.log_prepare})


def test_manifest_is_frozen_against_later_instance_changes() -> None:
    guardrail = ContentGuardrail(Selection("before"))
    manifest = build_manifest(guardrail)
    guardrail.selection = Selection("after")
    guardrail.enabled_hooks = frozenset()
    assert manifest.selection.name == "before"
    assert set(manifest.subscriptions) == {Hook.pre_call, Hook.post_call_success, Hook.log_prepare}


def test_enabling_a_hook_the_class_does_not_implement_is_rejected() -> None:
    guardrail = ContentGuardrail(Selection("g"), enabled_hooks=frozenset({Hook.pre_call, Hook.during_call}))
    with pytest.raises(InvalidDeclarationError, match="enables during_call, which it does not implement"):
        build_manifest(guardrail)


def test_instance_parallel_flag_applies_to_its_abort_only_subscriptions() -> None:
    guardrail = ContentGuardrail(
        Selection("g"), enabled_hooks=frozenset({Hook.pre_call, Hook.log_prepare}), parallel=True
    )
    manifest = build_manifest(guardrail)
    assert manifest.subscriptions[Hook.pre_call].parallel is True
    assert manifest.subscriptions[Hook.log_prepare].parallel is False
    assert ContentGuardrail.subscriptions[Hook.pre_call].parallel is False


def test_instance_parallel_flag_on_an_abortable_writer_is_rejected() -> None:
    guardrail = ContentGuardrail(Selection("g"), enabled_hooks=frozenset({Hook.post_call_success}), parallel=True)
    with pytest.raises(InvalidDeclarationError, match="parallel is only for"):
        build_manifest(guardrail)


def test_logging_only_step_can_be_required_while_the_instance_is_opt_in() -> None:
    guardrail = ContentGuardrail(Selection("g", activation=Activation.opt_in))
    guardrail.activation_overrides = {Hook.log_prepare: Activation.required}
    manifest = build_manifest(guardrail)
    assert manifest.subscriptions[Hook.log_prepare].activation is Activation.required
    assert manifest.subscriptions[Hook.pre_call].activation is None
    assert manifest.selection.activation is Activation.opt_in


def test_activation_override_for_a_hook_the_instance_does_not_enable_is_rejected() -> None:
    guardrail = ContentGuardrail(Selection("g"), enabled_hooks=frozenset({Hook.pre_call}))
    guardrail.activation_overrides = {Hook.log_prepare: Activation.required}
    with pytest.raises(InvalidDeclarationError, match="overrides activation for log_prepare"):
        build_manifest(guardrail)


@pytest.mark.parametrize(
    ("enabled_hooks", "selection"),
    [
        (frozenset({Hook.pre_call}), Selection("g", honors_no_log=True)),
        (frozenset({Hook.pre_call}), Selection("g", opt_out_by=frozenset({OptOutBy.admin, OptOutBy.caller}))),
        (frozenset({Hook.log_prepare}), Selection("g", honors_no_log=True)),
    ],
)
def test_callers_cannot_switch_off_a_callback_that_writes_or_aborts(
    enabled_hooks: frozenset[Hook], selection: Selection
) -> None:
    with pytest.raises(InvalidDeclarationError, match="neither no-log nor a caller opt-out may remove it"):
        build_manifest(ContentGuardrail(selection, enabled_hooks=enabled_hooks))


def test_read_only_logger_may_honor_no_log_and_caller_opt_out() -> None:
    logger = PureLogger()
    logger.selection = Selection("exporter", opt_out_by=frozenset(OptOutBy), honors_no_log=True)
    assert build_manifest(logger).selection.honors_no_log is True
