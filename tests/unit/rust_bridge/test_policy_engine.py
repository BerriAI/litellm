from collections.abc import Iterator
from typing import Final

import pytest

from litellm.proxy.policy_engine.attachment_registry import AttachmentRegistry, get_attachment_registry
from litellm.proxy.policy_engine.policy_registry import PolicyRegistry, get_policy_registry
from litellm.rust_bridge.policy_engine import current_policy_engine
from litellm.types.proxy.policy_engine import (
    GuardrailPipeline,
    PipelineStep,
    Policy,
    PolicyAttachment,
    PolicyGuardrails,
    PolicyMatchContext,
)


@pytest.fixture
def registries() -> Iterator[tuple[PolicyRegistry, AttachmentRegistry]]:
    policy_registry: Final = get_policy_registry()
    attachment_registry: Final = get_attachment_registry()
    policy_registry.clear()
    attachment_registry.clear()
    yield policy_registry, attachment_registry
    policy_registry.clear()
    attachment_registry.clear()


def test_a_registry_change_is_resolved_with_the_new_policies(
    registries: tuple[PolicyRegistry, AttachmentRegistry],
) -> None:
    policy_registry, attachment_registry = registries
    policy_registry.add_policy("base", Policy(guardrails=PolicyGuardrails(add=["pii"])))
    attachment_registry.add_attachment(PolicyAttachment(policy="base", scope="*"))
    context: Final = PolicyMatchContext(model="gpt-4")

    first: Final = current_policy_engine()
    assert first.resolve(context).guardrails == ("pii",)
    assert current_policy_engine() is first
    policy_registry.add_policy("base", Policy(guardrails=PolicyGuardrails(add=["audit"])))
    assert current_policy_engine().resolve(context).guardrails == ("audit",)
    attachment_registry.remove_attachments_for_policy("base")
    assert current_policy_engine().resolve(context).guardrails == ()


def test_pipelines_come_back_as_the_policy_objects_of_the_registry_or_version(
    registries: tuple[PolicyRegistry, AttachmentRegistry],
) -> None:
    policy_registry: Final = registries[0]
    pipeline: Final = GuardrailPipeline(mode="pre_call", steps=[PipelineStep(guardrail="judge")])
    version_pipeline: Final = GuardrailPipeline(mode="post_call", steps=[PipelineStep(guardrail="pii")])
    policy_registry.add_policy("base", Policy(pipeline=pipeline))
    policy_registry.add_policy("draft", Policy())
    policy_registry._policies_by_id = {"v1": ("draft", Policy(pipeline=version_pipeline))}  # pyright: ignore[reportPrivateUsage]  # no public setter for the version cache

    resolution: Final = current_policy_engine().resolve(PolicyMatchContext(model="gpt-4"), ("base",), ("v1",))

    assert resolution.pipelines == (("base", pipeline), ("draft", version_pipeline))
    assert resolution.pipelines[0][1] is pipeline
    assert resolution.pipeline_managed_guardrails == frozenset({"judge", "pii"})
