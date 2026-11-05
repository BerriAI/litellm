from datetime import date
from typing import Final

import pytest
from integration._support.actors import ManagementDatabase

from litellm.repositories.usage_alerts import UsageAlertsRepository
from litellm.types.proxy.usage_alerts import UsageAlertRuleInput


@pytest.mark.asyncio(loop_scope="session")
async def test_batch_checkpoints_preserve_rule_revisions_and_incomplete_status(db: ManagementDatabase) -> None:
    repository: Final = UsageAlertsRepository.from_prisma(db.prisma)
    definition: Final = UsageAlertRuleInput(
        name="Daily evaluation", scope_type="team", scope_id="team-example", metric="spend", direction="increase"
    )
    rule: Final = await repository.save_rule(definition, "qa-admin")
    assert rule is not None
    day: Final = date(2026, 11, 4)
    try:
        assert not await repository.completed_rule_ids((rule,), day)
        await repository.record_evaluations(((rule, "complete"),), day, completed=True)
        assert await repository.completed_rule_ids((rule,), day) == frozenset({rule.id})
        edited: Final = await repository.save_rule(definition.model_copy(update={"multiplier": 4}), "qa-admin", rule)
        assert edited is not None
        assert not await repository.completed_rule_ids((edited,), day)
        await repository.record_evaluations(((rule, "insufficient_coverage"),), day, completed=False)
        assert await repository.get_rule(rule.id) == edited
        await repository.record_evaluations(((edited, "insufficient_coverage"),), day, completed=False)
        incomplete: Final = await repository.get_rule(rule.id)
        assert incomplete is not None and incomplete.evaluation_status == "insufficient_coverage"
        assert incomplete.last_evaluated_day == day
        assert not await repository.completed_rule_ids((edited,), day)
    finally:
        await repository.delete_rule(rule.id)
