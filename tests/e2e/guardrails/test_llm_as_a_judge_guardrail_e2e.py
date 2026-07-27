"""Live e2e: the llm_as_a_judge guardrail rejects a response a second LLM scores below threshold.

This guardrail runs after the model, not before it: it hands the completion to a
judge model, scores it against weighted criteria, and rejects the whole call with
HTTP 422 when the weighted score falls under `overall_threshold`. The criterion
here is "the response must be written entirely in French", which keeps the judge
off a coin flip: an English answer scores 0 and a French answer scores ~100 against
a threshold of 80, so neither verdict is close to the line.

Both halves are asserted, and neither is asserted on the response alone. The judge
is itself an LLM call that fails open on any internal error, and a fail-open returns
an ordinary 200 that is indistinguishable from an approval: same status, same body,
and `x-litellm-applied-guardrails` still names the guardrail. So the passing case
also reads the guardrail's own run log at /guardrails/usage/logs, where an approval
is recorded as `passed` and a fail-open as `flagged`. Without that leg the accept
half would pass just as happily on a build where adjudication never ran at all.

The judge model is `openai/gpt-4.1` rather than the suite's usual gpt-5.5 because
the guardrail hardcodes `temperature=0` on its judge call and gpt-5.5 rejects any
temperature other than 1; the resulting BadRequestError is swallowed and the
guardrail fails open, so gpt-5.5 cannot adjudicate anything. Prompts carry a
unique marker so the proxy's response cache cannot serve a previous run's answer
and skip the judge.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import UnknownApiError, unwrap
from guardrails_client import GuardrailsClient
from lifecycle import ResourceManager
from native_guardrails import (
    JudgeCriterionBody,
    LLMAsAJudgeParamsBody,
    chat,
    poll_guardrail_usage_logs,
    register_guardrail,
    response_text,
)

pytestmark = pytest.mark.e2e

JUDGE_MODEL = "openai/gpt-4.1"
CRITERION = "answers_in_french"
THRESHOLD = 80.0


class TestLLMAsAJudgeGuardrail:
    @pytest.mark.covers(
        "guardrail.llm_as_a_judge.post_call.blocks",
        exercised_on=["chat_completions"],
    )
    def test_response_failing_the_criterion_is_rejected_while_a_passing_one_is_returned(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        marker = unique_marker()
        name = f"e2e-llm-judge-{marker}"
        guardrail_id = register_guardrail(
            client,
            name,
            LLMAsAJudgeParamsBody(
                mode="post_call",
                default_on=False,
                judge_model=JUDGE_MODEL,
                overall_threshold=THRESHOLD,
                on_failure="block",
                criteria=[
                    JudgeCriterionBody(
                        name=CRITERION,
                        weight=100,
                        description="The assistant's response must be written entirely in French.",
                    )
                ],
            ),
        )
        resources.defer(lambda: client.delete_guardrail(guardrail_id))

        rejected = chat(
            client,
            scoped_key,
            prompt=f"Case {marker}. Reply with exactly this and nothing else: The weather is nice today.",
            guardrail_name=name,
        )
        match rejected:
            case UnknownApiError(status_code=422, body=body):
                assert "score below threshold" in body, (
                    f"the rejection must say the judge scored the response too low, got: {body[:400]}"
                )
                assert CRITERION in body, (
                    "the rejection must carry the verdict for the criterion under test, so the "
                    f"caller learns which one failed; got: {body[:400]}"
                )
            case UnknownApiError(status_code=status, body=body):
                pytest.fail(f"expected a 422 judge rejection, got {status}: {body[:400]}")
            case _:
                pytest.fail(
                    "the judge returned an English response that cannot satisfy a French-only "
                    f"criterion; got {rejected}"
                )

        accepted = unwrap(
            chat(
                client,
                scoped_key,
                prompt=(
                    f"Cas {marker}. Reponds exactement ceci et rien d'autre: "
                    "Il fait beau aujourd'hui."
                ),
                guardrail_name=name,
            )
        )
        answer = response_text(accepted)
        assert "beau" in answer.lower(), (
            "the passing case must actually come back in French, otherwise the judge was never "
            f"given a response that satisfies the criterion; got: {answer[:200]!r}"
        )

        runs = poll_guardrail_usage_logs(client, guardrail_id, min_rows=2)
        approvals = [entry.action for entry in runs if entry.id == accepted.id]
        assert approvals == ["passed"], (
            "the guardrail must record the accepted completion as adjudicated and approved; a "
            "judge that errored would have let the same French response through and recorded "
            f"'flagged' instead. Runs for this guardrail: {runs}"
        )
        assert [entry.action for entry in runs if entry.id != accepted.id] == ["blocked"], (
            "the rejected completion must be recorded as a guardrail intervention, not as an "
            f"error the guardrail failed open on. Runs for this guardrail: {runs}"
        )
