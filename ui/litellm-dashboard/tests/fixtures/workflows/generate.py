import json
from pathlib import Path
from typing import Final

from liteflow.authoring import ChatCompletion, ChatMessage, Input, Python, Retry, Workflow
from liteflow.client import Draft
from liteflow.compiler import bundle_digest, compile_workflow
from pydantic import BaseModel, ConfigDict


def prepare_followup(text: str, instruction: str) -> list[ChatMessage]:
    return [{"role": "user", "content": instruction + "\n\n" + text}]


class Review(BaseModel):
    model_config = ConfigDict(frozen=True)
    summary: str
    approved: bool


def draft_fixture(workflow: Workflow) -> Draft:
    bundle: Final = compile_workflow(workflow)
    digest: Final = bundle_digest(bundle)
    return Draft(
        name=workflow.name,
        workflow_id=f"fixture-{workflow.name}",
        team_id="fixture-team",
        etag=f'"{digest}"',
        digest=digest,
        bundle=bundle,
    )


SEQUENTIAL: Final = Workflow(
    name="email-reply",
    nodes=(
        ChatCompletion(model="support-model", messages=Input.messages, temperature=0.4),
        Python(
            fn=prepare_followup,
            inputs={"text": Input.previous, "instruction": "Review this draft for clarity and accuracy"},
            timeout_seconds=15,
            retry=Retry(max_attempts=2, backoff_seconds=0.5),
        ),
        ChatCompletion(model="review-model", messages=Input.previous, response_model=Review, max_completion_tokens=512),
    ),
)

LITERAL: Final = Workflow(
    name="welcome-message",
    nodes=(
        ChatCompletion(
            model="support-model",
            messages=({"role": "user", "content": "Write a short welcome message"},),
        ),
    ),
)

FIXTURES: Final = {workflow.name: draft_fixture(workflow).model_dump(mode="json") for workflow in (SEQUENTIAL, LITERAL)}

Path(__file__).with_name("drafts.json").write_text(json.dumps(FIXTURES, indent=2) + "\n")
