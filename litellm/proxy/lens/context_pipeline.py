from typing import Final

from .agent_review import review_context
from .agent_workspace import load_workspace
from .analysis import Examined, ModelCall, ReadContent, ReportProgress, analyze_executions, analyze_with
from .models import Claim, Execution, Result, Sample


async def analyze_sample(
    claim: Claim, sample: Sample, read: ReadContent, model: ModelCall, progress: ReportProgress
) -> Result:
    return await analyze_with(claim, sample, read, model, progress, analyze_context)


async def analyze_context(
    claim: Claim, sample: Sample, read: ReadContent, model: ModelCall, progress: ReportProgress
) -> Result:
    workspace: Final = await load_workspace(sample, read, claim.job.settings.concurrency)

    async def extract(claim: Claim, execution: Execution, _read: ReadContent, model: ModelCall) -> Examined:
        session: Final = next(session for session in workspace.sessions if session.execution.id == execution.id)
        return await review_context(claim, session, workspace, model)

    return await analyze_executions(claim, sample, read, model, progress, extractor=extract)
