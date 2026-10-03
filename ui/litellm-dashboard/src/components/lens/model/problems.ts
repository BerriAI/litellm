import type { Finding, Job, Lens } from "./types";

type Execution = NonNullable<Job["sample"]>["executions"][number];

export interface ProblemRun {
  readonly executionId: string;
  readonly hit: boolean;
  readonly startTime: string;
  readonly quote: string;
}

export interface Problem {
  readonly lensId: string;
  readonly jobId: string;
  readonly findingId: string;
  readonly title: string;
  readonly priority: Finding["priority"];
  readonly quote: string;
  readonly agent: string;
  readonly runs: readonly ProblemRun[];
  readonly hits: number;
}

export interface AgentSummary {
  readonly agent: string;
  readonly runs: number;
  readonly affected: number;
}

export interface ProblemsOverview {
  readonly problems: readonly Problem[];
  readonly agents: readonly AgentSummary[];
  readonly runs: number;
  readonly affected: number;
}

const rank = { high: 0, medium: 1, low: 2 } as const;

const agentOf = (execution: Execution, lens: Lens): string => {
  const candidates = [
    execution.metadata.find((m) => m.key === "gen_ai.agent.name")?.value,
    lens.settings.agent_name,
    execution.name,
  ];
  return candidates.find(Boolean) ?? "unknown";
};

function latestCompleted(lens: Lens): Job | undefined {
  return lens.jobs.find((job) => job.status === "completed" && job.findings != null);
}

function problemAgent(finding: Finding, executions: readonly Execution[], lens: Lens): string {
  const hit = executions.find((execution) => finding.occurrences.includes(execution.id));
  return hit ? agentOf(hit, lens) : lens.settings.agent_name || "unknown";
}

function problemsFor(lens: Lens, job: Job): Problem[] {
  const executions = job.sample?.executions ?? [];
  return (job.findings ?? [])
    .filter((finding) => finding.kind === "issue" && finding.status === "open")
    .map((finding) => {
      const agent = problemAgent(finding, executions, lens);
      const quotes = new Map(finding.evidence.map((e) => [e.execution_id, e.quote]));
      const runs = executions
        .filter((execution) => agentOf(execution, lens) === agent)
        .map((execution) => ({
          executionId: execution.id,
          hit: finding.occurrences.includes(execution.id),
          startTime: execution.start_time,
          quote: quotes.get(execution.id) ?? "",
        }))
        .sort((a, b) => Date.parse(a.startTime) - Date.parse(b.startTime));
      return {
        lensId: lens.id,
        jobId: job.id,
        findingId: finding.id,
        title: finding.title,
        priority: finding.priority,
        quote: finding.evidence.find((e) => e.role === "support")?.quote ?? finding.description,
        agent,
        runs,
        hits: runs.filter((run) => run.hit).length || finding.occurrences.length,
      };
    });
}

export function problemsOverview(lenses: readonly Lens[]): ProblemsOverview {
  const scanned = lenses.flatMap((lens) => {
    const job = latestCompleted(lens);
    return job ? [{ lens, job }] : [];
  });
  const problems = scanned
    .flatMap(({ lens, job }) => problemsFor(lens, job))
    .sort((a, b) => rank[a.priority] - rank[b.priority] || b.hits - a.hits);
  const affectedIds = new Set(problems.flatMap((p) => p.runs.filter((run) => run.hit).map((run) => run.executionId)));
  const runAgents = new Map(
    scanned.flatMap(({ lens, job }) =>
      (job.sample?.executions ?? []).map((execution) => [execution.id, agentOf(execution, lens)] as const),
    ),
  );
  const agents = [...new Set(runAgents.values())]
    .map((agent) => {
      const ids = [...runAgents].filter(([, name]) => name === agent).map(([id]) => id);
      return { agent, runs: ids.length, affected: ids.filter((id) => affectedIds.has(id)).length };
    })
    .sort((a, b) => b.affected - a.affected || a.agent.localeCompare(b.agent));
  return { problems, agents, runs: runAgents.size, affected: affectedIds.size };
}
