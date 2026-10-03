import { describe, expect, it } from "vitest";
import { createLensDemoData } from "../demo/createLensDemo";
import { problemsOverview } from "./problems";
import type { Finding, Job, Lens } from "./types";

type Execution = NonNullable<Job["sample"]>["executions"][number];

const lenses = () => createLensDemoData().lenses;

function withFinding(lens: Lens, change: (finding: Finding) => Finding): Lens {
  const [latest, ...rest] = lens.jobs;
  return { ...lens, jobs: [{ ...latest, findings: (latest.findings ?? []).map(change) }, ...rest] };
}

function labelAgents(lens: Lens): Lens {
  const label = (execution: Execution): Execution => ({
    ...execution,
    name: "invoke_agent",
    metadata: [{ key: "gen_ai.agent.name", value: `${lens.id}-bot` }],
  });
  const relabel = (job: Job): Job => ({
    ...job,
    sample: job.sample && { ...job.sample, executions: job.sample.executions.map(label) },
  });
  return { ...lens, settings: { ...lens.settings, agent_name: "" }, jobs: lens.jobs.map(relabel) };
}

describe("problemsOverview", () => {
  it("lists only open issues from each investigation's latest completed run", () => {
    const { problems } = problemsOverview(lenses());
    expect(problems.map((p) => p.title).sort()).toEqual([
      "Performance claim has no supporting benchmark",
      "Repeated lookups leave customers without an answer",
    ]);
  });

  it("names the agent and counts hits against that agent's reviewed runs", () => {
    const lookup = problemsOverview(lenses()).problems.find((p) => p.findingId === "failed-lookups");
    expect(lookup?.agent).toBe("support_agent");
    expect(lookup?.hits).toBe(2);
    expect(lookup?.runs.length).toBe(5);
    expect(lookup?.runs.filter((run) => run.hit).every((run) => run.quote === "I will check that for you.")).toBe(true);
    expect(lookup?.jobId).toBe("support-scan-0");
  });

  it("orders frequency dots oldest to newest", () => {
    const runs = problemsOverview(lenses()).problems[0].runs.map((run) => Date.parse(run.startTime));
    expect(runs).toEqual([...runs].sort((a, b) => a - b));
  });

  it("sorts high severity first, then by how often it happened", () => {
    const data = lenses().map((lens) =>
      lens.id === "support" ? withFinding(lens, (f) => ({ ...f, priority: "low" as const })) : lens,
    );
    expect(problemsOverview(data).problems.map((p) => p.findingId)).toEqual(["unsupported-claim", "failed-lookups"]);
    expect(problemsOverview(lenses()).problems.map((p) => p.findingId)).toEqual([
      "failed-lookups",
      "unsupported-claim",
    ]);
  });

  it("drops dismissed and resolved problems", () => {
    const data = lenses().map((lens) => withFinding(lens, (f) => ({ ...f, status: "dismissed" as const })));
    expect(problemsOverview(data).problems).toEqual([]);
  });

  it("summarizes affected runs per agent for the filter", () => {
    const overview = problemsOverview(lenses());
    expect(overview.agents.find((a) => a.agent === "support_agent")).toEqual({
      agent: "support_agent",
      runs: 5,
      affected: 2,
    });
    expect(overview.agents.find((a) => a.agent === "release_agent")?.affected).toBe(0);
    expect(overview.runs).toBe(overview.agents.reduce((sum, a) => sum + a.runs, 0));
    expect(overview.affected).toBe(3);
  });

  it("ignores investigations that have not finished a run", () => {
    const data = lenses().map((lens) => ({
      ...lens,
      jobs: lens.jobs.map((job) => ({ ...job, status: "running" as const })),
    }));
    const empty = { problems: [], agents: [], runs: 0, affected: 0 };
    expect(problemsOverview(data)).toEqual(empty);
  });

  it("names the agent from the run's agent attribute before the run label", () => {
    const data = lenses().map(labelAgents);
    expect(problemsOverview(data).problems.map((p) => p.agent)).toEqual(["support-bot", "research-bot"]);
  });
});
