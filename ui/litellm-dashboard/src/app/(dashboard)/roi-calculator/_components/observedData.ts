import { z } from "zod";

const windowSchema = z.object({ start: z.string(), end: z.string() });
const personPeriodFields = {
  merged_prs: z.number(),
  prs_per_week: z.number(),
  median_merge_hours: z.number().nullable(),
  direct_authored: z.number(),
  declared_agent_owned: z.number(),
  gateway_recorded_spend: z.number(),
  recorded_spend_per_attributed_pr: z.number().nullable(),
  spend_observation: z.enum(["records_present", "no_records"]),
  pr_urls: z.array(z.string()),
};
const personPeriodSchema = z.object(personPeriodFields);
const periodFields = {
  window: windowSchema,
  merged_prs: z.number(),
  median_merge_hours: z.number().nullable(),
  human_authored: z.number(),
  agent_authored: z.number(),
  missing_author: z.number(),
  agents_without_requester: z.number(),
  matched_internal_prs: z.number(),
  new_bug_labeled_issues: z.number().nullable(),
  new_regression_labeled_issues: z.number().nullable(),
  explicitly_titled_revert_prs: z.number(),
  matched_users_recorded_spend: z.number(),
  spend_observation: z.enum(["records_present", "no_records"]),
  human_summary: z.object({ median_merge_hours: z.number().nullable() }),
};
const periodSchema = z.object(periodFields);
const periods = <T extends z.ZodType>(schema: T) => z.object({ current: schema, previous: schema, last_year: schema });

const personFields = {
  name: z.string(),
  email: z.string(),
  logins: z.array(z.string()),
  periods: periods(personPeriodSchema),
  accounts: z.array(z.object({ connection_id: z.string(), login: z.string() })).optional(),
};
const branchCostFields = {
  repo: z.string(),
  branch: z.string(),
  spend: z.number().nullable(),
  requests: z.number(),
  status: z.enum(["matched", "unattributed", "ambiguous", "unavailable"]),
};
const branchSpendFields = { repo: z.string(), branch: z.string(), spend: z.number(), requests: z.number() };
const pullFields = {
  connection_id: z.string().optional(),
  number: z.number(),
  title: z.string(),
  url: z
    .string()
    .url()
    .refine((url) => new URL(url).protocol === "https:"),
  author: z.string(),
  agent: z.boolean(),
  merged_at: z.string(),
  merge_hours: z.number().nullable(),
  repo: z.string(),
  source_repo: z.string(),
  source_branch: z.string(),
  branch_cost: z.object(branchCostFields),
};
const snapshotFields = {
  source_provider: z.enum(["github", "gitlab", "mixed"]),
  repos: z.array(z.string()),
  unmatched_logins: z.array(z.string()),
  unlinked_branches: z.array(z.object(branchSpendFields)),
  captured_at: z.string(),
  periods: periods(periodSchema),
  people: z.array(z.object(personFields)),
  pulls: periods(z.array(z.object(pullFields))),
};
export const observedSnapshotSchema = z.object(snapshotFields);
const settingsFields = {
  id: z.string().optional(),
  source_provider: z.enum(["github", "gitlab"]),
  api_url: z.string(),
  repos: z.array(z.string()),
  has_token: z.boolean(),
  connection_type: z.enum(["token", "app"]),
  update_interval_minutes: z.number(),
  ready: z.boolean(),
};
export const observedConnectionSchema = z.object(settingsFields);
export const observedSettingsSchema = z.object({
  ...settingsFields,
  connections: z.array(observedConnectionSchema).optional(),
});
export type ObservedConnection = z.infer<typeof observedConnectionSchema>;
const statusFields = {
  running: z.boolean(),
  phase: z.string(),
  stage: z.string(),
  done: z.number(),
  total: z.number(),
  error: z.string().nullable(),
  finished_at: z.string().nullable().optional(),
};
export const observedStatusSchema = z.object(statusFields);
export const observedReportResponseSchema = z.object({ report: observedSnapshotSchema.nullable() });
export type ObservedSettings = z.infer<typeof observedSettingsSchema>;
export type ObservedStatus = z.infer<typeof observedStatusSchema>;

export type ObservedSnapshot = z.infer<typeof observedSnapshotSchema>;
export type ObservedPerson = ObservedSnapshot["people"][number];
export type ObservedPull = ObservedSnapshot["pulls"]["current"][number];
export type Period = keyof ObservedSnapshot["periods"];
export type Comparison = Exclude<Period, "current">;
export type PeopleSort = "merged" | "spend" | "cost" | "name";

export const number = (value: number | null) =>
  value === null ? "Unavailable" : value.toLocaleString("en-US", { maximumFractionDigits: 1 });
export function money(value: number | null) {
  if (value === null) return "Unavailable";
  if (value > 0 && value < 0.01) return "<$0.01";
  return value.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });
}
export function duration(value: number | null) {
  if (value === null) return "Unavailable";
  if (value === 0) return "0m";
  if (value < 1 / 60) return "<1m";
  if (value < 1) return `${number(value * 60)}m`;
  return `${number(value)}h`;
}
export const change = (current: number, baseline: number) =>
  baseline === 0 ? null : ((current - baseline) / baseline) * 100;

export const accountLogins = (value: string) => [
  ...new Set(
    value
      .split(/[\s,]+/)
      .map((login) => login.replace(/^@/, "").toLowerCase())
      .filter(Boolean),
  ),
];
export const repositoryNames = (value: string) => [
  ...new Set(
    value
      .split(/[\s,]+/)
      .filter(Boolean)
      .map((repo) => {
        const path = repo.replace(/^https:\/\/[^/]+\//, "");
        return path.replace(/\/$/, "").replace(/\.git$/, "");
      }),
  ),
];

export function dateRange(window: { start: string; end: string }) {
  const date = (value: string) =>
    new Date(`${value}T00:00:00Z`).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      timeZone: "UTC",
    });
  return `${date(window.start)} – ${date(window.end)}, ${window.end.slice(0, 4)}`;
}

export function visiblePeople(people: ObservedPerson[], query: string, sort: PeopleSort) {
  const value = (person: ObservedPerson) => {
    const current = person.periods.current;
    if (sort === "spend") return current.spend_observation === "no_records" ? -1 : current.gateway_recorded_spend;
    if (sort === "cost") return current.recorded_spend_per_attributed_pr ?? -1;
    return current.merged_prs;
  };
  return people
    .filter((person) =>
      [person.name, person.email, ...person.logins].join(" ").toLowerCase().includes(query.trim().toLowerCase()),
    )
    .toSorted((a, b) => (sort === "name" ? a.name.localeCompare(b.name) : value(b) - value(a)));
}

export function weeklyMerges(snapshot: ObservedSnapshot, period: Period) {
  const start = Date.parse(`${snapshot.periods[period].window.start}T00:00:00Z`);
  const end = Date.parse(`${snapshot.periods[period].window.end}T00:00:00Z`) + 86_400_000;
  const days = (end - start) / 86_400_000;
  return Array.from(
    { length: Math.ceil(days / 7) },
    (_, week) =>
      snapshot.pulls[period].filter((pull) => {
        const day = (Date.parse(pull.merged_at) - start) / 86_400_000;
        return day >= week * 7 && day < Math.min((week + 1) * 7, days);
      }).length,
  );
}

export function syncMessage(status: ObservedStatus, report: ObservedSnapshot | null) {
  if (status.running) return status.total ? `${status.stage} · ${status.done} / ${status.total}` : status.stage;
  return report ? `Updated ${new Date(report.captured_at).toLocaleString()}` : "Ready to sync";
}

export function recordedBranches(snapshot: ObservedSnapshot) {
  const matched = snapshot.pulls.current.flatMap((pull) => {
    const cost = pull.branch_cost;
    if (cost.status !== "matched" || cost.spend === null) return [];
    return [{ repo: cost.repo, branch: cost.branch, spend: cost.spend, requests: cost.requests }];
  });
  return [
    ...new Map([...matched, ...snapshot.unlinked_branches].map((row) => [`${row.repo}\n${row.branch}`, row])).values(),
  ].toSorted((a, b) => b.spend - a.spend);
}

export function changeTerms(provider: ObservedSnapshot["source_provider"]) {
  if (provider === "mixed")
    return { singular: "change", plural: "changes", requests: "Merged changes", lower: "merged changes" };
  return provider === "gitlab"
    ? { singular: "MR", plural: "MRs", requests: "Merge requests", lower: "merge requests" }
    : { singular: "PR", plural: "PRs", requests: "Pull requests", lower: "pull requests" };
}
