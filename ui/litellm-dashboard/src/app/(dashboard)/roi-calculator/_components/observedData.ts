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
  requester: z.string().optional(),
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
export type ReportPerson = ObservedPerson & { id: string; matched: boolean; host: string };

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

export function visiblePeople<T extends ObservedPerson>(people: T[], query: string, sort: PeopleSort) {
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
  return report ? `Updated ${new Date(report.captured_at).toLocaleString()}` : "";
}

export function recordedBranches(snapshot: ObservedSnapshot, matchedOnly = false) {
  const matched = snapshot.pulls.current.flatMap((pull) => {
    const cost = pull.branch_cost;
    if (cost.status !== "matched" || cost.spend === null) return [];
    return [{ repo: cost.repo, branch: cost.branch, spend: cost.spend, requests: cost.requests }];
  });
  const rows = [
    ...new Map([...matched, ...snapshot.unlinked_branches].map((row) => [`${row.repo}\n${row.branch}`, row])).values(),
  ];
  const keys = new Set(
    filterObservedPulls(snapshot, "current", true).map(
      (pull) => `${pull.branch_cost.repo}\n${pull.branch_cost.branch}`,
    ),
  );
  return rows
    .filter((row) => !matchedOnly || keys.has(`${row.repo}\n${row.branch}`))
    .toSorted((a, b) => b.spend - a.spend);
}

export function filterObservedPulls(snapshot: ObservedSnapshot, period: Period, matchedOnly: boolean) {
  if (!matchedOnly) return snapshot.pulls[period];
  const urls = new Set(snapshot.people.flatMap((person) => person.periods[period].pr_urls));
  return snapshot.pulls[period].filter((pull) => urls.has(pull.url));
}

function unmatchedPeriod(
  pulls: ObservedPull[],
  window: { start: string; end: string },
): ObservedPerson["periods"]["current"] {
  const hours = pulls
    .flatMap((pull) => (pull.merge_hours === null ? [] : [pull.merge_hours]))
    .toSorted((a, b) => a - b);
  const middle = Math.floor(hours.length / 2);
  const median = hours.length % 2 ? hours[middle] : (hours[middle - 1] + hours[middle]) / 2;
  const days = (Date.parse(window.end) - Date.parse(window.start)) / 86_400_000 + 1;
  return {
    merged_prs: pulls.length,
    prs_per_week: (pulls.length * 7) / days,
    median_merge_hours: hours.length ? median : null,
    direct_authored: pulls.filter((pull) => !pull.agent).length,
    declared_agent_owned: pulls.filter((pull) => pull.agent).length,
    gateway_recorded_spend: 0,
    recorded_spend_per_attributed_pr: null,
    spend_observation: "no_records",
    pr_urls: pulls.map((pull) => pull.url),
  };
}

const ownerLogin = (pull: ObservedPull) => (pull.agent ? pull.requester ?? "" : pull.author);
const accountKey = (pull: ObservedPull) =>
  `${pull.connection_id ?? new URL(pull.url).origin}\n${ownerLogin(pull).toLowerCase()}`;

export function reportPeople(snapshot: ObservedSnapshot, matchedOnly: boolean): ReportPerson[] {
  const matched = snapshot.people.map((person) => ({ ...person, id: person.email, matched: true, host: "" }));
  if (matchedOnly) return matched;
  const unmatchedGroups = (period: Period) => {
    const urls = new Set(snapshot.people.flatMap((person) => person.periods[period].pr_urls));
    const pulls = snapshot.pulls[period].filter((pull) => !urls.has(pull.url) && ownerLogin(pull));
    const sorted = pulls
      .map((pull) => ({ key: accountKey(pull), pull }))
      .toSorted((a, b) => a.key.localeCompare(b.key));
    const starts = sorted.flatMap((entry, index) =>
      index === 0 || entry.key !== sorted[index - 1].key ? [index] : [],
    );
    return new Map(
      starts.map((start, index) => [
        sorted[start].key,
        sorted.slice(start, starts[index + 1]).map((entry) => entry.pull),
      ]),
    );
  };
  const groups = {
    current: unmatchedGroups("current"),
    previous: unmatchedGroups("previous"),
    last_year: unmatchedGroups("last_year"),
  };
  const accounts = new Map([...groups.current, ...groups.previous, ...groups.last_year]);
  const unmatched = [...accounts].map(([id, pulls]) => {
    const pull = pulls[0];
    const login = ownerLogin(pull);
    const period = (key: Period) => unmatchedPeriod(groups[key].get(id) ?? [], snapshot.periods[key].window);
    return {
      id,
      matched: false,
      host: new URL(pull.url).host,
      name: login,
      email: "",
      logins: [login],
      accounts: [{ connection_id: pull.connection_id ?? "", login }],
      periods: { current: period("current"), previous: period("previous"), last_year: period("last_year") },
    };
  });
  return [...matched, ...unmatched];
}

export function changeTerms(provider: ObservedSnapshot["source_provider"]) {
  if (provider === "mixed")
    return { singular: "change", plural: "changes", requests: "Merged changes", lower: "merged changes" };
  return provider === "gitlab"
    ? { singular: "MR", plural: "MRs", requests: "Merge requests", lower: "merge requests" }
    : { singular: "PR", plural: "PRs", requests: "Pull requests", lower: "pull requests" };
}
