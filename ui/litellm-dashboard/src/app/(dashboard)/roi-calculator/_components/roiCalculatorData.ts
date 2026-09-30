import type { components } from "@/lib/http/schema";

export type ROISummary = components["schemas"]["ROISummaryResponse"];
export type ROIPull = components["schemas"]["ROIPullResponse"];
export type ROIPerson = components["schemas"]["ROIPersonResponse"];
export type ROIEstimate = components["schemas"]["ROIEstimateResponse"];
export type ROISyncStatus = components["schemas"]["ROISyncStatus"];
export type ROISettings = components["schemas"]["ROISettingsResponse"];
export type ROISettingsUpdate = components["schemas"]["ROISettingsUpdate"];
export type ROIRepository = components["schemas"]["ROIRepository"];
export type ROIRepositoriesResponse = components["schemas"]["ROIRepositoriesResponse"];
export type ROIReportResponse = components["schemas"]["ROIReportResponse"];
export type ROIIdentityMapUpdate = components["schemas"]["ROIIdentityMapUpdate"];
export type ROIIdentityMapResponse = components["schemas"]["ROIIdentityMapResponse"];

const SYNCED_AT_FORMAT_OPTIONS: Intl.DateTimeFormatOptions = {
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
  timeZone: "UTC",
  timeZoneName: "short",
};

export const formatMoney = (value: number | null | undefined): string => {
  if (value == null) return "—";
  if (value > 0 && value < 0.01) return "<$0.01";
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(value);
};

export const formatNumber = (value: number | null | undefined): string =>
  value == null ? "—" : new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(value);

export const formatSyncedAt = (value: string): string => {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return value;
  return new Intl.DateTimeFormat("en-US", SYNCED_AT_FORMAT_OPTIONS).format(timestamp);
};

export const effortNote = (basis: string | null | undefined): string =>
  basis === "without_ai"
    ? "Estimated engineering hours without AI assistance, not actual hours worked or hours saved."
    : "Earlier estimates did not specify AI assistance. Sync to estimate engineering hours without AI.";

export const coverageLabel = (summary: {
  metrics: Pick<ROISummary["metrics"], "matched_prs" | "merged_prs">;
}): string => `${summary.metrics.matched_prs} of ${summary.metrics.merged_prs} PRs have email matches`;

export const estimateLabel = (estimate: ROIEstimate): string => {
  if (estimate.status === "estimated") return `${formatNumber(estimate.hours)} hrs`;
  if (estimate.status === "error") return "Estimate failed";
  return "Needs review";
};

export const filterPulls = (pulls: ROIPull[], query: string): ROIPull[] => {
  const normalized = query.trim().toLocaleLowerCase();
  if (!normalized) return pulls;
  return pulls.filter((pull) =>
    `${pull.title} ${pull.repo} ${pull.number} ${pull.login}`.toLocaleLowerCase().includes(normalized),
  );
};

export const peopleCsv = (summary: ROISummary): string => {
  const escape = (value: unknown): string => {
    const text = value == null ? "" : String(value);
    const safe = /^[=+@\-\t\r]/.test(text) ? `'${text}` : text;
    return `"${safe.replaceAll('"', '""')}"`;
  };
  const rows = summary.people.map((person) => [
    person.email,
    person.logins.join(";"),
    person.spend,
    person.hours,
    person.prs,
    person.pending_prs,
    person.eligible,
    person.cost_per_hour,
    summary.start,
    summary.end,
    summary.effort_basis ?? "unspecified",
  ]);
  return [
    [
      "email",
      "github_logins",
      "gateway_spend_usd",
      "estimated_hours",
      "merged_prs",
      "pending_estimates",
      "in_matched_cohort",
      "cost_per_estimated_hour",
      "start_utc",
      "end_utc",
      "effort_basis",
    ],
    ...rows,
  ]
    .map((row) => row.map(escape).join(","))
    .join("\r\n");
};
