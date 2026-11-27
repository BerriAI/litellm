import type { TFunction } from "i18next";
import { z } from "zod";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import type { EntityType, ExportType } from "./types";

export const USAGE_REPORT_MAX_ROWS = 1000;

const usageReportRowShape = {
  date: z.string(),
  entity_id: z.string(),
  entity_alias: z.string().nullish(),
  api_key: z.string().nullish(),
  key_alias: z.string().nullish(),
  user_id: z.string().nullish(),
  user_email: z.string().nullish(),
  model: z.string().nullish(),
  spend: z.number(),
  prompt_tokens: z.number(),
  completion_tokens: z.number(),
  api_requests: z.number(),
  successful_requests: z.number(),
  failed_requests: z.number(),
};

const usageReportRowSchema = z.object(usageReportRowShape);

export type UsageReportRow = z.infer<typeof usageReportRowSchema>;

export const parseUsageReportRows = async (blob: Blob): Promise<readonly UsageReportRow[]> =>
  z.array(usageReportRowSchema).parse(JSON.parse(await blob.text()));

interface UsageReportInput {
  title: string;
  heading: string;
  entityType: EntityType;
  exportType: ExportType;
  rows: readonly UsageReportRow[];
  dateRange: DateRangePickerValue;
  selectedFilters: readonly string[];
  generatedAt: Date;
  t: TFunction;
}

const HTML_ESCAPES: Readonly<Record<string, string>> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

export const escapeHtml = (value: string): string => value.replace(/[&<>"']/g, (char) => HTML_ESCAPES[char]);

const formatSpend = (value: number): string =>
  `$${formatNumberWithCommas(value, value !== 0 && Math.abs(value) < 0.01 ? 6 : 2)}`;

const formatCount = (value: number): string => formatNumberWithCommas(value, 0);

const breakdownColumn = (
  exportType: ExportType,
  t: TFunction,
): { label: string; value: (row: UsageReportRow) => string } | null => {
  switch (exportType) {
    case "daily_with_keys":
      return { label: t("usage.export.report.columns.key"), value: (row) => row.key_alias ?? row.api_key ?? "-" };
    case "daily_with_models":
      return { label: t("usage.export.report.columns.model"), value: (row) => row.model ?? "-" };
    case "daily_with_users":
      return { label: t("usage.export.report.columns.user"), value: (row) => row.user_email ?? row.user_id ?? "-" };
    case "daily":
      return null;
  }
};

const sum = (rows: readonly UsageReportRow[], pick: (row: UsageReportRow) => number): number =>
  rows.reduce((total, row) => total + pick(row), 0);

export const buildUsageReportHtml = ({
  title,
  heading,
  entityType,
  exportType,
  rows,
  dateRange,
  selectedFilters,
  generatedAt,
  t,
}: UsageReportInput): string => {
  const breakdown = breakdownColumn(exportType, t);
  const shownRows = rows.slice(0, USAGE_REPORT_MAX_ROWS);
  const promptTokens = sum(rows, (row) => row.prompt_tokens);
  const completionTokens = sum(rows, (row) => row.completion_tokens);
  const summaryItems: readonly (readonly [string, string])[] = [
    [t("usage.export.report.summary.spend"), formatSpend(sum(rows, (row) => row.spend))],
    [t("usage.export.report.summary.requests"), formatCount(sum(rows, (row) => row.api_requests))],
    [t("usage.export.report.summary.successful"), formatCount(sum(rows, (row) => row.successful_requests))],
    [t("usage.export.report.summary.failed"), formatCount(sum(rows, (row) => row.failed_requests))],
    [t("usage.export.report.summary.tokens"), formatCount(promptTokens + completionTokens)],
    [t("usage.export.report.summary.promptTokens"), formatCount(promptTokens)],
    [t("usage.export.report.summary.completionTokens"), formatCount(completionTokens)],
  ];
  const headers = [
    t("usage.export.report.columns.date"),
    t(`usage.entityLabels.title.${entityType}`),
    ...(breakdown ? [breakdown.label] : []),
    t("usage.export.report.columns.spend"),
    t("usage.export.report.columns.requests"),
    t("usage.export.report.columns.successful"),
    t("usage.export.report.columns.failed"),
    t("usage.export.report.columns.promptTokens"),
    t("usage.export.report.columns.completionTokens"),
  ];
  const cells = (row: UsageReportRow): readonly string[] => [
    row.date,
    row.entity_alias ?? row.entity_id,
    ...(breakdown ? [breakdown.value(row)] : []),
    formatSpend(row.spend),
    formatCount(row.api_requests),
    formatCount(row.successful_requests),
    formatCount(row.failed_requests),
    formatCount(row.prompt_tokens),
    formatCount(row.completion_tokens),
  ];
  const range = `${dateRange.from?.toLocaleDateString() ?? "-"} - ${dateRange.to?.toLocaleDateString() ?? "-"}`;
  const meta: readonly (readonly [string, string])[] = [
    [t("usage.export.report.meta.dateRange"), range],
    [t("usage.export.report.meta.exportType"), exportType],
    ...(selectedFilters.length > 0
      ? [[t("usage.export.report.meta.filters"), selectedFilters.join(", ")] as const]
      : []),
    [t("usage.export.report.meta.generated"), generatedAt.toLocaleString()],
  ];

  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>${escapeHtml(title)}</title>
<style>
  @page { size: landscape; margin: 12mm; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #1f2937; margin: 0; padding: 24px; }
  h1 { font-size: 20px; margin: 0 0 8px; }
  h2 { font-size: 15px; margin: 24px 0 8px; }
  .meta { font-size: 12px; color: #4b5563; margin: 0 0 16px; }
  .meta div { margin: 2px 0; }
  .summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
  .summary div { border: 1px solid #e5e7eb; border-radius: 6px; padding: 8px 10px; }
  .summary .label { font-size: 11px; color: #6b7280; }
  .summary .value { font-size: 16px; font-weight: 600; font-family: ui-monospace, monospace; }
  table { width: 100%; border-collapse: collapse; font-size: 11px; }
  th, td { text-align: left; padding: 4px 6px; border-bottom: 1px solid #e5e7eb; overflow-wrap: anywhere; }
  th { background: #f3f4f6; }
  thead { display: table-header-group; }
  tr { break-inside: avoid; }
  .note { font-size: 11px; color: #92400e; margin: 0 0 8px; }
</style>
</head>
<body>
<h1>${escapeHtml(heading)}</h1>
<div class="meta">${meta.map(([label, value]) => `<div><strong>${escapeHtml(label)}:</strong> ${escapeHtml(value)}</div>`).join("")}</div>
<h2>${escapeHtml(t("usage.export.report.summary.title"))}</h2>
<div class="summary" data-testid="usage-report-summary">${summaryItems
    .map(
      ([label, value]) =>
        `<div><div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(value)}</div></div>`,
    )
    .join("")}</div>
<h2>${escapeHtml(t("usage.export.report.breakdown"))}</h2>
${
  rows.length > shownRows.length
    ? `<p class="note">${escapeHtml(
        t("usage.export.report.truncated", {
          shown: formatCount(shownRows.length),
          total: formatCount(rows.length),
        }),
      )}</p>`
    : ""
}
<table>
<thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead>
<tbody>${
    shownRows.length > 0
      ? shownRows
          .map(
            (row) =>
              `<tr>${cells(row)
                .map((cell) => `<td>${escapeHtml(cell)}</td>`)
                .join("")}</tr>`,
          )
          .join("")
      : `<tr><td colspan="${headers.length}">${escapeHtml(t("usage.export.report.empty"))}</td></tr>`
  }</tbody>
</table>
</body>
</html>`;
};

export const printUsageReport = (html: string, frameTitle: string, fileName: string): void => {
  document.querySelector(`iframe[data-usage-report]`)?.remove();
  const frame = document.createElement("iframe");
  frame.title = frameTitle;
  frame.setAttribute("data-usage-report", "");
  frame.setAttribute("aria-hidden", "true");
  frame.setAttribute("sandbox", "allow-same-origin allow-modals");
  frame.style.position = "fixed";
  frame.style.width = "0";
  frame.style.height = "0";
  frame.style.border = "0";
  frame.style.visibility = "hidden";
  document.body.appendChild(frame);
  const frameWindow = frame.contentWindow;
  if (!frameWindow) throw new Error("Could not open the usage report for printing");
  frameWindow.document.open();
  frameWindow.document.write(html);
  frameWindow.document.close();
  frameWindow.focus();
  const pageTitle = document.title;
  document.title = fileName;
  try {
    frameWindow.print();
  } finally {
    document.title = pageTitle;
  }
};
