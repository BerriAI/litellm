import type { ParseKeys } from "i18next";
import i18n from "@/i18n/i18n";

export interface RequestErrorDailyEntry {
  date: string;
  successful_requests: number;
  failed_requests: number;
  client_errors: number;
  server_errors: number;
  by_status_code: { status_code: number; failed_requests: number }[];
}

export interface RequestErrorEntityEntry {
  id: string;
  label: string | null;
  api_requests: number;
  failed_requests: number;
  top_status_code: number | null;
  top_status_code_requests: number;
}

export interface RequestErrorActivity {
  total_successful_requests: number;
  total_failed_requests: number;
  by_date: RequestErrorDailyEntry[];
  by_status_code: { status_code: number; failed_requests: number }[];
  by_key: RequestErrorEntityEntry[];
  by_team: RequestErrorEntityEntry[];
  by_user: RequestErrorEntityEntry[];
  by_model: RequestErrorEntityEntry[];
}

export const ERROR_ENTITY_KINDS = ["key", "team", "user", "model"] as const;
export type ErrorEntityKind = (typeof ERROR_ENTITY_KINDS)[number];

export const ERROR_ENTITY_LABEL: Record<ErrorEntityKind, { singular: ParseKeys; plural: ParseKeys }> = {
  key: { singular: "usage.errors.entityLabels.key.singular", plural: "usage.errors.entityLabels.key.plural" },
  team: { singular: "usage.errors.entityLabels.team.singular", plural: "usage.errors.entityLabels.team.plural" },
  user: { singular: "usage.errors.entityLabels.user.singular", plural: "usage.errors.entityLabels.user.plural" },
  model: { singular: "usage.errors.entityLabels.model.singular", plural: "usage.errors.entityLabels.model.plural" },
};

const STATUS_LABELS: Record<number, ParseKeys> = {
  400: "usage.errors.statusLabels.badRequest",
  401: "usage.errors.statusLabels.unauthorized",
  402: "usage.errors.statusLabels.paymentRequired",
  403: "usage.errors.statusLabels.forbidden",
  404: "usage.errors.statusLabels.notFound",
  408: "usage.errors.statusLabels.requestTimeout",
  413: "usage.errors.statusLabels.payloadTooLarge",
  422: "usage.errors.statusLabels.unprocessableEntity",
  429: "usage.errors.statusLabels.rateLimited",
  500: "usage.errors.statusLabels.internalServerError",
  502: "usage.errors.statusLabels.badGateway",
  503: "usage.errors.statusLabels.serviceUnavailable",
  504: "usage.errors.statusLabels.gatewayTimeout",
  529: "usage.errors.statusLabels.overloaded",
};

export const statusCodeLabel = (code: number): string => {
  if (code === 0) return i18n.t("usage.errors.statusLabels.notRecorded");
  const known = STATUS_LABELS[code];
  if (known && Object.hasOwn(STATUS_LABELS, code)) return i18n.t(known);
  if (code >= 500) return i18n.t("usage.errors.statusLabels.serverError");
  if (code >= 400) return i18n.t("usage.errors.statusLabels.clientError");
  return i18n.t("usage.errors.statusLabels.error");
};

export const errorRate = (failed: number, total: number): number | null => (total > 0 ? (failed / total) * 100 : null);

export const formatRate = (rate: number | null): string => {
  if (rate === null) return "—";
  if (rate > 0 && rate < 0.1) return "<0.1%";
  return `${rate.toFixed(1)}%`;
};

export interface ErrorSummary {
  requests: number;
  failed: number;
  rate: number | null;
  clientErrors: number;
  serverErrors: number;
  rateLimited: number;
  authFailures: number;
}

export const errorSummary = (activity: RequestErrorActivity): ErrorSummary => {
  const requests = activity.total_successful_requests + activity.total_failed_requests;
  const byCode = (predicate: (code: number) => boolean) =>
    activity.by_status_code
      .filter((row) => predicate(row.status_code))
      .reduce((sum, row) => sum + row.failed_requests, 0);
  return {
    requests,
    failed: activity.total_failed_requests,
    rate: errorRate(activity.total_failed_requests, requests),
    clientErrors: byCode((code) => code >= 400 && code <= 499),
    serverErrors: byCode((code) => code >= 500 && code <= 599),
    rateLimited: byCode((code) => code === 429),
    authFailures: byCode((code) => code === 401 || code === 403),
  };
};

export interface ErrorSeriesRow extends Record<string, unknown> {
  date: string;
  requests: number;
  failed: number;
  rate: number;
}

export interface ErrorSeries {
  rows: ErrorSeriesRow[];
  codes: string[];
}

export const ERROR_SERIES_MAX_CODES = 6;
export const ERROR_SERIES_OTHER: ParseKeys = "usage.errors.series.other";

const seriesKey = (code: number): string => (code === 0 ? i18n.t("usage.errors.series.noStatus") : String(code));

export const errorSeries = (activity: RequestErrorActivity): ErrorSeries => {
  const ranked = [...activity.by_status_code]
    .filter((row) => row.failed_requests > 0)
    .sort((a, b) => b.failed_requests - a.failed_requests || a.status_code - b.status_code);
  const shown = ranked.slice(0, ERROR_SERIES_MAX_CODES).map((row) => row.status_code);
  const codes = shown.map(seriesKey);
  const needsOther = ranked.length > shown.length || activity.by_date.some(hasUnrecordedFailures);
  const rows = [...activity.by_date]
    .sort((a, b) => a.date.localeCompare(b.date))
    .map((day) => {
      const requests = day.successful_requests + day.failed_requests;
      const counts: Record<string, number> = Object.fromEntries(
        shown.map((code) => [
          seriesKey(code),
          day.by_status_code.find((entry) => entry.status_code === code)?.failed_requests ?? 0,
        ]),
      );
      const recorded = Object.values(counts).reduce((sum, count) => sum + count, 0);
      const other: Record<string, number> = needsOther
        ? { [i18n.t(ERROR_SERIES_OTHER)]: Math.max(0, day.failed_requests - recorded) }
        : {};
      return {
        date: day.date,
        requests,
        failed: day.failed_requests,
        rate: errorRate(day.failed_requests, requests) ?? 0,
        ...counts,
        ...other,
      };
    });
  return { rows, codes: needsOther ? [...codes, i18n.t(ERROR_SERIES_OTHER)] : codes };
};

export type ErrorRateView = "total" | "status";

export const ERROR_RATE_VIEW_OPTIONS: readonly { value: ErrorRateView; labelKey: ParseKeys }[] = [
  { value: "total", labelKey: "usage.errors.rateViews.total" },
  { value: "status", labelKey: "usage.errors.rateViews.status" },
];

export const rateByStatusRows = (series: ErrorSeries): ErrorSeriesRow[] =>
  series.rows.map((row) => ({
    date: row.date,
    requests: row.requests,
    failed: row.failed,
    rate: row.rate,
    ...Object.fromEntries(series.codes.map((code) => [code, errorRate(Number(row[code] ?? 0), row.requests) ?? 0])),
  }));

const hasUnrecordedFailures = (day: RequestErrorDailyEntry): boolean =>
  day.failed_requests > day.by_status_code.reduce((sum, entry) => sum + entry.failed_requests, 0);

export interface StatusCodeRow {
  status_code: number;
  label: string;
  failed_requests: number;
  share: number;
}

export const statusCodeRows = (activity: RequestErrorActivity): StatusCodeRow[] => {
  const recorded = activity.by_status_code.reduce((sum, row) => sum + row.failed_requests, 0);
  const notRecorded = Math.max(0, activity.total_failed_requests - recorded);
  const recordedRows = activity.by_status_code
    .filter((row) => row.failed_requests > 0)
    .map((row) => ({
      status_code: row.status_code,
      label: statusCodeLabel(row.status_code),
      failed_requests: row.failed_requests,
    }));
  const unrecordedRows =
    notRecorded > 0 ? [{ status_code: 0, label: statusCodeLabel(0), failed_requests: notRecorded }] : [];
  const total = activity.total_failed_requests || 1;
  return [...recordedRows, ...unrecordedRows]
    .sort((a, b) => b.failed_requests - a.failed_requests || a.status_code - b.status_code)
    .map((row) => ({ ...row, share: (row.failed_requests / total) * 100 }));
};

export interface ErrorEntityRow {
  id: string;
  name: string;
  requests: number;
  failed: number;
  rate: number | null;
  topStatus: string | null;
}

export const entityRows = (activity: RequestErrorActivity, kind: ErrorEntityKind): ErrorEntityRow[] => {
  const source = { key: activity.by_key, team: activity.by_team, user: activity.by_user, model: activity.by_model }[
    kind
  ];
  return source.map((entry) => ({
    id: entry.id,
    name: entry.label ?? entry.id,
    requests: entry.api_requests,
    failed: entry.failed_requests,
    rate: errorRate(entry.failed_requests, entry.api_requests),
    topStatus:
      entry.top_status_code === null
        ? null
        : `${entry.top_status_code === 0 ? i18n.t("usage.errors.series.noStatus") : entry.top_status_code} · ${entry.top_status_code_requests.toLocaleString()}`,
  }));
};
