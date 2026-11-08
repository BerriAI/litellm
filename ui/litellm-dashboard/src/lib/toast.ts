import type { ParseKeys } from "i18next";
import type { ReactNode } from "react";
import { toast as sonner } from "sonner";
import i18n from "@/i18n/i18n";
import { ApiError, deriveErrorMessage, unwrapProxyErrorMessage } from "@/lib/http/client";

export type ToastKind = "success" | "info" | "warning" | "error";

export type ToastOptions = {
  readonly description?: ReactNode;
  readonly durationMs?: number;
  readonly id?: string;
};

type ErrorFacts = {
  readonly status: number | undefined;
  readonly proxyType: string | undefined;
  readonly anthropicType: string | undefined;
  readonly text: string;
};

const DEFAULT_DURATION_MS: Readonly<Record<ToastKind, number>> = {
  success: 4000,
  info: 4000,
  warning: 6000,
  error: 6000,
};

type ToastTitleKey = Extract<ParseKeys, `toast.titles.${string}`>;

const PROXY_TYPE_TITLES: Readonly<Record<string, ToastTitleKey>> = {
  budget_exceeded: "toast.titles.budgetExceeded",
  no_db_connection: "toast.titles.serviceUnavailable",
  expired_key: "toast.titles.authenticationError",
  token_not_found_in_db: "toast.titles.authenticationError",
  team_member_permission_error: "toast.titles.accessDenied",
  not_found_error: "toast.titles.notFound",
  validation_error: "toast.titles.validationError",
  bad_request_error: "toast.titles.requestError",
  team_member_already_in_team: "toast.titles.alreadyExists",
};

const STATUS_TITLES: Readonly<Record<number, ToastTitleKey>> = {
  400: "toast.titles.requestError",
  401: "toast.titles.authenticationError",
  403: "toast.titles.accessDenied",
  404: "toast.titles.notFound",
  409: "toast.titles.alreadyExists",
  422: "toast.titles.validationError",
  429: "toast.titles.rateLimitExceeded",
  503: "toast.titles.serviceUnavailable",
};

const ANTHROPIC_TYPE_TITLES: Readonly<Record<string, ToastTitleKey>> = {
  invalid_request_error: "toast.titles.requestError",
  authentication_error: "toast.titles.authenticationError",
  billing_error: "toast.titles.requestError",
  permission_error: "toast.titles.accessDenied",
  not_found_error: "toast.titles.notFound",
  conflict_error: "toast.titles.alreadyExists",
  request_too_large: "toast.titles.requestError",
  rate_limit_error: "toast.titles.rateLimitExceeded",
  api_error: "toast.titles.serverError",
  timeout_error: "toast.titles.serverError",
  overloaded_error: "toast.titles.serviceUnavailable",
};

const WARNING_TITLES: ReadonlySet<ToastTitleKey> = new Set<ToastTitleKey>([
  "toast.titles.budgetExceeded",
  "toast.titles.rateLimitExceeded",
]);

const asRecord = (value: unknown): Record<string, unknown> | undefined =>
  value !== null && typeof value === "object" ? (value as Record<string, unknown>) : undefined;

const parseJson = (raw: string): unknown => {
  try {
    return JSON.parse(raw);
  } catch {
    return undefined;
  }
};

const toStatus = (value: unknown): number | undefined => {
  if (typeof value === "number") return value;
  if (typeof value === "string" && /^\d{3}$/.test(value)) return Number(value);
  return undefined;
};

const proxyEnvelope = (payload: unknown): Record<string, unknown> | undefined => {
  const record = asRecord(payload);
  return asRecord(record?.error) ?? record;
};

const proxyTypeOf = (payload: unknown): string | undefined => {
  const envelope = proxyEnvelope(payload);
  const type = envelope?.litellm_error_type ?? envelope?.type;
  return typeof type === "string" ? type : undefined;
};

const anthropicTypeOf = (payload: unknown): string | undefined => {
  const record = asRecord(payload);
  const type = asRecord(record?.error)?.type;
  return record?.type === "error" && typeof type === "string" ? type : undefined;
};

const EMBEDDED_JSON = /\{[\s\S]*\}/;
const STATUS_BEFORE_JSON = /(?<!\d)([1-5]\d{2})\s*$/;

const statusBefore = (text: string, embedded: string): number | undefined =>
  toStatus(text.slice(0, text.indexOf(embedded)).match(STATUS_BEFORE_JSON)?.[1]);

const describeText = (text: string): ErrorFacts => {
  const embedded = text.match(EMBEDDED_JSON)?.[0];
  const parsed = embedded === undefined ? undefined : parseJson(embedded);
  if (embedded === undefined || asRecord(parsed) === undefined) {
    return { status: undefined, proxyType: undefined, anthropicType: undefined, text: unwrapProxyErrorMessage(text) };
  }
  const anthropicType = anthropicTypeOf(parsed);
  return {
    status:
      toStatus(proxyEnvelope(parsed)?.code) ?? (anthropicType === undefined ? undefined : statusBefore(text, embedded)),
    proxyType: proxyTypeOf(parsed),
    anthropicType,
    text: text.replace(embedded, unwrapProxyErrorMessage(deriveErrorMessage(parsed))).trim(),
  };
};

const describeError = (input: unknown): ErrorFacts => {
  if (input instanceof ApiError) {
    return {
      status: input.status,
      proxyType: proxyTypeOf(input.body),
      anthropicType: anthropicTypeOf(input.body),
      text: unwrapProxyErrorMessage(input.message),
    };
  }
  if (input instanceof Error || typeof input === "string") {
    return describeText(input instanceof Error ? input.message : input);
  }
  const record = asRecord(input) ?? {};
  const response = asRecord(record.response);
  const payload = asRecord(response?.data) ?? record;
  return {
    status:
      toStatus(response?.status) ??
      toStatus(record.status_code) ??
      toStatus(record.code) ??
      toStatus(proxyEnvelope(payload)?.code),
    proxyType: proxyTypeOf(payload),
    anthropicType: anthropicTypeOf(payload),
    text: unwrapProxyErrorMessage(deriveErrorMessage(payload)),
  };
};

const titleForStatus = (status: number): ToastTitleKey => {
  const known = STATUS_TITLES[status];
  if (known !== undefined) return known;
  if (status >= 500) return "toast.titles.serverError";
  if (status >= 400) return "toast.titles.requestError";
  return "toast.titles.error";
};

const titleFor = ({ status, proxyType, anthropicType }: ErrorFacts): ToastTitleKey => {
  if (proxyType?.endsWith("_access_denied")) return "toast.titles.accessDenied";
  const byType =
    proxyType === undefined || !Object.hasOwn(PROXY_TYPE_TITLES, proxyType) ? undefined : PROXY_TYPE_TITLES[proxyType];
  if (byType !== undefined) return byType;
  if (status !== undefined) return titleForStatus(status);
  return (
    (anthropicType === undefined || !Object.hasOwn(ANTHROPIC_TYPE_TITLES, anthropicType)
      ? undefined
      : ANTHROPIC_TYPE_TITLES[anthropicType]) ?? "toast.titles.error"
  );
};

const show = (kind: ToastKind, message: ReactNode, options?: ToastOptions): void => {
  sonner[kind](message, {
    id: options?.id,
    description: options?.description,
    duration: options?.durationMs ?? DEFAULT_DURATION_MS[kind],
  });
};

export const toast = {
  success: (message: ReactNode, options?: ToastOptions): void => show("success", message, options),
  info: (message: ReactNode, options?: ToastOptions): void => show("info", message, options),
  warning: (message: ReactNode, options?: ToastOptions): void => show("warning", message, options),
  error: (message: ReactNode, options?: ToastOptions): void => show("error", message, options),
  fromError: (input: unknown, options?: ToastOptions): void => {
    const facts = describeError(input);
    const title = titleFor(facts);
    show(WARNING_TITLES.has(title) ? "warning" : "error", i18n.t(title), { description: facts.text, ...options });
  },
  dismiss: (): void => {
    sonner.dismiss();
  },
} as const;
