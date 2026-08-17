import type { ReactNode } from "react";
import { toast as sonner } from "sonner";
import { ApiError, deriveErrorMessage, extractProxyErrorMessage, unwrapProxyErrorMessage } from "@/lib/http/client";

export type ToastKind = "success" | "info" | "warning" | "error";

export type ToastOptions = {
  readonly description?: ReactNode;
  readonly durationMs?: number;
};

type ErrorFacts = {
  readonly status: number | undefined;
  readonly proxyType: string | undefined;
  readonly text: string;
};

const DEFAULT_DURATION_MS: Readonly<Record<ToastKind, number>> = {
  success: 4000,
  info: 4000,
  warning: 6000,
  error: 6000,
};

/**
 * Titles for the proxy `ProxyErrorTypes` values (litellm/proxy/_types.py) that are set deliberately and say more
 * than the status does. `auth_error` and `internal_server_error` are deliberately absent: the proxy uses them as
 * catch-alls regardless of status (see `handle_exception_on_proxy`), so for those the HTTP status is the truth.
 */
const PROXY_TYPE_TITLES: Readonly<Record<string, string>> = {
  budget_exceeded: "Budget Exceeded",
  no_db_connection: "Service Unavailable",
  expired_key: "Authentication Error",
  token_not_found_in_db: "Authentication Error",
  team_member_permission_error: "Access Denied",
  not_found_error: "Not Found",
  validation_error: "Validation Error",
  bad_request_error: "Request Error",
  team_member_already_in_team: "Already Exists",
};

const STATUS_TITLES: Readonly<Record<number, string>> = {
  400: "Request Error",
  401: "Authentication Error",
  403: "Access Denied",
  404: "Not Found",
  409: "Already Exists",
  422: "Validation Error",
  429: "Rate Limit Exceeded",
  503: "Service Unavailable",
};

const WARNING_TITLES: ReadonlySet<string> = new Set(["Budget Exceeded", "Rate Limit Exceeded"]);

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

/** The proxy wraps errors as `{ error: { message, type, code } }`; bare `{ message, type, code }` also occurs. */
const proxyEnvelope = (payload: unknown): Record<string, unknown> | undefined => {
  const record = asRecord(payload);
  return asRecord(record?.error) ?? record;
};

const proxyTypeOf = (payload: unknown): string | undefined => {
  const type = proxyEnvelope(payload)?.type;
  return typeof type === "string" ? type : undefined;
};

const describeError = (input: unknown): ErrorFacts => {
  if (input instanceof ApiError) {
    return { status: input.status, proxyType: proxyTypeOf(input.body), text: unwrapProxyErrorMessage(input.message) };
  }
  if (input instanceof Error || typeof input === "string") {
    const parsed = parseJson(input instanceof Error ? input.message : input);
    return {
      status: toStatus(proxyEnvelope(parsed)?.code),
      proxyType: proxyTypeOf(parsed),
      text: extractProxyErrorMessage(input),
    };
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
    text: unwrapProxyErrorMessage(deriveErrorMessage(payload)),
  };
};

const titleForStatus = (status: number): string => {
  const known = STATUS_TITLES[status];
  if (known !== undefined) return known;
  if (status >= 500) return "Server Error";
  if (status >= 400) return "Request Error";
  return "Error";
};

const titleFor = ({ status, proxyType }: ErrorFacts): string => {
  if (proxyType?.endsWith("_access_denied")) return "Access Denied";
  const byType = proxyType === undefined ? undefined : PROXY_TYPE_TITLES[proxyType];
  if (byType !== undefined) return byType;
  return status === undefined ? "Error" : titleForStatus(status);
};

const show = (kind: ToastKind, message: ReactNode, options?: ToastOptions): void => {
  sonner[kind](message, {
    description: options?.description,
    duration: options?.durationMs ?? DEFAULT_DURATION_MS[kind],
  });
};

/**
 * The dashboard's single toast surface, backed by sonner (`<Toaster />` in the root layout).
 * `fromError` turns anything a request can reject with (ApiError, Error, raw proxy string, response payload)
 * into a titled toast: the title comes from the proxy error `type` or the HTTP status, never from the prose.
 */
export const toast = {
  success: (message: ReactNode, options?: ToastOptions): void => show("success", message, options),
  info: (message: ReactNode, options?: ToastOptions): void => show("info", message, options),
  warning: (message: ReactNode, options?: ToastOptions): void => show("warning", message, options),
  error: (message: ReactNode, options?: ToastOptions): void => show("error", message, options),
  fromError: (input: unknown, options?: ToastOptions): void => {
    const facts = describeError(input);
    const title = titleFor(facts);
    show(WARNING_TITLES.has(title) ? "warning" : "error", title, { description: facts.text, ...options });
  },
  dismiss: (): void => {
    sonner.dismiss();
  },
} as const;
