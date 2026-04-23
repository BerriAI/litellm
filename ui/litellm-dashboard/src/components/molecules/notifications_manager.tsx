import React from "react";
import { toast } from "sonner";
import { parseErrorMessage } from "../shared/errorUtils";

/**
 * Global notification manager — thin wrapper around sonner's `toast.*` API.
 *
 * Phase-1 shadcn migration: this file previously delegated to antd's
 * `notification.*` API. It now delegates to sonner. The public shape of
 * the module is preserved (same functions + `setNotificationInstance`)
 * so existing call sites don't need to change; `setNotificationInstance`
 * is now a no-op kept for backwards-compatibility.
 */

/**
 * No-op retained for API compatibility with call sites that import
 * `setNotificationInstance` from this module. Sonner is rendered globally via
 * `<Toaster />` in the root layout; no per-tree registration is required.
 */
// eslint-disable-next-line @typescript-eslint/no-unused-vars
export const setNotificationInstance = (_instance: unknown) => {
  // no-op — sonner is singleton
};

/** Kept public for callers that used to spread it into an antd notification config. */
export const COMMON_NOTIFICATION_PROPS = {
  // sonner's equivalent toggles are `richColors` (global on <Toaster>) and
  // `closeButton`; individual toasts don't need these fields.
};

type Placement = "top" | "topLeft" | "topRight" | "bottom" | "bottomLeft" | "bottomRight";

type NotificationConfig = {
  message?: string | React.ReactNode;
  description?: string | React.ReactNode;
  duration?: number;
  placement?: Placement;
  key?: string;
};

type NotificationConfigResolved = Omit<NotificationConfig, "message"> & { message: string | React.ReactNode };

function normalize(input: string | NotificationConfig, fallbackTitle: string): NotificationConfigResolved {
  if (typeof input === "string") return { message: fallbackTitle, description: input };
  return { message: input.message ?? fallbackTitle, ...input };
}

function toIntMaybe(val: unknown): number | undefined {
  if (typeof val === "number") return val;
  if (typeof val === "string" && /^\d+$/.test(val)) return parseInt(val, 10);
  return undefined;
}

const AUTH_MATCH = [
  "invalid api key",
  "invalid authorization header format",
  "authentication error",
  "invalid proxy server token",
  "invalid jwt token",
  "invalid jwt submitted",
  "unauthorized access to metrics endpoint",
];

const FORBIDDEN_MATCH = [
  "admin-only endpoint",
  "not allowed to access model",
  "user does not have permission",
  "access forbidden",
  "invalid credentials used to access ui",
  "user not allowed to access proxy",
];

const DB_MATCH = [
  "db not connected",
  "database not initialized",
  "no db connected",
  "prisma client not initialized",
  "service unhealthy",
];

const ROUTER_MATCH = [
  "no models configured on proxy",
  "llm router not initialized",
  "no deployments available",
  "no healthy deployment available",
  "not allowed to access model due to tags configuration",
  "invalid model name passed in",
];

const RATE_LIMIT_EXTRA = [
  "deployment over user-defined ratelimit",
  "crossed tpm / rpm / max parallel request limit",
  "max parallel request limit",
];

const BUDGET_MATCH = ["budget exceeded", "crossed budget", "provider budget"];

const ENTERPRISE_MATCH = [
  "must be a litellm enterprise user",
  "only be available for liteLLM enterprise users",
  "missing litellm-enterprise package",
  "only available on the docker image",
  "enterprise feature",
  "premium user",
];

const VALIDATION_MATCH = [
  "invalid json payload",
  "invalid request type",
  "invalid key format",
  "invalid hash key",
  "invalid sort column",
  "invalid sort order",
  "invalid limit",
  "invalid file type",
  "invalid field",
  "invalid date format",
];

const NOT_FOUND_MATCH = [
  "model not found",
  "model with id",
  "credential not found",
  "user not found",
  "team not found",
  "organization not found",
  "mcp server with id",
  "tool '",
];

const EXISTS_MATCH = ["already exists", "team member is already in team", "user already exists"];

const GUARDRAIL_MATCH = [
  "violated openai moderation policy",
  "violated jailbreak threshold",
  "violated prompt_injection threshold",
  "violated content safety policy",
  "violated lasso guardrail policy",
  "blocked by pillar security guardrail",
  "violated azure prompt shield guardrail policy",
  "content blocked by model armor",
  "response blocked by model armor",
  "streaming response blocked by model armor",
  "guardrail",
  "moderation",
];

const FILE_UPLOAD_MATCH = [
  "invalid purpose",
  "service must be specified",
  "invalid response - response.response is none",
];

const CLOUDZERO_MATCH = [
  "cloudzero settings not configured",
  "failed to decrypt cloudzero api key",
  "cloudzero settings not found",
];

function titleFor(status?: number, desc?: string): string {
  const d = (desc || "").toLowerCase();

  if (AUTH_MATCH.some((s) => d.includes(s))) return "Authentication Error";
  if (FORBIDDEN_MATCH.some((s) => d.includes(s))) return "Access Denied";
  if (DB_MATCH.some((s) => d.includes(s)) || status === 503) return "Service Unavailable";
  if (BUDGET_MATCH.some((s) => d.includes(s))) return "Budget Exceeded";
  if (ENTERPRISE_MATCH.some((s) => d.includes(s))) return "Feature Unavailable";
  if (ROUTER_MATCH.some((s) => d.includes(s))) return "Routing Error";

  if (EXISTS_MATCH.some((s) => d.includes(s))) return "Already Exists";
  if (GUARDRAIL_MATCH.some((s) => d.includes(s))) return "Content Blocked";

  if (FILE_UPLOAD_MATCH.some((s) => d.includes(s))) return "Validation Error";
  if (CLOUDZERO_MATCH.some((s) => d.includes(s))) return "Integration Error";

  if (VALIDATION_MATCH.some((s) => d.includes(s))) return "Validation Error";
  if (status === 404 || d.includes("not found") || NOT_FOUND_MATCH.some((s) => d.includes(s))) return "Not Found";
  if (
    status === 429 ||
    d.includes("rate limit") ||
    d.includes("tpm") ||
    d.includes("rpm") ||
    RATE_LIMIT_EXTRA.some((s) => d.includes(s))
  )
    return "Rate Limit Exceeded";
  if (status && status >= 500) return "Server Error";
  if (status === 401) return "Authentication Error";
  if (status === 403) return "Access Denied";
  if (d.includes("enterprise") || d.includes("premium")) return "Info";
  if (status && status >= 400) return "Request Error";
  return "Error";
}

const SUCCESS_MATCH = [
  "created successfully",
  "updated successfully",
  "deleted successfully",
  "credential created successfully",
  "model added successfully",
  "team created successfully",
  "user created successfully",
  "organization created successfully",
  "cloudzero settings initialized successfully",
  "cloudzero settings updated successfully",
  "cloudzero export completed successfully",
  "mock llm request made",
  "mock slack alert sent",
  "mock email alert sent",
  "spend for all api keys and teams reset successfully",
  "monthlyglobalspend view refreshed",
  "cache cleared successfully",
  "cache set successfully",
  "ip ",
  "deleted successfully",
];

const INFO_MATCH = ["rate limit reached for deployment", "deployment cooldown period active"];

const DEPRECATION_FEATURE_WARN_MATCH = [
  "this feature is only available for litellm enterprise users",
  "enterprise features are not available",
  "regenerating virtual keys is an enterprise feature",
  "trying to set allowed_routes. this is an enterprise feature",
];

const CONFIG_WARN_MATCH = [
  "invalid maximum_spend_logs_retention_interval value",
  "error has invalid or non-convertible code",
  "failed to save health check to database",
];

function classifyGeneralMessage(desc?: string): { kind: "success" | "info" | "warning"; title: string } | null {
  const d = (desc || "").toLowerCase();

  if (SUCCESS_MATCH.some((s) => d.includes(s))) return { kind: "success", title: "Success" };
  if (DEPRECATION_FEATURE_WARN_MATCH.some((s) => d.includes(s))) return { kind: "warning", title: "Feature Notice" };
  if (CONFIG_WARN_MATCH.some((s) => d.includes(s))) return { kind: "warning", title: "Configuration Warning" };
  if (INFO_MATCH.some((s) => d.includes(s))) return { kind: "warning", title: "Rate Limit" };

  return null;
}

function extractStatus(input: unknown): number | undefined {
  const obj = (input ?? {}) as Record<string, unknown>;
  const response = obj?.response as Record<string, unknown> | undefined;
  return toIntMaybe(response?.status) ?? toIntMaybe(obj?.status_code) ?? toIntMaybe(obj?.code);
}

function extractDescription(input: unknown): string {
  if (typeof input === "string") return input;
  const obj = (input ?? {}) as Record<string, unknown>;
  const response = obj?.response as Record<string, unknown> | undefined;
  const data = response?.data as Record<string, unknown> | undefined;
  const dataError = data?.error as Record<string, unknown> | string | undefined;
  const backendMsg =
    (typeof dataError === "object" ? dataError?.message : dataError) ??
    data?.message ??
    obj?.detail ??
    obj?.message ??
    input;
  return parseErrorMessage(backendMsg);
}

function looksErrorPayload(input: unknown, status?: number): boolean {
  if (status !== undefined) return true;
  if (input instanceof Error) return true;
  if (typeof input === "string") return true;
  if (input && typeof input === "object" && ("error" in input || "detail" in input)) return true;
  return false;
}

/** antd durations are seconds; sonner durations are milliseconds. */
function toMs(d?: number): number | undefined {
  if (d == null) return undefined;
  return d * 1000;
}

/**
 * Sonner accepts a `description` field on every level, so we bundle
 * the (message, description) pair by using message as the main text and
 * description as the subtitle — same visual shape as the old antd stack.
 */
function callToast(
  level: "error" | "warning" | "info" | "success",
  cfg: NotificationConfigResolved,
  defaultDurationSec: number,
) {
  const title =
    typeof cfg.message === "string" ? cfg.message : String(cfg.message ?? "");
  const description =
    typeof cfg.description === "string" || typeof cfg.description === "number"
      ? String(cfg.description)
      : cfg.description
      ? (cfg.description as React.ReactNode)
      : undefined;
  const duration = toMs(cfg.duration ?? defaultDurationSec);
  const opts: Parameters<typeof toast>[1] = { description, duration };
  switch (level) {
    case "error":
      return toast.error(title, opts);
    case "warning":
      return toast.warning(title, opts);
    case "info":
      return toast.info(title, opts);
    case "success":
      return toast.success(title, opts);
  }
}

const NotificationManager = {
  error(input: string | NotificationConfig) {
    const cfg = normalize(input, "Error");
    callToast("error", cfg, 6);
  },

  warning(input: string | NotificationConfig) {
    const cfg = normalize(input, "Warning");
    callToast("warning", cfg, 5);
  },

  info(input: string | NotificationConfig) {
    const cfg = normalize(input, "Info");
    callToast("info", cfg, 4);
  },

  success(input: string | React.ReactNode | NotificationConfig) {
    if (React.isValidElement(input)) {
      toast.success("Success", { description: input, duration: toMs(3.5) });
      return;
    }
    const cfg = normalize(input as string | NotificationConfig, "Success");
    callToast("success", cfg, 3.5);
  },

  fromBackend(input: unknown, extra?: Omit<NotificationConfig, "message" | "description">) {
    const status = extractStatus(input);
    const description = extractDescription(input);
    const base: NotificationConfigResolved = {
      ...(extra ?? {}),
      message: "Info",
      description,
    };

    if (looksErrorPayload(input, status)) {
      const title = titleFor(status, description);
      const payload: NotificationConfigResolved = { ...base, message: title };

      if (
        title === "Rate Limit Exceeded" ||
        title === "Info" ||
        title === "Budget Exceeded" ||
        title === "Feature Unavailable" ||
        title === "Content Blocked" ||
        title === "Integration Error"
      ) {
        callToast("warning", payload, extra?.duration ?? 7);
        return;
      }
      if (title === "Server Error") {
        callToast("error", payload, extra?.duration ?? 8);
        return;
      }
      if (
        title === "Request Error" ||
        title === "Authentication Error" ||
        title === "Access Denied" ||
        title === "Not Found" ||
        title === "Error" ||
        title === "Already Exists"
      ) {
        callToast("error", payload, extra?.duration ?? 6);
        return;
      }
      callToast("info", payload, extra?.duration ?? 4);
      return;
    }

    const cls = classifyGeneralMessage(description);
    const payload: NotificationConfigResolved = { ...base, message: cls?.title ?? "Info" };

    if (cls?.kind === "success") {
      callToast("success", payload, extra?.duration ?? 3.5);
      return;
    }
    if (cls?.kind === "warning") {
      callToast("warning", payload, extra?.duration ?? 6);
      return;
    }
    callToast("info", payload, extra?.duration ?? 4);
  },

  clear() {
    toast.dismiss();
  },
};

export default NotificationManager;
