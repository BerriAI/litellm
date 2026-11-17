import type { ParseKeys } from "i18next";

export const ERROR_CODE_OPTIONS: { labelKey: ParseKeys; value: string }[] = [
  { labelKey: "logs.common.errorCodes.badRequest", value: "400" },
  { labelKey: "logs.common.errorCodes.invalidAuthentication", value: "401" },
  { labelKey: "logs.common.errorCodes.permissionDenied", value: "403" },
  { labelKey: "logs.common.errorCodes.notFound", value: "404" },
  { labelKey: "logs.common.errorCodes.requestTimeout", value: "408" },
  { labelKey: "logs.common.errorCodes.unprocessableEntity", value: "422" },
  { labelKey: "logs.common.errorCodes.rateLimited", value: "429" },
  { labelKey: "logs.common.errorCodes.internalServerError", value: "500" },
  { labelKey: "logs.common.errorCodes.badGateway", value: "502" },
  { labelKey: "logs.common.errorCodes.serviceUnavailable", value: "503" },
  { labelKey: "logs.common.errorCodes.overloaded", value: "529" },
];

/** Call types that represent MCP tool invocations (shared across columns, index, drawer). */
export const MCP_CALL_TYPES = ["call_mcp_tool", "list_mcp_tools"];

/** Call types that represent agent/A2A requests (e.g. asend_message). */
export const AGENT_CALL_TYPES = ["asend_message"];

/** Call types that represent Batch API operations (creation and retrieval, sync and async). */
export const BATCH_CALL_TYPES = ["acreate_batch", "create_batch", "aretrieve_batch", "retrieve_batch"];

export const SPAN_TYPE_LABELS: Record<string, ParseKeys> = {
  llm: "logs.common.spanTypes.llm",
  agent: "logs.common.spanTypes.agent",
  mcp: "logs.common.spanTypes.mcp",
  batch: "logs.common.spanTypes.batch",
};

export const CREDENTIAL_LABELS: Record<string, ParseKeys> = {
  true: "logs.common.credentials.clientOauthToken",
  false: "logs.common.credentials.configuredKey",
};

export const KUBERNETES_POD_ROUTING_LABELS = {
  round_robin: "logs.common.podRouting.roundRobin",
  session_affinity: "logs.common.podRouting.sessionAffinity",
  session_affinity_retry: "logs.common.podRouting.sessionAffinityRetry",
} as const;

export const QUICK_SELECT_OPTIONS: { labelKey: ParseKeys; value: number; unit: string }[] = [
  { labelKey: "logs.common.quickSelect.lastMinute", value: 1, unit: "minutes" },
  { labelKey: "logs.common.quickSelect.last15Minutes", value: 15, unit: "minutes" },
  { labelKey: "logs.common.quickSelect.lastHour", value: 1, unit: "hours" },
  { labelKey: "logs.common.quickSelect.last4Hours", value: 4, unit: "hours" },
  { labelKey: "logs.common.quickSelect.last24Hours", value: 24, unit: "hours" },
  { labelKey: "logs.common.quickSelect.last7Days", value: 7, unit: "days" },
];
