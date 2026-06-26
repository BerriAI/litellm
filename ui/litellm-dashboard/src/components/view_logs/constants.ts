import type { TFunction } from "i18next";

export const getErrorCodeOptions = (t: TFunction): { label: string; value: string }[] => [
  { label: t("viewLogs.constants.errorCode400"), value: "400" },
  { label: t("viewLogs.constants.errorCode401"), value: "401" },
  { label: t("viewLogs.constants.errorCode403"), value: "403" },
  { label: t("viewLogs.constants.errorCode404"), value: "404" },
  { label: t("viewLogs.constants.errorCode408"), value: "408" },
  { label: t("viewLogs.constants.errorCode422"), value: "422" },
  { label: t("viewLogs.constants.errorCode429"), value: "429" },
  { label: t("viewLogs.constants.errorCode500"), value: "500" },
  { label: t("viewLogs.constants.errorCode502"), value: "502" },
  { label: t("viewLogs.constants.errorCode503"), value: "503" },
  { label: t("viewLogs.constants.errorCode529"), value: "529" },
];

/** Call types that represent MCP tool invocations (shared across columns, index, drawer). */
export const MCP_CALL_TYPES = ["call_mcp_tool", "list_mcp_tools"];

/** Call types that represent agent/A2A requests (e.g. asend_message). */
export const AGENT_CALL_TYPES = ["asend_message"];

export const getQuickSelectOptions = (t: TFunction): { label: string; value: number; unit: string }[] => [
  { label: t("viewLogs.constants.quickSelectLastMinute"), value: 1, unit: "minutes" },
  { label: t("viewLogs.constants.quickSelectLast15Minutes"), value: 15, unit: "minutes" },
  { label: t("viewLogs.constants.quickSelectLastHour"), value: 1, unit: "hours" },
  { label: t("viewLogs.constants.quickSelectLast4Hours"), value: 4, unit: "hours" },
  { label: t("viewLogs.constants.quickSelectLast24Hours"), value: 24, unit: "hours" },
  { label: t("viewLogs.constants.quickSelectLast7Days"), value: 7, unit: "days" },
];
