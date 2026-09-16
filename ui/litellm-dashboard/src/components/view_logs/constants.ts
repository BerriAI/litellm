export const ERROR_CODE_OPTIONS: { label: string; value: string }[] = [
  { label: "400 - Bad Request", value: "400" },
  { label: "401 - Invalid Authentication", value: "401" },
  { label: "403 - Permission Denied", value: "403" },
  { label: "404 - Not Found", value: "404" },
  { label: "408 - Request Timeout", value: "408" },
  { label: "422 - Unprocessable Entity", value: "422" },
  { label: "429 - Rate Limited", value: "429" },
  { label: "500 - Internal Server Error", value: "500" },
  { label: "502 - Bad Gateway", value: "502" },
  { label: "503 - Service Unavailable", value: "503" },
  { label: "529 - Overloaded", value: "529" },
];

/** Call types that represent MCP tool invocations (shared across columns, index, drawer). */
export const MCP_CALL_TYPES = ["call_mcp_tool", "list_mcp_tools"];

/** Call types that represent agent/A2A requests (e.g. asend_message). */
export const AGENT_CALL_TYPES = ["asend_message"];

/** Call types that represent Batch API operations (creation and retrieval, sync and async). */
export const BATCH_CALL_TYPES = ["acreate_batch", "create_batch", "aretrieve_batch", "retrieve_batch"];

const LAST_24_HOURS = { id: "24h", label: "Last 24 Hours", value: 24, unit: "hours" } as const;

export const QUICK_SELECT_OPTIONS = [
  { id: "1m", label: "Last Minute", value: 1, unit: "minutes" },
  { id: "15m", label: "Last 15 Minutes", value: 15, unit: "minutes" },
  { id: "1h", label: "Last Hour", value: 1, unit: "hours" },
  { id: "4h", label: "Last 4 Hours", value: 4, unit: "hours" },
  LAST_24_HOURS,
  { id: "7d", label: "Last 7 Days", value: 7, unit: "days" },
] as const;

export type QuickSelectOption = (typeof QUICK_SELECT_OPTIONS)[number];
export type QuickSelectPresetId = QuickSelectOption["id"];

export const QUICK_SELECT_PRESET_IDS: readonly QuickSelectPresetId[] = QUICK_SELECT_OPTIONS.map((option) => option.id);

export const DEFAULT_QUICK_SELECT_OPTION: QuickSelectOption = LAST_24_HOURS;
export const DEFAULT_QUICK_SELECT_PRESET: QuickSelectPresetId = DEFAULT_QUICK_SELECT_OPTION.id;
