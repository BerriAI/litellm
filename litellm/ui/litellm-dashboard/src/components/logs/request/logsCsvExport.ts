import Papa from "papaparse";

import { downloadCsv } from "@/app/(dashboard)/usage/_components/components/EntityUsage/teamUserSpend";
import { uiSpendLogsCall } from "../../networking";
import type { LogEntry } from "../types";
import type { PaginatedResponse } from "./useLogFilterLogic";

export const LOGS_EXPORT_LIMIT = 10_000;

export async function exportLogsCsv(options: Parameters<typeof uiSpendLogsCall>[0]) {
  const logs: LogEntry[] = [];
  for (let page = 1; logs.length < LOGS_EXPORT_LIMIT; page += 1) {
    const requestOptions = {
      ...options,
      page,
      page_size: 100,
      params: { ...options.params, group_by_session: false, session_cursor: undefined },
    };
    const response: PaginatedResponse = await uiSpendLogsCall(requestOptions);
    logs.push(...response.data.slice(0, LOGS_EXPORT_LIMIT - logs.length));
    if (response.data.length === 0 || page >= response.total_pages) break;
  }
  const csv = Papa.unparse(
    logs.map((log) => ({
      "Start Time": log.startTime,
      Status: log.status ?? "",
      "Request ID": log.request_id,
      Model: log.model,
      Provider: log.custom_llm_provider ?? "",
      "Model Group": log.model_group ?? "",
      "Key Alias": log.metadata?.user_api_key_alias ?? "",
      "Key Hash": log.api_key,
      Team: log.metadata?.user_api_key_team_alias ?? log.team_id,
      User: log.user ?? "",
      "End User": log.end_user ?? "",
      "Prompt Tokens": log.prompt_tokens,
      "Completion Tokens": log.completion_tokens,
      "Total Tokens": log.total_tokens,
      "Spend (USD)": log.spend,
      "Duration (ms)": log.request_duration_ms ?? "",
      "Session ID": log.session_id ?? "",
    })),
    { escapeFormulae: /^[=+\-@\t\r]/ },
  );
  if (logs.length > 0) downloadCsv(csv, `litellm-logs-${new Date().toISOString().slice(0, 10)}.csv`);
  return { count: logs.length, capped: logs.length === LOGS_EXPORT_LIMIT };
}
