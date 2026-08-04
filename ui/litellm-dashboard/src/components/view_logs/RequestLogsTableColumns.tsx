"use client";

import type { ColumnDef } from "@tanstack/react-table";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { CellTooltip, DateCell, IdCell, MoneyCell, StatusBadge } from "@/components/shared/table_cells";
import { getSpendString } from "@/utils/dataUtils";

import { getProviderLogoAndName } from "../provider_info_helpers";
import type { LogEntry } from "./columns";
import { AGENT_CALL_TYPES, MCP_CALL_TYPES } from "./constants";
import { AgentBadge, AgentIcon, LlmBadge, McpBadge, SparkleIcon, WrenchIcon } from "./TypeBadges";

export interface RequestLogsTableColumnsDeps {
  onKeyHashClick: (keyHash: string) => void;
  onSessionClick: (sessionId: string) => void;
}

const readMetaString = (metadata: Record<string, unknown> | undefined, key: string): string | undefined => {
  const value = metadata?.[key];
  return typeof value === "string" && value !== "" ? value : undefined;
};

const readMcpLogoUrl = (metadata: Record<string, unknown> | undefined): string | undefined => {
  const mcpMetadata = metadata?.["mcp_tool_call_metadata"];
  if (typeof mcpMetadata !== "object" || mcpMetadata === null) return undefined;
  const url = (mcpMetadata as Record<string, unknown>)["mcp_server_logo_url"];
  return typeof url === "string" && url !== "" ? url : undefined;
};

const getLogoUrl = (row: LogEntry, provider: string): string =>
  readMcpLogoUrl(row.metadata) ?? (provider ? getProviderLogoAndName(provider).logo : "");

function TruncatedText({ value }: { value: string | undefined }) {
  const display = value ?? "-";
  return <CellTooltip content={display} trigger={<span className="max-w-[15ch] truncate block">{display}</span>} />;
}

export const getRequestLogsTableColumns = ({
  onKeyHashClick,
  onSessionClick,
}: RequestLogsTableColumnsDeps): ColumnDef<LogEntry>[] => [
  {
    id: "startTime",
    accessorKey: "startTime",
    header: ({ column }) => <DataTableSortHeader column={column} title="Time" variant="dropdown-tristate" />,
    size: 200,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.startTime} />,
  },
  {
    id: "type",
    header: "Type",
    size: 90,
    enableSorting: false,
    meta: { skeleton: "badge" },
    cell: ({ row }) => {
      const log = row.original;
      const sessionCount = log.session_total_count || 1;
      const isMcp = MCP_CALL_TYPES.includes(log.call_type);
      const isAgent = AGENT_CALL_TYPES.includes(log.call_type);
      const sessionLlmCount = log.session_llm_count ?? (isMcp || isAgent ? 0 : sessionCount);
      const sessionAgentCount = log.session_agent_count ?? (isAgent ? sessionCount : 0);
      const sessionMcpCount = log.session_mcp_count ?? (isMcp ? sessionCount : 0);

      if (isMcp) return <McpBadge />;
      if (isAgent && sessionCount <= 1) return <AgentBadge />;
      if (sessionCount <= 1) return <LlmBadge />;

      const sessionTypeBadge = (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-blue-50 text-blue-700 border border-blue-200 rounded-full text-[11px] font-medium whitespace-nowrap">
          <SparkleIcon />
          <span>{sessionCount}</span>
          {sessionAgentCount > 0 && (
            <>
              <span className="text-blue-300">·</span>
              <AgentIcon size={10} />
            </>
          )}
          {sessionMcpCount > 0 && (
            <>
              <span className="text-blue-300">·</span>
              <WrenchIcon />
            </>
          )}
        </span>
      );

      const tooltipParts = [
        sessionLlmCount > 0 && `${sessionLlmCount} LLM`,
        sessionAgentCount > 0 && `${sessionAgentCount} Agent`,
        sessionMcpCount > 0 && `${sessionMcpCount} MCP`,
      ].filter(Boolean);
      return <CellTooltip content={tooltipParts.join(" • ")} trigger={sessionTypeBadge} />;
    },
  },
  {
    id: "status",
    header: "Status",
    size: 100,
    enableSorting: false,
    meta: { skeleton: "badge" },
    cell: ({ row }) => {
      const status = readMetaString(row.original.metadata, "status") ?? "Success";
      const isSuccess = status.toLowerCase() !== "failure";
      return <StatusBadge tone={isSuccess ? "success" : "error"} label={isSuccess ? "Success" : "Failure"} />;
    },
  },
  {
    id: "session_id",
    accessorKey: "session_id",
    header: "Session ID",
    size: 120,
    enableSorting: false,
    cell: ({ row }) => <IdCell value={row.original.session_id} onClick={onSessionClick} />,
  },
  {
    id: "request_id",
    accessorKey: "request_id",
    header: "Request ID",
    enableSorting: false,
    cell: ({ row }) => <IdCell value={row.original.request_id} variant="plain" />,
  },
  {
    id: "spend",
    accessorKey: "spend",
    header: ({ column }) => <DataTableSortHeader column={column} title="Cost" variant="dropdown-tristate" />,
    size: 110,
    enableSorting: true,
    meta: { numeric: true, skeleton: "twoLine" },
    cell: ({ row }) => {
      const log = row.original;
      const mcpCount = log.mcp_tool_call_count || 0;
      const mcpSpend = log.mcp_tool_call_spend || 0;
      const isMultiCallSession = (log.session_total_count || 1) > 1;
      const spend = isMultiCallSession && log.session_total_spend != null ? log.session_total_spend : log.spend;
      const money = (
        <span>
          <MoneyCell value={spend} decimals={6} />
        </span>
      );

      return (
        <div className="flex flex-col items-end">
          {spend ? <CellTooltip content={`$${String(spend)}`} trigger={money} /> : money}
          {isMultiCallSession && <span className="text-[10px] text-gray-400">session total</span>}
          {mcpCount > 0 && mcpSpend > 0 && (
            <span className="text-[10px] text-amber-600">
              incl. {getSpendString(mcpSpend)} from {mcpCount} MCP
            </span>
          )}
        </div>
      );
    },
  },
  {
    id: "request_duration_ms",
    accessorKey: "request_duration_ms",
    header: ({ column }) => <DataTableSortHeader column={column} title="Duration (s)" variant="dropdown-tristate" />,
    enableSorting: true,
    meta: { numeric: true },
    cell: ({ row }) => {
      const ms = row.original.request_duration_ms;
      if (ms == null) return <span>-</span>;
      return (
        <CellTooltip
          content={`${ms}ms`}
          trigger={<span className="max-w-[15ch] truncate inline-block">{(ms / 1000).toFixed(2)}</span>}
        />
      );
    },
  },
  {
    id: "ttft_ms",
    accessorKey: "completionStartTime",
    header: ({ column }) => <DataTableSortHeader column={column} title="TTFT (s)" variant="dropdown-tristate" />,
    enableSorting: true,
    meta: { numeric: true },
    cell: ({ row }) => {
      const log = row.original;
      const completionStartTime = log.completionStartTime;
      if (!completionStartTime) return <span>-</span>;
      if (completionStartTime === log.endTime) return <span>-</span>;
      const ttftMs = new Date(completionStartTime).getTime() - new Date(log.startTime).getTime();
      if (ttftMs <= 0) return <span>-</span>;
      return (
        <CellTooltip
          content={`${ttftMs}ms`}
          trigger={<span className="max-w-[15ch] truncate inline-block">{(ttftMs / 1000).toFixed(2)}</span>}
        />
      );
    },
  },
  {
    id: "team_alias",
    header: "Team Name",
    size: 150,
    enableSorting: false,
    cell: ({ row }) => <TruncatedText value={readMetaString(row.original.metadata, "user_api_key_team_alias")} />,
  },
  {
    id: "key_hash",
    header: "Key Hash",
    size: 110,
    enableSorting: false,
    cell: ({ row }) => (
      <IdCell value={readMetaString(row.original.metadata, "user_api_key")} variant="plain" onClick={onKeyHashClick} />
    ),
  },
  {
    id: "key_alias",
    header: "Key Alias",
    size: 150,
    enableSorting: false,
    cell: ({ row }) => <TruncatedText value={readMetaString(row.original.metadata, "user_api_key_alias")} />,
  },
  {
    id: "model",
    accessorKey: "model",
    header: ({ column }) => <DataTableSortHeader column={column} title="Model" variant="dropdown-tristate" />,
    size: 200,
    enableSorting: true,
    cell: ({ row }) => {
      const log = row.original;
      const provider = log.custom_llm_provider;
      const modelName = log.model ?? "";
      return (
        <div className="flex items-center space-x-2">
          {provider && (
            <img
              src={getLogoUrl(log, provider)}
              alt=""
              className="w-4 h-4"
              onError={(event) => {
                event.currentTarget.style.display = "none";
              }}
            />
          )}
          <CellTooltip content={modelName} trigger={<span className="max-w-[15ch] truncate block">{modelName}</span>} />
        </div>
      );
    },
  },
  {
    id: "total_tokens",
    accessorKey: "total_tokens",
    header: ({ column }) => <DataTableSortHeader column={column} title="Tokens" variant="dropdown-tristate" />,
    size: 140,
    enableSorting: true,
    meta: { numeric: true },
    cell: ({ row }) => {
      const log = row.original;
      return (
        <span className="text-sm">
          {String(log.total_tokens || "0")}
          <span className="text-gray-400 text-xs ml-1">
            ({String(log.prompt_tokens || "0")}+{String(log.completion_tokens || "0")})
          </span>
        </span>
      );
    },
  },
  {
    id: "user",
    accessorKey: "user",
    header: "Internal User",
    size: 150,
    enableSorting: false,
    cell: ({ row }) => <TruncatedText value={row.original.user} />,
  },
  {
    id: "end_user",
    accessorKey: "end_user",
    header: "End User",
    size: 140,
    enableSorting: false,
    cell: ({ row }) => <TruncatedText value={row.original.end_user} />,
  },
  {
    id: "request_tags",
    accessorKey: "request_tags",
    header: "Tags",
    size: 150,
    enableSorting: false,
    meta: { skeleton: "chips" },
    cell: ({ row }) => {
      const tags = row.original.request_tags;
      if (!tags || Object.keys(tags).length === 0) return "-";

      const tagEntries = Object.entries(tags);
      const [firstTagKey, firstTagValue] = tagEntries[0];
      const remainingCount = tagEntries.length - 1;

      return (
        <div className="flex flex-wrap gap-1">
          <CellTooltip
            content={
              <div className="flex flex-col gap-1">
                {tagEntries.map(([key, value]) => (
                  <span key={key}>
                    {key}: {String(value)}
                  </span>
                ))}
              </div>
            }
            trigger={
              <span className="px-2 py-1 bg-gray-100 rounded-full text-xs">
                {firstTagKey}: {String(firstTagValue)}
                {remainingCount > 0 && ` +${remainingCount}`}
              </span>
            }
          />
        </div>
      );
    },
  },
];
