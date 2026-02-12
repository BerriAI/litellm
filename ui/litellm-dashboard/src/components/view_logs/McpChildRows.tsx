import { useQuery } from "@tanstack/react-query";
import type { Row } from "@tanstack/react-table";
import { Tooltip } from "antd";
import { TableRow, TableCell } from "@tremor/react";
import { getSpendString } from "@/utils/dataUtils";
import { sessionSpendLogsCall } from "../networking";
import type { LogEntry } from "./columns";
import { TimeCell } from "./time_cell";
import { getProviderLogoAndName } from "../provider_info_helpers";

interface SessionChildRowsProps {
  row: Row<LogEntry>;
  accessToken: string;
  onChildClick?: (log: LogEntry) => void;
}

const MCP_CALL_TYPES = ["call_mcp_tool", "list_mcp_tools"];

const LlmBadge = () => (
  <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-blue-50 text-blue-700 border border-blue-200 rounded-full text-[11px] font-medium whitespace-nowrap">
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0 text-gray-400">
      <path d="M12 3l1.912 5.813a2 2 0 0 0 1.275 1.275L21 12l-5.813 1.912a2 2 0 0 0-1.275 1.275L12 21l-1.912-5.813a2 2 0 0 0-1.275-1.275L3 12l5.813-1.912a2 2 0 0 0 1.275-1.275L12 3z" />
    </svg>
    LLM
  </span>
);

const McpBadge = () => (
  <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-amber-50 text-amber-700 border border-amber-200 rounded-full text-[11px] font-medium whitespace-nowrap">
    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0">
      <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
    </svg>
    MCP
  </span>
);

export function SessionChildRows({ row, accessToken, onChildClick }: SessionChildRowsProps) {
  const sessionId = row.original.session_id;

  const { data: children, isLoading } = useQuery({
    queryKey: ["sessionChildren", sessionId],
    queryFn: async () => {
      if (!sessionId) return [];
      const response = await sessionSpendLogsCall(accessToken, sessionId);
      const allLogs: LogEntry[] = response.data || response || [];
      // Sort chronologically
      return allLogs.sort((a, b) =>
        new Date(a.startTime).getTime() - new Date(b.startTime).getTime()
      );
    },
    enabled: !!sessionId && !!accessToken,
    staleTime: 60_000,
  });

  if (isLoading) {
    return (
      <TableRow className="bg-gray-50">
        <TableCell colSpan={99} className="py-2 pl-12 text-sm text-gray-400">
          Loading session calls...
        </TableCell>
      </TableRow>
    );
  }

  if (!children || children.length === 0) {
    return (
      <TableRow className="bg-gray-50">
        <TableCell colSpan={99} className="py-2 pl-12 text-sm text-gray-400">
          No calls found
        </TableCell>
      </TableRow>
    );
  }

  return (
    <>
      {children.map((child) => {
        const isMcp = MCP_CALL_TYPES.includes(child.call_type);
        const modelOrTool = isMcp
          ? (child.model?.replace("MCP: ", "") || "unknown")
          : (child.model || "-");
        const serverName = isMcp
          ? (child.metadata?.mcp_tool_call_metadata?.mcp_server_name || "")
          : "";
        const provider = child.custom_llm_provider || "";
        const logoUrl = isMcp
          ? (child.metadata?.mcp_tool_call_metadata?.mcp_server_logo_url || "")
          : (provider ? getProviderLogoAndName(provider).logo : "");
        const duration =
          child.startTime && child.endTime
            ? ((Date.parse(child.endTime) - Date.parse(child.startTime)) / 1000).toFixed(3)
            : "-";
        const status = (child.metadata?.status || "Success").toLowerCase();
        const isSuccess = status !== "failure";

        return (
          <TableRow
            key={child.request_id}
            className="h-8 bg-gray-50 cursor-pointer hover:bg-gray-100"
            onClick={() => onChildClick?.(child)}
          >
            {/* Expander: branch connector */}
            <TableCell className="py-0.5 max-h-8 overflow-hidden pl-3">
              <div className="flex items-center justify-center text-gray-400">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="text-indigo-400">
                  <path d="M4 0 L4 8 L12 8" stroke="currentColor" strokeWidth="1.5" fill="none" />
                  <path d="M10 5.5 L12.5 8 L10 10.5" stroke="currentColor" strokeWidth="1.5" fill="none" />
                </svg>
              </div>
            </TableCell>

            {/* Time */}
            <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
              <TimeCell utcTime={child.startTime} />
            </TableCell>

            {/* Type */}
            <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
              {isMcp ? <McpBadge /> : <LlmBadge />}
            </TableCell>

            {/* Status */}
            <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
              <span
                className={`px-2 py-1 rounded-md text-xs font-medium inline-block text-center w-16 ${
                  isSuccess ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
                }`}
              >
                {isSuccess ? "Success" : "Failure"}
              </span>
            </TableCell>

            {/* Session ID */}
            <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
              <span className="font-mono text-xs max-w-[15ch] truncate block text-gray-400">
                {child.session_id || ""}
              </span>
            </TableCell>

            {/* Request ID */}
            <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
              <Tooltip title={child.request_id}>
                <span className="font-mono text-xs max-w-[15ch] truncate block">{child.request_id}</span>
              </Tooltip>
            </TableCell>

            {/* Cost */}
            <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
              <span>{getSpendString(child.spend || 0)}</span>
            </TableCell>

            {/* Duration */}
            <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
              <span>{duration}</span>
            </TableCell>

            {/* Team Name */}
            <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
              <span className="max-w-[15ch] truncate block">
                {child.metadata?.user_api_key_team_alias || "-"}
              </span>
            </TableCell>

            {/* Key Hash */}
            <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
              <span className="font-mono max-w-[15ch] truncate block">
                {child.metadata?.user_api_key || "-"}
              </span>
            </TableCell>

            {/* Key Name */}
            <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
              <span className="max-w-[15ch] truncate block">
                {child.metadata?.user_api_key_alias || "-"}
              </span>
            </TableCell>

            {/* Model / Tool */}
            <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
              <div className="flex items-center space-x-2">
                {logoUrl && (
                  <img
                    src={logoUrl}
                    alt=""
                    className="w-4 h-4"
                    onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                  />
                )}
                <Tooltip title={`${modelOrTool}${serverName ? ` (${serverName})` : ""}`}>
                  <span className="max-w-[15ch] truncate block font-semibold">{modelOrTool}</span>
                </Tooltip>
                {serverName && (
                  <span className="text-xs text-gray-400">{serverName}</span>
                )}
              </div>
            </TableCell>

            {/* Tokens */}
            <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
              <span className="text-sm">{child.total_tokens || "0"}</span>
            </TableCell>

            {/* Internal User */}
            <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
              <span className="max-w-[15ch] truncate block">{child.user || "-"}</span>
            </TableCell>

            {/* End User */}
            <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
              <span className="max-w-[15ch] truncate block">{child.end_user || "-"}</span>
            </TableCell>

            {/* Tags */}
            <TableCell className="py-0.5 max-h-8 overflow-hidden text-ellipsis whitespace-nowrap">
              <span>-</span>
            </TableCell>
          </TableRow>
        );
      })}
    </>
  );
}
